"""
optimizer: Handles Bayesian optimization for experiment planning using Atlas/Olympus.

Example usage:

# 1. Basic usage with default bounds
optimizer = SDLOptimizer(
    initial_experiment_ids=[1, 2, 3, 4, 5],
    goal='minimize'
)

# 2. Custom parameter bounds (equipment limits)
custom_bounds = ParameterBounds(
    dispense=DispenseBounds(
        conc_ch1=(0.05, 2.0),
        conc_ch2=(0.05, 2.0),
        purge_time=(5000, 120000),
        dispensing_time=(2000, 15000)
    ),
    heat_treatment=HeatTreatmentBounds(
        target_temp=(150, 700),
        holding_time=(1, 120),
        ramp_rate=(5, 150)
    )
)

optimizer = SDLOptimizer(
    initial_experiment_ids=[1, 2, 3, 4, 5],
    parameter_bounds=custom_bounds,
    goal='minimize'
)

# 3. Combine custom bounds with fixed parameters
optimizer = SDLOptimizer(
    initial_experiment_ids=[1, 2, 3, 4, 5],
    parameter_bounds=custom_bounds,
    fixed_parameters={
        'dispense_0_conc_ch1': 0.5,  # Fix concentration at 0.5
        'furnace_0_target_temp': 350  # Fix temperature at 350°C
    },
    goal='minimize'
)

# Get suggestions (fixed parameters automatically included)
suggestions = optimizer.suggest_next_experiments(num_suggestions=3)
"""
from __future__ import annotations

import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
from numbers import Number
from dataclasses import dataclass, field
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import (
    ParameterContinuous,
    ParameterDiscrete,
    ParameterCategorical,
    ParameterVector,
)
# Handle Atlas/Olympus version compatibility
try:
    from atlas.planners.gp.planner import GPPlanner
    ATLAS_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"⚠️  Atlas import failed: {e}")
    print("   Falling back to Olympus-only implementation...")
    
    # Fallback: Use Olympus Planner directly
    try:
        from olympus.planners import Planner as GPPlanner
        ATLAS_AVAILABLE = False
        print("✅ Using Olympus Planner as fallback.")
    except ImportError:
        print("❌ Neither Atlas nor Olympus Planner could be imported.")
        raise ImportError("Could not import optimization planner. Please check Atlas/Olympus installation.")
from database import session, Experiment, EidTemplate, SECCMResult, Process



@dataclass
class DispenseBounds:
    """Parameter bounds for DISPENSE process."""
    conc_ch1: Tuple[float, float] = (0.1, 1.0)
    conc_ch2: Tuple[float, float] = (0.1, 1.0)
    purge_time: Tuple[float, float] = (1000, 60000)  # ms
    dispensing_time: Tuple[float, float] = (1000, 10000)  # ms


@dataclass
class PreHeatBounds:
    """Parameter bounds for PRE_HEAT process."""
    heating_time: Tuple[float, float] = (60, 3600)  # seconds


@dataclass
class HeatTreatmentBounds:
    """Parameter bounds for HEAT_TREATMENT process."""
    target_temp: Tuple[float, float] = (100, 600)  # °C
    holding_time: Tuple[float, float] = (5, 3000)  # minutes
    ramp_rate: Tuple[float, float] = (5, 100)  # °C/min
    cooling_time: Tuple[float, float] = (60, 3600)  # seconds


@dataclass
class PressWorkBounds:
    """Parameter bounds for PRESS_WORK process."""
    position: Tuple[float, float] = (100, 200)
    holding_time: Tuple[float, float] = (1, 10)  # minutes


@dataclass
class ParameterBounds:
    """
    Complete parameter bounds configuration for all process types.
    
    Example:
        bounds = ParameterBounds(
            dispense=DispenseBounds(
                conc_ch1=(0.05, 2.0),
                conc_ch2=(0.05, 2.0),
                purge_time=(5000, 120000),
                dispensing_time=(2000, 15000)
            ),
            heat_treatment=HeatTreatmentBounds(
                target_temp=(150, 700),
                holding_time=(1, 120),
                ramp_rate=(5, 150)
            )
        )
    """
    dispense: DispenseBounds = field(default_factory=DispenseBounds)
    pre_heat: PreHeatBounds = field(default_factory=PreHeatBounds)
    heat_treatment: HeatTreatmentBounds = field(default_factory=HeatTreatmentBounds)
    press_work: PressWorkBounds = field(default_factory=PressWorkBounds)


class SDLOptimizer:
    """
    Bayesian optimizer for Self-Driving Laboratory using Atlas/Olympus.
    
    Integrates with SDL database to:
    - Start from existing experiments as initial training data
    - Check process sequence compatibility between experiments
    - Build parameter space from actual experimental variations
    - Suggest next experiments for autonomous optimization
    """
    
    def __init__(self, 
                 initial_experiment_ids: Optional[List[int]] = None,
                 process_sequence: Optional[List[str]] = None,
                 parameter_ranges: Optional[Dict] = None,
                 parameter_bounds: Optional[ParameterBounds] = None,
                 fixed_parameters: Optional[Dict] = None,
                 goal: str = 'minimize',
                 init_design_strategy: str = 'random',
                 num_init_design: int = 5,
                 batch_size: int = 1,
                 acquisition_optimizer_kind: str = 'gradient',
                 known_constraints: Optional[List[Callable]] = None,
                 campaign_name: Optional[str] = None):
        """
        Initialize SDL Bayesian optimizer.
        
        Two initialization modes:
        1. From existing experiments (recommended)
        2. From scratch with defined process sequence
        
        Args:
            initial_experiment_ids: List of experiment database IDs to start with
            process_sequence: List of process types (if starting from scratch)
            parameter_ranges: Dict of parameter ranges (if starting from scratch)
            parameter_bounds: ParameterBounds dataclass for custom equipment limits
                              If not provided, uses sensible defaults
            fixed_parameters: Dict of parameters to fix at specific values (not optimized)
                              Accepts either fully-qualified keys
                              ('dispense_0_purge_time') or base aliases shared
                              across all repetitions (e.g., {'PURGE_TIME': 30000})
            goal: 'minimize' or 'maximize' the objective
            init_design_strategy: 'random', 'lhs', 'sobol'
            num_init_design: Number of initial random experiments (if no initial data)
            batch_size: Number of experiments to suggest per iteration
            acquisition_optimizer_kind: 'gradient', 'pymoo', 'genetic'
            known_constraints: List of constraint functions
            campaign_name: Name for this optimization campaign (auto-generated if None)
        """
        self.goal = goal
        self.compatible_experiments = []
        self.reference_process_sequence = None
        self.fixed_parameters = {}
        self.fixed_parameter_aliases = {}
        self.dispense_dual_channel_constraints = []
        self.dual_channel_constraint_tolerance = 1e-3
        self._initialize_fixed_parameters(fixed_parameters or {})
        self.parameter_bounds = parameter_bounds or ParameterBounds()  # Use defaults if not provided
        
        # Ensure known_constraints is always a list (handle single function or None)
        if known_constraints is None:
            self.known_constraints = []
        elif callable(known_constraints) and not isinstance(known_constraints, (list, tuple)):
            # Single function passed - wrap in list
            self.known_constraints = [known_constraints]
        else:
            # Already a list/iterable
            self.known_constraints = list(known_constraints) if known_constraints else []
        
        # Campaign management
        from datetime import datetime
        from database import get_next_campaign_number
        
        if campaign_name:
            self.campaign_name = campaign_name
        else:
            # Generate date-based campaign name: YYMMDD_NN
            date_str = datetime.now().strftime('%y%m%d')
            campaign_num = get_next_campaign_number(date_str)
            self.campaign_name = f"{date_str}_{campaign_num}"
        
        self.campaign_iteration = 0  # Track which iteration we're on (4-digit)
        self.created_templates = []  # Track all templates created in this campaign
        
        # Mode 1: Initialize from existing experiments
        if initial_experiment_ids:
            print(f"🚀 Initializing optimizer from {len(initial_experiment_ids)} existing experiments")
            self._initialize_from_experiments(initial_experiment_ids)
            
        # Mode 2: Initialize from scratch  
        elif process_sequence and parameter_ranges:
            print(f"🚀 Initializing optimizer from scratch with process sequence: {process_sequence}")
            self._initialize_from_scratch(process_sequence, parameter_ranges)
            
        else:
            raise ValueError("Must provide either 'initial_experiment_ids' or both 'process_sequence' and 'parameter_ranges'")
        
        # Configure automatic constraints (e.g., dual-channel dispense sums)
        self._configure_dispense_channel_constraints()
        
        # Initialize planner (Atlas or Olympus fallback)
        if ATLAS_AVAILABLE:
            # Ensure known_constraints is always a list for Atlas
            if not isinstance(self.known_constraints, list):
                self.known_constraints = [self.known_constraints] if self.known_constraints else []
            
            # Debug: verify constraints format before passing to Atlas
            print(f"🔍 Passing {len(self.known_constraints)} constraint(s) to Atlas GPPlanner")
            if self.known_constraints:
                for i, c in enumerate(self.known_constraints):
                    print(f"   Constraint {i}: {type(c).__name__} - {getattr(c, '__name__', 'unnamed')}")
            
            # Use Atlas GPPlanner
            # NOTE: Atlas has a bug with constraint handling - we'll enforce constraints
            # manually after getting suggestions instead of passing them to Atlas
            print(f"⚠️  Atlas constraint handling has issues - will enforce constraints manually")
            print(f"   Storing {len(self.known_constraints)} constraint(s) for manual enforcement")
            
            # Store constraints separately but don't pass to Atlas (workaround for Atlas bug)
            self._manual_constraints = list(self.known_constraints) if self.known_constraints else []
            self.known_constraints = []  # Clear to avoid passing to Atlas
            
            self.planner = GPPlanner(
                goal=goal,
                init_design_strategy=init_design_strategy,
                num_init_design=num_init_design,
                batch_size=batch_size,
                acquisition_optimizer_kind=acquisition_optimizer_kind,
                known_constraints=None,  # Don't pass constraints to Atlas
                use_descriptors=True
            )
        else:
            # Use Olympus Planner (different API)
            # Store manual constraints for Olympus too (if any)
            self._manual_constraints = list(self.known_constraints) if self.known_constraints else []
            from olympus.planners import Planner
            # Try different GP-based planners available in this Olympus version
            gp_planners = ['Gpyopt', 'Botorch', 'Dragonfly']  # GP-based planners
            planner_used = None
            
            for planner_name in gp_planners:
                try:
                    self.planner = Planner(kind=planner_name, goal=goal)
                    planner_used = planner_name
                    break
                except Exception as e:
                    continue
            
            if planner_used:
                print(f"ℹ️  Using Olympus {planner_used} Planner (some advanced features may not be available)")
            else:
                # Fallback to a simple planner
                self.planner = Planner(kind='RandomSearch', goal=goal)
                print("ℹ️  Using Olympus RandomSearch Planner as fallback (limited optimization capability)")
        self.planner.set_param_space(self.param_space)
        
        # Initialize Olympus campaign
        self.campaign = Campaign()
        self.campaign.set_param_space(self.param_space)
        
        # Load training data
        if self.compatible_experiments:
            self._load_training_data_from_experiments()
        
        print(f"✅ SDL optimizer initialized successfully")
        print(f"🏷️  Campaign: {self.campaign_name}")
        print(f"📊 Parameter space: {len(self.param_space)} parameters (optimizable)")
        total_fixed = len(self.fixed_parameters) + len(self.fixed_parameter_aliases)
        if total_fixed:
            fixed_keys = list(self.fixed_parameters.keys())
            if self.fixed_parameter_aliases:
                fixed_keys.extend([f"{alias} (all)" for alias in self.fixed_parameter_aliases.keys()])
            print(f"🔒 Fixed parameters: {total_fixed} ({', '.join(fixed_keys)})")
        print(f"🎯 Goal: {goal}")
        print(f"📈 Training observations: {len(self.campaign.observations.get_values())}")
        print(f"🔗 Compatible experiments: {len(self.compatible_experiments)}")

    def _initialize_from_experiments(self, initial_experiment_ids: List[int]):
        """
        Initialize optimizer from existing experiments.
        
        Args:
            initial_experiment_ids: List of experiment database IDs
        """
        print(f"🔍 Loading experiments: {initial_experiment_ids}")
        
        # Get all experiments from database
        experiments = []
        for exp_id in initial_experiment_ids:
            experiment = session.query(Experiment).filter_by(id=exp_id).first()
            if not experiment:
                print(f"⚠️ Experiment {exp_id} not found in database")
                continue
            experiments.append(experiment)
        
        if not experiments:
            raise ValueError("No valid experiments found")
        
        # Use first experiment as reference for process sequence
        reference_experiment = experiments[0]
        self.reference_process_sequence = self._extract_process_sequence(reference_experiment)
        print(f"📋 Reference process sequence: {self.reference_process_sequence}")
        
        # Create parameter space from process sequence using predefined boundaries
        # This ensures consistent search space based on equipment limits, not learned from data
        self.param_space = self._create_parameter_space_from_process_sequence(self.reference_process_sequence)
        
        # After creating initial parameter space, refine it based on actually used parameters
        self._refine_parameter_space_from_experiments()
        
        # Check compatibility of all experiments with the reference sequence
        self.compatible_experiments = [reference_experiment]  # Reference is always compatible
        
        for experiment in experiments[1:]:
            exp_sequence = self._extract_process_sequence(experiment)
            if self._sequences_compatible(self.reference_process_sequence, exp_sequence):
                self.compatible_experiments.append(experiment)
                print(f"✅ Experiment {experiment.id} compatible: {exp_sequence}")
            else:
                print(f"❌ Experiment {experiment.id} incompatible: {exp_sequence}")
        
        print(f"📊 Using {len(self.compatible_experiments)}/{len(experiments)} compatible experiments as training data")
        
        # Track initial experiments in BO Campaign database
        from database import add_bo_campaign_record
        print(f"🗄️  Adding initial experiments to BO Campaign database...")
        
        for experiment in self.compatible_experiments:
            try:
                # Use the actual EID template name from the experiment
                actual_template_name = None
                if experiment.eid_template and experiment.eid_template.template_name:
                    actual_template_name = experiment.eid_template.template_name
                else:
                    # Fallback if no template found
                    actual_template_name = f"Unknown_Template_Exp_{experiment.id}"
                
                add_bo_campaign_record(
                    campaign_name=self.campaign_name,
                    eid_template_name=actual_template_name,  # Use actual EID template name
                    experiment_id=experiment.id,
                    exp_id=experiment.exp_id
                )
                print(f"   ✅ Tracked experiment {experiment.exp_id} (template: {actual_template_name}) in campaign {self.campaign_name}")
            except Exception as e:
                print(f"   ⚠️ Could not track experiment {experiment.exp_id}: {e}")
        
        print(f"🗄️  Added {len(self.compatible_experiments)} initial experiments to BO Campaign database")

    def _initialize_from_scratch(self, process_sequence: List[str], parameter_ranges: Dict):
        """
        Initialize optimizer from scratch with defined process sequence.
        
        Args:
            process_sequence: List of process types
            parameter_ranges: Dict of parameter ranges
        """
        self.reference_process_sequence = process_sequence
        self.compatible_experiments = []  # No existing experiments
        
        # Create parameter space from provided ranges
        self.param_space = self._create_parameter_space_from_ranges(parameter_ranges)
        
        print(f"📋 Process sequence: {process_sequence}")
        print(f"📊 Parameter ranges: {len(parameter_ranges)} parameters")

    def _extract_process_sequence(self, experiment: Experiment) -> List[str]:
        """
        Extract the enabled process sequence from an experiment's EID template.
        
        Args:
            experiment: Experiment object from database
            
        Returns:
            List of enabled process types in order
        """
        if not experiment.eid_template:
            raise ValueError(f"Experiment {experiment.id} has no EID template")
        
        process_config = experiment.eid_template.process_config
        enabled_sequence = []
        
        for process in process_config:
            if process.get('USE', 'False').lower() == 'true':
                enabled_sequence.append(process.get('TYPE'))
        
        return enabled_sequence

    def _sequences_compatible(self, seq1: List[str], seq2: List[str]) -> bool:
        """
        Check if two process sequences are compatible for same optimization model.
        
        Args:
            seq1: First process sequence
            seq2: Second process sequence
            
        Returns:
            True if sequences are compatible
        """
        return seq1 == seq2  # For now, require exact match

    def _create_parameter_space_from_ranges(self, parameter_ranges: Dict) -> ParameterSpace:
        """
        Create parameter space from explicitly provided ranges.
        
        Args:
            parameter_ranges: Dict mapping parameter names to (low, high) tuples
            
        Returns:
            ParameterSpace object for Atlas/Olympus
        """
        param_space = ParameterSpace()
        
        for param_name, (low, high) in parameter_ranges.items():
            param_space.add(ParameterContinuous(
                name=param_name,
                low=low,
                high=high,
                description=f'Parameter {param_name}'
            ))
        
        return param_space

    def _configure_dispense_channel_constraints(self):
        """
        Detect DISPENSE steps that use both channels and enforce sum-to-one constraints.
        """
        self.dispense_dual_channel_constraints = []
        
        if not getattr(self, 'param_space', None):
            return
        
        if not self.compatible_experiments:
            return
        
        dual_channel_indices = set()
        
        for experiment in self.compatible_experiments:
            process_type_counters = {}
            processes = session.query(Process).filter_by(
                experiment_id=experiment.id
            ).order_by(Process.sequence_order).all()
            
            for process in processes:
                process_type = process.process_type
                if process_type not in process_type_counters:
                    process_type_counters[process_type] = 0
                
                type_idx = process_type_counters[process_type]
                process_type_counters[process_type] += 1
                
                if process_type == 'DISPENSE':
                    settings = process.settings or {}
                    use_ch1 = str(settings.get('USE_CH1', 'False')).lower() == 'true'
                    use_ch2 = str(settings.get('USE_CH2', 'False')).lower() == 'true'
                    
                    if use_ch1 and use_ch2:
                        dual_channel_indices.add(type_idx)
        
        if not dual_channel_indices:
            return
        
        param_index_lookup = {param.name: idx for idx, param in enumerate(self.param_space)}
        
        for idx in sorted(dual_channel_indices):
            name_ch1 = f'dispense_{idx}_conc_ch1'
            name_ch2 = f'dispense_{idx}_conc_ch2'
            
            if name_ch1 not in param_index_lookup or name_ch2 not in param_index_lookup:
                continue
            
            constraint = self._build_sum_constraint(
                param_index_lookup[name_ch1],
                param_index_lookup[name_ch2],
                name_ch1,
                name_ch2
            )
            
            if constraint:
                # Ensure known_constraints is a list before appending
                if not isinstance(self.known_constraints, list):
                    self.known_constraints = [self.known_constraints] if self.known_constraints else []
                self.known_constraints.append(constraint)
                self.dispense_dual_channel_constraints.append((name_ch1, name_ch2))
                print(f"🔗 Enforcing constraint: {name_ch1} + {name_ch2} = 1.0")

    def _normalize_dual_channel_constraints(self, param_dict: Dict):
        """
        Normalize dual-channel dispense concentrations to satisfy sum-to-one constraint.
        
        Args:
            param_dict: Parameter dictionary to normalize in-place
        """
        if not hasattr(self, 'dispense_dual_channel_constraints'):
            return
        
        tolerance = getattr(self, 'dual_channel_constraint_tolerance', 1e-3)
        for name_ch1, name_ch2 in self.dispense_dual_channel_constraints:
            if name_ch1 in param_dict and name_ch2 in param_dict:
                val1 = float(param_dict[name_ch1])
                val2 = float(param_dict[name_ch2])
                total = val1 + val2
                
                # Normalize if sum is not 1.0 (within tolerance)
                if abs(total - 1.0) > tolerance:
                    if total > 0:
                        # Scale both proportionally
                        param_dict[name_ch1] = val1 / total
                        param_dict[name_ch2] = val2 / total
                        print(f"🔧 Normalized {name_ch1} + {name_ch2} from {total:.4f} to 1.0")
                    else:
                        # Both zero - set to equal split
                        param_dict[name_ch1] = 0.5
                        param_dict[name_ch2] = 0.5
                        print(f"🔧 Set {name_ch1} + {name_ch2} to 0.5 + 0.5 (both were zero)")

    def _check_manual_constraints(self, param_dict: Dict) -> bool:
        """
        Check if parameter dictionary satisfies manual constraints.
        
        Args:
            param_dict: Parameter dictionary from suggestion
            
        Returns:
            True if all constraints are satisfied
        """
        if not hasattr(self, '_manual_constraints') or not self._manual_constraints:
            return True
        
        # Check dual-channel dispense constraints
        if hasattr(self, 'dispense_dual_channel_constraints'):
            tolerance = getattr(self, 'dual_channel_constraint_tolerance', 1e-3)
            for name_ch1, name_ch2 in self.dispense_dual_channel_constraints:
                if name_ch1 in param_dict and name_ch2 in param_dict:
                    val1 = float(param_dict[name_ch1])
                    val2 = float(param_dict[name_ch2])
                    if abs((val1 + val2) - 1.0) > tolerance:
                        return False
        
        # Add other constraint checks here if needed
        return True

    def _build_sum_constraint(self, idx1: int, idx2: int, name1: str, name2: str):
        """
        Create a constraint function enforcing param[idx1] + param[idx2] == 1.
        
        Handles both numpy arrays and torch tensors (for Atlas compatibility).
        """
        tolerance = getattr(self, 'dual_channel_constraint_tolerance', 1e-3)
        
        def constraint(params):
            try:
                # Handle torch tensors (Atlas uses PyTorch internally)
                if hasattr(params, 'cpu'):  # torch tensor
                    params_np = params.detach().cpu().numpy()
                    val1 = float(params_np[idx1])
                    val2 = float(params_np[idx2])
                # Handle numpy arrays
                elif hasattr(params, '__getitem__'):
                    val1 = float(params[idx1])
                    val2 = float(params[idx2])
                else:
                    return False
                
                return abs((val1 + val2) - 1.0) <= tolerance
            except (TypeError, ValueError, IndexError, AttributeError):
                return False
        
        constraint.__name__ = f"sum_constraint_{name1}_{name2}"
        return constraint

    def _initialize_fixed_parameters(self, fixed_parameters: Dict):
        """
        Normalize user-provided fixed parameters.
        
        Supports two formats:
            1. Fully-qualified parameter names (e.g., 'dispense_0_purge_time')
            2. Base aliases applied across all repetitions (e.g., 'PURGE_TIME')
        """
        for raw_key, value in fixed_parameters.items():
            if raw_key is None:
                continue
            key_str = str(raw_key)
            if any(char.isdigit() for char in key_str):
                normalized = key_str.lower()
                self.fixed_parameters[normalized] = value
            else:
                alias = key_str.upper()
                self.fixed_parameter_aliases[alias] = value
    
    def _get_base_param_key(self, param_name: str) -> str:
        """Return the base alias key (e.g., 'PURGE_TIME') for a parameter."""
        parts = param_name.split('_')
        if len(parts) <= 2:
            return param_name.upper()
        base_parts = parts[2:]
        return '_'.join(base_parts).upper()
    
    def _is_parameter_fixed(self, param_name: str) -> bool:
        """Check whether a parameter should be fixed (exact or alias match)."""
        normalized = param_name.lower()
        if normalized in self.fixed_parameters:
            return True
        
        base_key = self._get_base_param_key(normalized)
        if base_key in self.fixed_parameter_aliases:
            value = self.fixed_parameter_aliases[base_key]
            self.fixed_parameters[normalized] = value
            return True
        
        return False

    def _add_parameter_if_not_fixed(self, param_space: ParameterSpace, name: str, 
                                     low: float, high: float, description: str):
        """
        Add parameter to space only if it's not in fixed_parameters.
        
        Args:
            param_space: ParameterSpace to add to
            name: Parameter name
            low: Lower bound
            high: Upper bound
            description: Parameter description
        """
        if not self._is_parameter_fixed(name):
            param_space.add(ParameterContinuous(
                name=name,
                low=low,
                high=high,
                description=description
            ))
        else:
            fixed_value = self.fixed_parameters.get(name.lower())
            print(f"🔒 Skipping fixed parameter: {name} = {fixed_value}")

    def _create_parameter_space_from_process_sequence(self, process_sequence: List[str]) -> ParameterSpace:
        """
        Create parameter space from process sequence using predefined physical boundaries.
        
        This defines the search space based on equipment limits, not learned from data.
        Fixed parameters (from self.fixed_parameters) are excluded from the parameter space.
        
        Args:
            process_sequence: Ordered list of process types (e.g., ['DISPENSE', 'HEAT_TREATMENT'])
            
        Returns:
            ParameterSpace object with predefined boundaries for each process type
        """
        param_space = ParameterSpace()
        
        # Track process type occurrences for naming (e.g., dispense_0, dispense_1)
        process_type_counters = {}
        
        for process_type in process_sequence:
            # Get counter for this process type
            if process_type not in process_type_counters:
                process_type_counters[process_type] = 0
            
            type_idx = process_type_counters[process_type]
            process_type_counters[process_type] += 1
            
            # Define parameter ranges based on process type (using configured bounds)
            if process_type == 'DISPENSE':
                bounds = self.parameter_bounds.dispense
                self._add_parameter_if_not_fixed(
                    param_space, f'dispense_{type_idx}_conc_ch1',
                    bounds.conc_ch1[0], bounds.conc_ch1[1],
                    f'Dispense {type_idx} CH1 concentration'
                )
                self._add_parameter_if_not_fixed(
                    param_space, f'dispense_{type_idx}_conc_ch2',
                    bounds.conc_ch2[0], bounds.conc_ch2[1],
                    f'Dispense {type_idx} CH2 concentration'
                )
                self._add_parameter_if_not_fixed(
                    param_space, f'dispense_{type_idx}_purge_time',
                    bounds.purge_time[0], bounds.purge_time[1],
                    f'Dispense {type_idx} purge time (ms)'
                )
                self._add_parameter_if_not_fixed(
                    param_space, f'dispense_{type_idx}_dispensing_time',
                    bounds.dispensing_time[0], bounds.dispensing_time[1],
                    f'Dispense {type_idx} dispensing time (ms)'
                )
                
            elif process_type == 'PRE_HEAT':
                bounds = self.parameter_bounds.pre_heat
                self._add_parameter_if_not_fixed(
                    param_space, f'preheat_{type_idx}_heating_time',
                    bounds.heating_time[0], bounds.heating_time[1],
                    f'Pre-heat {type_idx} heating time (s)'
                )
                
            elif process_type == 'HEAT_TREATMENT':
                bounds = self.parameter_bounds.heat_treatment
                self._add_parameter_if_not_fixed(
                    param_space, f'furnace_{type_idx}_target_temp',
                    bounds.target_temp[0], bounds.target_temp[1],
                    f'Furnace {type_idx} target temperature (°C)'
                )
                self._add_parameter_if_not_fixed(
                    param_space, f'furnace_{type_idx}_holding_time',
                    bounds.holding_time[0], bounds.holding_time[1],
                    f'Furnace {type_idx} holding time (min)'
                )
                self._add_parameter_if_not_fixed(
                    param_space, f'furnace_{type_idx}_ramp_rate',
                    bounds.ramp_rate[0], bounds.ramp_rate[1],
                    f'Furnace {type_idx} temperature ramp rate (°C/min)'
                )
                self._add_parameter_if_not_fixed(
                    param_space, f'furnace_{type_idx}_cooling_time',
                    bounds.cooling_time[0], bounds.cooling_time[1],
                    f'Furnace {type_idx} cooling time (s)'
                )
                
            elif process_type == 'PRESS_WORK':
                bounds = self.parameter_bounds.press_work
                self._add_parameter_if_not_fixed(
                    param_space, f'press_{type_idx}_position',
                    bounds.position[0], bounds.position[1],
                    f'Press {type_idx} position'
                )
                self._add_parameter_if_not_fixed(
                    param_space, f'press_{type_idx}_holding_time',
                    bounds.holding_time[0], bounds.holding_time[1],
                    f'Press {type_idx} holding time (min)'
                )
            
            else:
                print(f"⚠️ Unknown process type '{process_type}' - no parameters added")
            
            print(f"📊 Added parameters for {process_type} (index {type_idx})")
        
        return param_space

    def _refine_parameter_space_from_experiments(self):
        """
        Refine parameter space by removing parameters that are never used in experiments.
        
        For DISPENSE processes, this removes CH1/CH2 concentration parameters if the 
        corresponding USE_CH1/USE_CH2 flags are never True in any experiment, or if
        only one channel is ever used (making concentration fixed at 1.0).
        """
        if not self.compatible_experiments:
            return
        
        # Collect all parameters that are actually used across all experiments
        used_parameters = set()
        fixed_single_channel_params = set()
        
        for experiment in self.compatible_experiments:
            params = self._extract_parameters_from_experiment(experiment)
            if params:
                used_parameters.update(params.keys())
                
                # Check for single-channel dispense processes (concentration fixed at 1.0)
                for param_name, param_value in params.items():
                    if ('_conc_ch1' in param_name or '_conc_ch2' in param_name) and param_value == 1.0:
                        # This is likely a single-channel case - check if it's always 1.0
                        fixed_single_channel_params.add(param_name)
        
        # Check if single-channel parameters are always fixed at 1.0 across all experiments
        truly_fixed_params = set()
        for param_name in fixed_single_channel_params:
            is_always_fixed = True
            for experiment in self.compatible_experiments:
                params = self._extract_parameters_from_experiment(experiment)
                if params and param_name in params:
                    if params[param_name] != 1.0:
                        is_always_fixed = False
                        break
            
            if is_always_fixed:
                truly_fixed_params.add(param_name)
                # Add to fixed_parameters so it gets included in suggestions
                self.fixed_parameters[param_name.lower()] = 1.0
                print(f"🔒 Detected always-fixed single-channel parameter: {param_name} = 1.0")
        
        # Remove unused parameters and truly fixed single-channel parameters from parameter space
        params_to_remove = []
        for param in self.param_space:
            param_fixed = self._is_parameter_fixed(param.name)
            if ((param.name not in used_parameters and not param_fixed) or
                param.name in truly_fixed_params):
                params_to_remove.append(param.name)
        
        for param_name in params_to_remove:
            # Find and remove the parameter
            for i, param in enumerate(self.param_space):
                if param.name == param_name:
                    del self.param_space[i]
                    if param_name in truly_fixed_params:
                        print(f"🔒 Removed fixed single-channel parameter: {param_name} (always 1.0)")
                    else:
                        print(f"🚫 Removed unused parameter: {param_name}")
                    break

    def _load_training_data_from_experiments(self):
        """
        Load X,Y training data from compatible experiments.
        
        Fixed parameters are filtered out before adding to observations.
        """
        loaded_count = 0
        
        for experiment in self.compatible_experiments:
            # Get SECCM results for this experiment
            seccm_results = session.query(SECCMResult).filter_by(
                experiment_id=experiment.id,
                soundness_status='sound'  # Only use sound data
            ).all()
            
            if seccm_results:
                # Extract parameters (X)
                params = self._extract_parameters_from_experiment(experiment)
                
                # Get objective values (Y)
                objective_values = []
                for result in seccm_results:
                    if result.potential_at_0_05mA is not None:
                        objective_values.append(result.potential_at_0_05mA)
                
                if params and objective_values:
                    # Use average if multiple CV measurements
                    avg_objective = np.mean(objective_values)
                    
                    # Filter out fixed parameters (they're not in the parameter space)
                    optimizable_params = {k: v for k, v in params.items() 
                                         if not self._is_parameter_fixed(k)}
                    
                    # Convert to ParameterVector format
                    param_vector = ParameterVector().from_dict(optimizable_params)
                    
                    # Add to campaign
                    self.campaign.add_observation(param_vector, avg_objective)
                    loaded_count += 1
                    
                    print(f"📊 Loaded experiment {experiment.exp_id}: objective = {avg_objective:.4f}")
        
        print(f"✅ Loaded {loaded_count} experiments as training data")

    def _create_parameter_space_from_template(self) -> ParameterSpace:
        """
        Create Olympus parameter space from EID template process configurations.
        
        Returns:
            ParameterSpace object for Atlas/Olympus
        """
        param_space = ParameterSpace()
        
        # Extract process parameters from template configuration
        process_config = self.template.process_config
        
        for i, process in enumerate(process_config):
            process_type = process.get('TYPE', 'UNKNOWN')
            
            # Skip processes that are not enabled (USE = 'False')
            if process.get('USE', 'False').lower() != 'true':
                print(f"⏭️ Skipping disabled process {i}: {process_type} (USE = {process.get('USE', 'False')})")
                continue
            
            print(f"✅ Adding enabled process {i}: {process_type}")
            
            # Define parameter ranges based on process type
            if process_type == 'DISPENSE':
                # Concentration parameters (0.1 to 1.0)
                param_space.add(ParameterContinuous(
                    name=f'dispense_{i}_conc_ch1',
                    low=0.1, high=1.0,
                    description=f'Dispense {i} CH1 concentration'
                ))
                param_space.add(ParameterContinuous(
                    name=f'dispense_{i}_conc_ch2', 
                    low=0.1, high=1.0,
                    description=f'Dispense {i} CH2 concentration'
                ))
                # Timing parameters (1000 to 60000 ms)
                param_space.add(ParameterContinuous(
                    name=f'dispense_{i}_purge_time',
                    low=1000, high=60000,
                    description=f'Dispense {i} purge time (ms)'
                ))
                param_space.add(ParameterContinuous(
                    name=f'dispense_{i}_dispensing_time',
                    low=1000, high=10000,
                    description=f'Dispense {i} dispensing time (ms)'
                ))
                
            elif process_type == 'PRE_HEAT':
                # Heating time (300 to 3600 seconds)
                param_space.add(ParameterContinuous(
                    name=f'preheat_{i}_heating_time',
                    low=300, high=3600,
                    description=f'Pre-heat {i} heating time (s)'
                ))
                
            elif process_type == 'HEAT_TREATMENT':
                # Temperature (200 to 600°C)
                param_space.add(ParameterContinuous(
                    name=f'furnace_{i}_target_temp',
                    low=200, high=600,
                    description=f'Furnace {i} target temperature (°C)'
                ))
                # Holding time (5 to 60 minutes)
                param_space.add(ParameterContinuous(
                    name=f'furnace_{i}_holding_time',
                    low=5, high=60,
                    description=f'Furnace {i} holding time (min)'
                ))
                # Ramp rate (10 to 100°C/min)
                param_space.add(ParameterContinuous(
                    name=f'furnace_{i}_ramp_rate',
                    low=10, high=100,
                    description=f'Furnace {i} temperature ramp rate (°C/min)'
                ))
                
            elif process_type == 'PRESS_WORK':
                # Press position (100 to 200)
                param_space.add(ParameterContinuous(
                    name=f'press_{i}_position',
                    low=100, high=200,
                    description=f'Press {i} position'
                ))
                # Holding time (1 to 10 minutes)
                param_space.add(ParameterContinuous(
                    name=f'press_{i}_holding_time',
                    low=1, high=10,
                    description=f'Press {i} holding time (min)'
                ))
        
        return param_space

    def _load_existing_data(self):
        """
        Load existing experimental data from database into the campaign.
        
        Uses experiments that used the same template and have SECCM results.
        """
        # Get all experiments using this template
        experiments = session.query(Experiment).filter_by(eid_template_id=self.template.id).all()
        
        loaded_count = 0
        for experiment in experiments:
            # Get SECCM results for this experiment
            seccm_results = session.query(SECCMResult).filter_by(
                experiment_id=experiment.id,
                soundness_status='sound'  # Only use sound data
            ).all()
            
            if seccm_results:
                # Extract parameters from experiment processes
                params = self._extract_parameters_from_experiment(experiment)
                
                # Get objective value (potential at 0.05 mA)
                objective_values = []
                for result in seccm_results:
                    if result.potential_at_0_05mA is not None:
                        objective_values.append(result.potential_at_0_05mA)
                
                if params and objective_values:
                    # Use average if multiple CV measurements
                    avg_objective = np.mean(objective_values)
                    
                    # Convert to ParameterVector format
                    param_vector = ParameterVector().from_dict(params)
                    
                    # Add to campaign
                    self.campaign.add_observation(param_vector, avg_objective)
                    loaded_count += 1
                    
                    print(f"📊 Loaded experiment {experiment.exp_id}: objective = {avg_objective:.4f}")
        
        print(f"✅ Loaded {loaded_count} existing experiments into optimizer")

    def _extract_parameters_from_experiment(self, experiment: Experiment) -> Optional[Dict]:
        """
        Extract parameter values from experiment's process settings.
        
        Args:
            experiment: Experiment object from database
            
        Returns:
            Dictionary mapping parameter names to values
        """
        try:
            params = {}
            
            # Get all processes for this experiment, ordered by sequence
            processes = session.query(Process).filter_by(experiment_id=experiment.id).order_by(Process.sequence_order).all()
            
            # Create mapping based on reference process sequence
            process_type_counters = {}
            
            for process in processes:
                settings = process.settings
                process_type = process.process_type
                
                # Skip if process is not enabled in settings
                if settings.get('USE', 'False').lower() != 'true':
                    continue
                
                # Get index for this process type
                if process_type not in process_type_counters:
                    process_type_counters[process_type] = 0
                
                type_idx = process_type_counters[process_type]
                process_type_counters[process_type] += 1
                
                # Extract parameters based on process type
                if process_type == 'DISPENSE':
                    # Check which channels are enabled
                    use_ch1 = settings.get('USE_CH1', 'False').lower() == 'true'
                    use_ch2 = settings.get('USE_CH2', 'False').lower() == 'true'
                    
                    # Handle channel concentrations based on availability
                    if use_ch1 and use_ch2:
                        # Both channels enabled - extract both concentrations as optimization parameters
                        params[f'dispense_{type_idx}_conc_ch1'] = float(settings.get('CONCERNTRATION_CH1', 0.5))
                        params[f'dispense_{type_idx}_conc_ch2'] = float(settings.get('CONCERNTRATION_CH2', 0.5))
                    elif use_ch1 and not use_ch2:
                        # Only CH1 enabled - fix concentration at 1.0 (100%)
                        params[f'dispense_{type_idx}_conc_ch1'] = 1.0
                        print(f"🔒 Fixed dispense_{type_idx}_conc_ch1 = 1.0 (only channel enabled)")
                    elif not use_ch1 and use_ch2:
                        # Only CH2 enabled - fix concentration at 1.0 (100%)
                        params[f'dispense_{type_idx}_conc_ch2'] = 1.0
                        print(f"🔒 Fixed dispense_{type_idx}_conc_ch2 = 1.0 (only channel enabled)")
                    # If neither channel is enabled, don't add any concentration parameters
                    
                    # Always include timing parameters if dispense process is enabled
                    params[f'dispense_{type_idx}_purge_time'] = float(settings.get('PURGE_TIME', 30000))
                    params[f'dispense_{type_idx}_dispensing_time'] = float(settings.get('DISPENSING_TIME', 5000))
                    
                elif process_type == 'PRE_HEAT':
                    params[f'preheat_{type_idx}_heating_time'] = float(settings.get('HEATING_TIME', 1800))
                    
                elif process_type == 'HEAT_TREATMENT':
                    params[f'furnace_{type_idx}_target_temp'] = float(settings.get('TARGET_TEMP', 300))
                    params[f'furnace_{type_idx}_holding_time'] = float(settings.get('HOLDING_TIME', 10))
                    params[f'furnace_{type_idx}_ramp_rate'] = float(settings.get('INCREASE', 50))
                    params[f'furnace_{type_idx}_cooling_time'] = float(settings.get('COOLING_TIME', 1800))
                    
                elif process_type == 'PRESS_WORK':
                    params[f'press_{type_idx}_position'] = float(settings.get('PRESS_POSITION', 172.921))
                    params[f'press_{type_idx}_holding_time'] = float(settings.get('HOLDING_TIME', 4))
            
            return params if params else None
            
        except Exception as e:
            print(f"⚠️ Error extracting parameters from experiment {experiment.exp_id}: {e}")
            return None

    def add_new_experiment_result(self, experiment_id: int) -> bool:
        """
        Add a new experiment result to the optimization campaign.
        
        Args:
            experiment_id: Database ID of the new experiment
            
        Returns:
            True if experiment was successfully added
        """
        try:
            # Get experiment from database
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
            if not experiment:
                print(f"❌ Experiment {experiment_id} not found")
                return False
            
            # Check if experiment is compatible with current optimizer
            exp_sequence = self._extract_process_sequence(experiment)
            if not self._sequences_compatible(self.reference_process_sequence, exp_sequence):
                print(f"❌ Experiment {experiment_id} incompatible: {exp_sequence}")
                print(f"   Expected: {self.reference_process_sequence}")
                return False
            
            # Get SECCM results
            seccm_results = session.query(SECCMResult).filter_by(
                experiment_id=experiment.id,
                soundness_status='sound'
            ).all()
            
            if not seccm_results:
                print(f"❌ No sound SECCM results found for experiment {experiment_id}")
                return False
            
            # Extract parameters and objective
            params = self._extract_parameters_from_experiment(experiment)
            if not params:
                print(f"❌ Could not extract parameters from experiment {experiment_id}")
                return False
            
            # Get objective values
            objective_values = []
            for result in seccm_results:
                if result.potential_at_0_05mA is not None:
                    objective_values.append(result.potential_at_0_05mA)
            
            if not objective_values:
                print(f"❌ No valid objective values found for experiment {experiment_id}")
                return False
            
            # Add to campaign
            avg_objective = np.mean(objective_values)
            
            # Filter out fixed parameters (they're not in the parameter space)
            optimizable_params = {k: v for k, v in params.items() 
                                 if not self._is_parameter_fixed(k)}
            
            param_vector = ParameterVector().from_dict(optimizable_params)
            self.campaign.add_observation(param_vector, avg_objective)
            
            # Add to compatible experiments list
            self.compatible_experiments.append(experiment)
            
            print(f"✅ Added experiment {experiment.exp_id}: objective = {avg_objective:.4f}")
            print(f"📈 Total observations: {len(self.campaign.observations.get_values())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error adding experiment {experiment_id}: {e}")
            return False

    def suggest_next_experiments(self, num_suggestions: int = 1) -> List[Dict]:
        """
        Suggest next experiment parameters using Bayesian optimization.
        
        Fixed parameters (if any) are automatically included in all suggestions.
        
        Args:
            num_suggestions: Number of experiments to suggest
            
        Returns:
            List of parameter dictionaries for next experiments (includes fixed parameters)
        """
        suggestions = []
        max_attempts = num_suggestions * 10  # Prevent infinite loops
        attempts = 0
        
        while len(suggestions) < num_suggestions and attempts < max_attempts:
            attempts += 1
            
            # Get recommendation from planner
            if ATLAS_AVAILABLE:
                # Atlas GPPlanner handles known_constraints internally during optimization
                samples = self.planner.recommend(self.campaign.observations)
            else:
                # Olympus Planner API
                samples = self.planner.recommend(self.campaign)
            
            for sample in samples:
                # Convert ParameterVector to dictionary
                param_dict = sample.to_dict()
                
                # Merge in fixed parameters
                if self.fixed_parameters:
                    param_dict.update(self.fixed_parameters)
                
                # Manually enforce constraints (workaround for Atlas bug)
                if hasattr(self, '_manual_constraints') and self._manual_constraints:
                    # Normalize dual-channel dispense to satisfy sum-to-one constraint
                    self._normalize_dual_channel_constraints(param_dict)
                
                suggestions.append(param_dict)
                print(f"🎯 Suggested parameters: {param_dict}")
                
                if len(suggestions) >= num_suggestions:
                    break
        
        if len(suggestions) < num_suggestions:
            print(f"⚠️ Only generated {len(suggestions)}/{num_suggestions} valid suggestions after {attempts} attempts")
        
        return suggestions

    def create_eid_template_from_suggestion(self, suggestion: Dict, 
                                          template_name: Optional[str] = None,
                                          eid_directory: str = "D:\\SDL\\EXP_PARAMETER",
                                          description: Optional[str] = None,
                                          decimal_places: int = 2) -> 'EidTemplate':
        """
        Create an EID template from optimizer suggestion using database.py infrastructure.
        
        This uses the existing dataclasses and create_eid_template function to properly
        create both the .eid file and database record.
        
        Args:
            suggestion: Parameter dictionary from suggest_next_experiments()
            template_name: Template name (auto-generated if None)
            eid_directory: Directory to save EID file
            description: Template description
            decimal_places: Number of decimal places to round suggestion values to
            
        Returns:
            EidTemplate object that can be used for experiment creation
        """
        from database import (create_eid_template, DispenseProcess, PreHeatProcess, 
                            HeatTreatmentProcess, PressWorkProcess, SECCMAnalysis)
        from datetime import datetime
        import os
        
        # Generate template name if not provided
        if not template_name:
            self.campaign_iteration += 1
            template_name = f"{self.campaign_name}_Iter{self.campaign_iteration:04d}.EID"
        
        # Ensure .EID extension
        if not template_name.upper().endswith('.EID'):
            template_name += '.EID'
        
        # Generate description if not provided
        if not description:
            description = f"Bayesian optimization suggestion created at {datetime.now().isoformat()}"
        
        # Round suggestion values and parse parameters by process type
        rounded_suggestion = self._round_suggestion_values(suggestion, decimal_places)
        suggestion_by_process = self._parse_suggestion_parameters(rounded_suggestion)
        
        # Create process sequence using dataclasses
        process_sequence = []
        
        # Get reference process sequence from compatible experiments
        if not self.compatible_experiments:
            raise ValueError("No compatible experiments available as reference")
        
        base_experiment = self.compatible_experiments[0]
        if not base_experiment.eid_template:
            raise ValueError(f"Base experiment {base_experiment.id} has no EID template")
        
        base_processes = base_experiment.eid_template.process_config
        process_type_counters = {}
        
        # Build process sequence based on reference experiment structure
        for base_process in base_processes:
            process_type = base_process.get('TYPE')
            
            # Skip disabled processes
            if base_process.get('USE', 'False').lower() != 'true':
                continue
            
            # Get process index
            if process_type not in process_type_counters:
                process_type_counters[process_type] = 0
            type_idx = process_type_counters[process_type]
            process_type_counters[process_type] += 1
            
            # Create process using dataclasses with suggested parameters
            if process_type == 'DISPENSE':
                process = self._create_dispense_process(suggestion_by_process, type_idx, base_process)
            elif process_type == 'PRE_HEAT':
                process = self._create_preheat_process(suggestion_by_process, type_idx, base_process)
            elif process_type == 'HEAT_TREATMENT':
                process = self._create_heat_treatment_process(suggestion_by_process, type_idx, base_process)
            elif process_type == 'PRESS_WORK':
                process = self._create_press_work_process(suggestion_by_process, type_idx, base_process)
            else:
                print(f"⚠️ Unknown process type '{process_type}' - skipping")
                continue
            
            if process:
                process_sequence.append(process)
        
        # Create analyses (copy from base experiment)
        analyses = {}
        if hasattr(base_experiment.eid_template, 'analysis_config') and base_experiment.eid_template.analysis_config:
            analyses = base_experiment.eid_template.analysis_config.copy()
        else:
            # Default SECCM analysis
            analyses = {
                'SECCM': {'enabled': True}
            }
        
        # Create EID template using database.py function
        eid_template = create_eid_template(
            template_name=template_name,
            process_sequence=process_sequence,
            analyses=analyses,
            eid_directory=eid_directory,
            description=description
        )
        
        # Track this template in the campaign
        self.created_templates.append(eid_template)
        
        print(f"✅ Created EID template: {template_name}")
        print(f"🏷️  Campaign: {self.campaign_name} (Iteration {self.campaign_iteration})")
        print(f"📁 File location: {eid_template.file_path}")
        print(f"🆔 Template ID: {eid_template.id}")
        
        return eid_template

    def _parse_suggestion_parameters(self, suggestion: Dict) -> Dict:
        """
        Parse suggestion parameters by process type and index.
        
        Returns:
            Dict with structure: {process_type: {index: {param_name: value}}}
        """
        parsed = {}
        
        for param_name, value in suggestion.items():
            # Parse parameter name: e.g., "dispense_0_conc_ch1" -> ("dispense", 0, "conc_ch1")
            parts = param_name.split('_')
            if len(parts) >= 3:
                process_type = parts[0].upper()
                try:
                    type_idx = int(parts[1])
                    param_key = '_'.join(parts[2:])
                    
                    if process_type not in parsed:
                        parsed[process_type] = {}
                    if type_idx not in parsed[process_type]:
                        parsed[process_type][type_idx] = {}
                    
                    parsed[process_type][type_idx][param_key] = value
                except ValueError:
                    print(f"⚠️ Could not parse parameter: {param_name}")
        
        return parsed

    def _round_suggestion_values(self, suggestion: Dict, decimal_places: int) -> Dict:
        """Round numeric suggestion values to the specified decimal places."""
        def round_value(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, Number):
                return round(float(value), decimal_places)
            return value

        rounded = {}
        for key, value in suggestion.items():
            if isinstance(value, dict):
                rounded[key] = {
                    sub_key: round_value(sub_value) if not isinstance(sub_value, dict)
                    else self._round_suggestion_values(sub_value, decimal_places)
                    for sub_key, sub_value in value.items()
                }
            elif isinstance(value, (list, tuple)):
                rounded[key] = type(value)(
                    round_value(item) if not isinstance(item, dict)
                    else self._round_suggestion_values(item, decimal_places)
                    for item in value
                )
            else:
                rounded[key] = round_value(value)
        return rounded

    def _create_dispense_process(self, suggestions: Dict, type_idx: int, base_process: Dict):
        """Create DispenseProcess dataclass with suggested parameters."""
        from database import DispenseProcess
        
        # Start with base process values
        process = DispenseProcess(
            NAME=base_process.get('NAME', 'dispense_name'),
            USE='True',
            USE_CH1=base_process.get('USE_CH1', 'True'),
            USE_CH2=base_process.get('USE_CH2', 'True'),
            CONCERNTRATION_CH1=str(base_process.get('CONCERNTRATION_CH1', '0.5')),
            CONCERNTRATION_CH2=str(base_process.get('CONCERNTRATION_CH2', '0.5')),
            PURGE_TIME=str(base_process.get('PURGE_TIME', '30000')),
            DISPENSING_TIME=str(base_process.get('DISPENSING_TIME', '5000'))
        )
        
        # Update with suggested parameters
        if 'DISPENSE' in suggestions and type_idx in suggestions['DISPENSE']:
            params = suggestions['DISPENSE'][type_idx]
            
            if 'conc_ch1' in params:
                process.CONCERNTRATION_CH1 = str(params['conc_ch1'])
            if 'conc_ch2' in params:
                process.CONCERNTRATION_CH2 = str(params['conc_ch2'])
            if 'purge_time' in params:
                process.PURGE_TIME = str(params['purge_time'])
            if 'dispensing_time' in params:
                process.DISPENSING_TIME = str(params['dispensing_time'])
        
        return process

    def _create_preheat_process(self, suggestions: Dict, type_idx: int, base_process: Dict):
        """Create PreHeatProcess dataclass with suggested parameters."""
        from database import PreHeatProcess
        
        process = PreHeatProcess(
            NAME=base_process.get('NAME', 'pre_heat_name'),
            USE='True',
            HEATING_TIME=str(base_process.get('HEATING_TIME', '1800')),
            COOLING_TIME=str(base_process.get('COOLING_TIME', '1'))
        )
        
        # Update with suggested parameters
        if 'PREHEAT' in suggestions and type_idx in suggestions['PREHEAT']:
            params = suggestions['PREHEAT'][type_idx]
            
            if 'heating_time' in params:
                process.HEATING_TIME = str(params['heating_time'])
        
        return process

    def _create_heat_treatment_process(self, suggestions: Dict, type_idx: int, base_process: Dict):
        """Create HeatTreatmentProcess dataclass with suggested parameters."""
        from database import HeatTreatmentProcess
        
        process = HeatTreatmentProcess(
            NAME=base_process.get('NAME', 'heat_treatment_name'),
            USE='True',
            SCHEDULE_NO=str(base_process.get('SCHEDULE_NO', '99')),
            TARGET_TEMP=str(base_process.get('TARGET_TEMP', '300')),
            INCREASE=str(base_process.get('INCREASE', '50')),
            HOLDING_TIME=str(base_process.get('HOLDING_TIME', '10')),
            COOLING_TIME=str(base_process.get('COOLING_TIME', '5'))
        )
        
        # Update with suggested parameters
        if 'FURNACE' in suggestions and type_idx in suggestions['FURNACE']:
            params = suggestions['FURNACE'][type_idx]
            
            if 'target_temp' in params:
                process.TARGET_TEMP = str(params['target_temp'])
            if 'holding_time' in params:
                process.HOLDING_TIME = str(params['holding_time'])
            if 'ramp_rate' in params:
                process.INCREASE = str(params['ramp_rate'])
        
        return process

    def _create_press_work_process(self, suggestions: Dict, type_idx: int, base_process: Dict):
        """Create PressWorkProcess dataclass with suggested parameters."""
        from database import PressWorkProcess
        
        process = PressWorkProcess(
            NAME=base_process.get('NAME', 'PRC_0003'),
            USE='True',
            PRESS_POSITION=str(base_process.get('PRESS_POSITION', '172.921')),
            HOLDING_TIME=str(base_process.get('HOLDING_TIME', '4'))
        )
        
        # Update with suggested parameters
        if 'PRESS' in suggestions and type_idx in suggestions['PRESS']:
            params = suggestions['PRESS'][type_idx]
            
            if 'position' in params:
                process.PRESS_POSITION = str(params['position'])
            if 'holding_time' in params:
                process.HOLDING_TIME = str(params['holding_time'])
        
        return process

    def create_eid_templates_from_suggestions(self, suggestions: List[Dict], 
                                            eid_directory: str = "D:\\SDL\\EXP_PARAMETER",
                                            base_template_name: Optional[str] = None,
                                            decimal_places: int = 2) -> List['EidTemplate']:
        """
        Create multiple EID templates from a list of suggestions.
        
        Args:
            suggestions: List of parameter dictionaries from suggest_next_experiments()
            eid_directory: Directory to save EID files
            base_template_name: Base name for templates (will be numbered)
            decimal_places: Number of decimal places to round suggestion values to
            
        Returns:
            List of EidTemplate objects
        """
        from datetime import datetime
        
        created_templates = []
        timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        
        for i, suggestion in enumerate(suggestions, 1):
            if base_template_name:
                template_name = f"{base_template_name}_{i:02d}.EID"
            else:
                # Use same campaign-based naming as create_eid_template_from_suggestion
                self.campaign_iteration += 1
                template_name = f"{self.campaign_name}_Iter{self.campaign_iteration:04d}.EID"
            
            description = f"Bayesian optimization suggestion {i}/{len(suggestions)} created at {datetime.now().isoformat()}"
            
            try:
                eid_template = self.create_eid_template_from_suggestion(
                    suggestion=suggestion,
                    template_name=template_name,
                    eid_directory=eid_directory,
                    description=description,
                    decimal_places=decimal_places
                )
                created_templates.append(eid_template)
            except Exception as e:
                print(f"❌ Error creating EID template {i}: {e}")
        
        print(f"✅ Created {len(created_templates)} EID templates in {eid_directory}")
        return created_templates

    def add_new_experiment_result(self, experiment_id: int) -> bool:
        """
        Add a new experiment result to the existing optimization campaign.
        
        This allows you to incrementally add new experiments without re-instantiating
        the optimizer. Perfect for closed-loop optimization.
        
        Args:
            experiment_id: Database ID of the new experiment
            
        Returns:
            True if experiment was successfully added
        """
        try:
            # Get experiment from database
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
            if not experiment:
                print(f"❌ Experiment {experiment_id} not found in database")
                return False
            
            # Check if experiment is compatible with current optimizer
            exp_sequence = self._extract_process_sequence(experiment)
            if not self._sequences_compatible(self.reference_process_sequence, exp_sequence):
                print(f"❌ Experiment {experiment_id} incompatible: {exp_sequence}")
                print(f"   Expected: {self.reference_process_sequence}")
                return False
            
            # Get SECCM results
            seccm_results = session.query(SECCMResult).filter_by(
                experiment_id=experiment.id,
                soundness_status='sound'
            ).all()
            
            if not seccm_results:
                print(f"❌ No sound SECCM results found for experiment {experiment_id}")
                return False
            
            # Extract parameters and objective
            params = self._extract_parameters_from_experiment(experiment)
            if not params:
                print(f"❌ Could not extract parameters from experiment {experiment_id}")
                return False
            
            # Get objective values
            objective_values = []
            for result in seccm_results:
                if result.potential_at_0_05mA is not None:
                    objective_values.append(result.potential_at_0_05mA)
            
            if not objective_values:
                print(f"❌ No valid objective values found for experiment {experiment_id}")
                return False
            
            # Add to campaign
            avg_objective = np.mean(objective_values)
            
            # Filter out fixed parameters (they're not in the parameter space)
            optimizable_params = {k: v for k, v in params.items() 
                                 if not self._is_parameter_fixed(k)}
            
            param_vector = ParameterVector().from_dict(optimizable_params)
            self.campaign.add_observation(param_vector, avg_objective)
            
            # Add to compatible experiments list
            self.compatible_experiments.append(experiment)
            
            # Track in BO Campaign database
            from database import add_bo_campaign_record
            try:
                # Find the most recent template for this campaign
                latest_template_name = None
                if self.created_templates:
                    latest_template_name = self.created_templates[-1].template_name
                else:
                    # If no templates created yet, use campaign name as placeholder
                    latest_template_name = f"{self.campaign_name}_Manual"
                
                add_bo_campaign_record(
                    campaign_name=self.campaign_name,
                    eid_template_name=latest_template_name,
                    experiment_id=experiment.id,
                    exp_id=experiment.exp_id
                )
            except Exception as e:
                print(f"⚠️ Could not track in BO campaign database: {e}")
            
            print(f"✅ Added experiment {experiment.exp_id}: objective = {avg_objective:.4f}")
            print(f"🏷️  Campaign: {self.campaign_name}")
            print(f"📈 Total observations: {len(self.campaign.observations.get_values())}")
            print(f"🔗 Compatible experiments: {len(self.compatible_experiments)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error adding experiment {experiment_id}: {e}")
            return False

    def get_campaign_summary(self) -> Dict:
        """
        Get summary of the entire optimization campaign.
        
        Returns:
            Dictionary with campaign statistics and history
        """
        summary = self.get_optimization_summary()
        
        # Get BO campaign database info
        from database import get_bo_campaign_summary
        try:
            bo_summary = get_bo_campaign_summary(self.campaign_name)
        except Exception as e:
            bo_summary = {'error': f'Could not fetch BO campaign data: {e}'}
        
        # Add campaign-specific information
        summary.update({
            'campaign_name': self.campaign_name,
            'campaign_iteration': self.campaign_iteration,
            'created_templates': len(self.created_templates),
            'template_names': [t.template_name for t in self.created_templates],
            'compatible_experiments': len(self.compatible_experiments),
            'experiment_ids': [exp.id for exp in self.compatible_experiments],
            'bo_database_info': bo_summary
        })
        
        return summary

    def add_experiment_result(self, params: Dict, objective_value: float):
        """
        Add a new experimental result to the optimization campaign.
        
        Args:
            params: Parameter dictionary
            objective_value: Measured objective value (e.g., potential at 0.05 mA)
        """
        # Convert to ParameterVector
        param_vector = ParameterVector().from_dict(params)
        
        # Add to campaign
        self.campaign.add_observation(param_vector, objective_value)
        
        print(f"📊 Added result: objective = {objective_value:.4f}")
        print(f"📈 Total observations: {len(self.campaign.observations.get_values())}")

    def get_optimization_summary(self) -> Dict:
        """
        Get summary of optimization progress.
        
        Returns:
            Dictionary with optimization statistics
        """
        try:
            # Get objective values and parameter vectors separately
            objectives = self.campaign.observations.get_values()
            param_vectors = self.campaign.observations.get_params()
            
            if len(objectives) == 0:
                return {'status': 'No observations yet'}
            
            # Convert to numpy array for easier processing
            objectives = np.array(objectives)
            
            if self.goal == 'minimize':
                best_idx = np.argmin(objectives)
                best_objective = np.min(objectives)
            else:
                best_idx = np.argmax(objectives)
                best_objective = np.max(objectives)
            
            # Extract best parameters
            if len(param_vectors) > best_idx:
                best_param_vector = param_vectors[best_idx]
                if hasattr(best_param_vector, 'to_dict'):
                    best_params = best_param_vector.to_dict()
                elif isinstance(best_param_vector, dict):
                    best_params = best_param_vector
                else:
                    best_params = {'error': 'Could not extract parameters'}
            else:
                best_params = {'error': 'Parameter vector not available'}
            
            return {
                'total_experiments': len(objectives),
                'best_objective': float(best_objective),
                'best_parameters': best_params,
                'objective_mean': float(np.mean(objectives)),
                'objective_std': float(np.std(objectives)),
                'goal': self.goal
            }
            
        except Exception as e:
            # Fallback: Just provide basic statistics without parameters
            try:
                objectives = self.campaign.observations.get_values()
                if len(objectives) > 0:
                    objectives = np.array(objectives)
                    if self.goal == 'minimize':
                        best_objective = np.min(objectives)
                    else:
                        best_objective = np.max(objectives)
                    
                    return {
                        'total_experiments': len(objectives),
                        'best_objective': float(best_objective),
                        'best_parameters': {'error': 'Parameters not accessible'},
                        'objective_mean': float(np.mean(objectives)),
                        'objective_std': float(np.std(objectives)),
                        'goal': self.goal,
                        'note': 'Parameter details not available due to API differences'
                    }
                else:
                    return {'status': 'No observations yet'}
            except:
                return {
                    'status': 'Error getting summary',
                    'error': str(e)
                }

    def plot_optimization_progress(self, save_path: Optional[str] = None):
        """
        Plot optimization progress over iterations.
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            observations = self.campaign.observations.get_values()
            if len(observations) == 0:
                print("⚠️ No observations to plot")
                return
            
            objectives = [obs[1] for obs in observations]
            iterations = range(1, len(objectives) + 1)
            
            # Calculate best so far
            if self.goal == 'minimize':
                best_so_far = np.minimum.accumulate(objectives)
            else:
                best_so_far = np.maximum.accumulate(objectives)
            
            plt.figure(figsize=(12, 5))
            
            # Plot objective values
            plt.subplot(1, 2, 1)
            plt.plot(iterations, objectives, 'bo-', alpha=0.7, label='Observed values')
            plt.plot(iterations, best_so_far, 'r-', linewidth=2, label='Best so far')
            plt.xlabel('Iteration')
            plt.ylabel('Objective Value')
            plt.title(f'Optimization Progress ({self.goal})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot objective distribution
            plt.subplot(1, 2, 2)
            plt.hist(objectives, bins=min(20, len(objectives)//2), alpha=0.7, edgecolor='black')
            plt.axvline(best_so_far[-1], color='red', linestyle='--', linewidth=2, label=f'Best: {best_so_far[-1]:.4f}')
            plt.xlabel('Objective Value')
            plt.ylabel('Frequency')
            plt.title('Objective Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Plot saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("📊 matplotlib not available for plotting") 