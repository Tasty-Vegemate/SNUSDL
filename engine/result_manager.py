"""
ResultManager: Handles parsing of experiment results and preprocessing for Bayesian optimization.
"""
import os
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import re
from database import session, Experiment, Process, Result, Device, SECCMResult

from galvani import BioLogic


class ResultManager:
    def __init__(self, results_dir: str = r"C:\Users\abc\Documents\EC-Lab\Data"):
        self.results_dir = results_dir
        self.seccm_parsers = {
            '.mpr': self._parse_mpr_file
        }
        # self.xrd_parsers = {
        #     '.txt': self._parse_xrd_txt,
        #     '.csv': self._parse_xrd_csv,
        #     '.xy': self._parse_xrd_xy
        # }
        
    def parse_experiment_results(self, experiment_id) -> Dict[str, Any]:
        """
        Parse all result files for a given experiment and return structured data.
        
        Args:
            experiment_id: Experiment ID - can be either:
                - String: experiment name (e.g., "250915_EXP_001_ITO_79") 
                - Integer: database ID (e.g., 17)
            
        Returns:
            Dictionary containing parsed results by analysis type
        """
        print(f"🔍 Parsing results for experiment: {experiment_id}")
        
        # Get experiment from database - handle both string names and numeric IDs
        if isinstance(experiment_id, int):
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
        else:
            experiment = session.query(Experiment).filter_by(exp_id=experiment_id).first()
            
        if not experiment:
            print(f"❌ Experiment {experiment_id} not found")
            return {}
        
        # Get all results for this experiment
        results = session.query(Result).filter_by(experiment_id=experiment.id).all()
        
        if not results:
            print(f"❌ No results found for experiment {experiment_id}")
            return {}
        
        parsed_results = {
            'experiment_id': experiment_id,
            'sample_id': experiment.sid,
            'timestamp': experiment.timestamp,
            'seccm_data': {},
            'xrd_data': {},
            'metadata': {}
        }
        
        # Parse each result file
        for result in results:
            if not result.file_path or not os.path.exists(result.file_path):
                print(f"⚠️ Result file not found: {result.file_path}")
                continue
                
            file_ext = os.path.splitext(result.file_path)[1].lower()
            
            if result.result_type == 'seccm_data':
                parsed_data = self._parse_seccm_file(result.file_path, file_ext)
                if parsed_data:
                    parsed_results['seccm_data'][os.path.basename(result.file_path)] = parsed_data
                    
            elif result.result_type == 'xrd_data':
                parsed_data = self._parse_xrd_file(result.file_path, file_ext)
                if parsed_data:
                    parsed_results['xrd_data'][os.path.basename(result.file_path)] = parsed_data
        
        print(f"✅ Parsed {len(parsed_results['seccm_data'])} SECCM files and {len(parsed_results['xrd_data'])} XRD files")
        return parsed_results
    
    def _parse_seccm_file(self, file_path: str, file_ext: str) -> Optional[Dict]:
        """Parse SECCM file based on extension"""
        if file_ext in self.seccm_parsers:
            try:
                return self.seccm_parsers[file_ext](file_path)
            except Exception as e:
                print(f"❌ Error parsing SECCM file {file_path}: {e}")
                return None
        return None

    def _check_data_soundness(self, df: pd.DataFrame, file_path: str) -> Tuple[str, Optional[float], Optional[str]]:
        """
        Check soundness of SECCM data based on DataFrame analysis.
        
        Args:
            df: Pandas DataFrame with measurement data
            file_path: Path to the file (for logging)
            
        Returns:
            Tuple of (soundness_status, rcp_ohm, soundness_reason)
            - soundness_status: 'sound', 'unsound', or 'pending'
            - rcp_ohm: Resistance value in Ohms (or None if not found)
            - soundness_reason: Reason if unsound (or None if sound)
        """
        try:
            # Check 1: DataFrame must not be empty
            if df.empty:
                reason = "DataFrame is empty - no data in file"
                print(f"⚠️ Data unsound: {reason}")
                return ('unsound', None, reason)
            
            # Check 2: Required columns must exist
            potential_col, current_col = self._find_data_columns(df)
            
            if potential_col is None or current_col is None:
                reason = f"Required columns not found. Available columns: {list(df.columns)}"
                print(f"⚠️ Data unsound: {reason}")
                return ('unsound', None, reason)
            
            # Check 3: Check for Rcp/Ohm resistance (if available in DataFrame)
            rcp_ohm = None
            if 'Rcp/Ohm' in df.columns:
                rcp_ohm = df['Rcp/Ohm'].iloc[0]  # Get first value
            elif 'R' in df.columns:
                rcp_ohm = df['R'].iloc[0]
            
            # If Rcp/Ohm found, check threshold
            if rcp_ohm is not None:
                print(f"📊 Found Rcp/Ohm: {rcp_ohm:.2f} Ω")
                if rcp_ohm > 300:
                    reason = f"Rcp/Ohm too high: {rcp_ohm:.2f} Ω (threshold: 300 Ω)"
                    print(f"⚠️ Data unsound: {reason}")
                    return ('unsound', rcp_ohm, reason)
            
            # Check 4: Current values must not be too low (check if all currents are near zero)
            currents = df[current_col].values
            max_current = np.max(np.abs(currents))
            
            if max_current < 0.01:  # Less than 1 nA
                reason = f"Current values too low: max = {max_current:.2e} A (threshold: 1e-9 A)"
                print(f"⚠️ Data unsound: {reason}")
                return ('unsound', rcp_ohm, reason)
            
            # Check 5: Must have reasonable number of data points
            if len(df) < 10:
                reason = f"Too few data points: {len(df)} (minimum: 10)"
                print(f"⚠️ Data unsound: {reason}")
                return ('unsound', rcp_ohm, reason)
            
            # All checks passed
            print(f"✅ Data sound: {len(df)} points, max current = {max_current:.2e} A")
            if rcp_ohm is not None:
                print(f"✅ Rcp/Ohm = {rcp_ohm:.2f} Ω")
            return ('sound', rcp_ohm, None)
                
        except Exception as e:
            reason = f"Error checking soundness: {str(e)}"
            print(f"❌ {reason}")
            return ('unsound', None, reason)

    def _parse_mpr_file(self, file_path: str) -> Dict:
        """
        Parse BioLogic EC-Lab .mpr file using BioLogic library
        Handles CV, OCV, and CI data
        """
        print(f"📊 Parsing MPR file: {os.path.basename(file_path)}")
        
        data = {}
        
        try:
            # Use BioLogic to read MPR file
            mpr_file = BioLogic.MPRfile(file_path)
            
            # Extract data from BioLogic object to DataFrame FIRST
            df = pd.DataFrame(mpr_file.data)
            
            # NOW check data soundness using the DataFrame
            soundness_status, rcp_ohm, soundness_reason = self._check_data_soundness(df, file_path)
            data['soundness_status'] = soundness_status
            data['rcp_ohm'] = rcp_ohm
            data['soundness_reason'] = soundness_reason
            
            if df.empty:
                print("⚠️ No data found in MPR file")
                return data
            
            # Extract potential and current columns
            potential_col, current_col = self._find_data_columns(df)
            
            if potential_col is None or current_col is None:
                print(f"⚠️ Could not find potential/current columns in MPR file")
                print(f"Available columns: {list(df.columns)}")
                return data
            
            # Extract data
            potentials = df[potential_col].values
            currents = df[current_col].values
            
            # Determine measurement type from filename
            filename = os.path.basename(file_path).lower()
            measurement_type = self._determine_measurement_type(filename)
            
            data = {
                'type': measurement_type,
                'potential': np.array(potentials),
                'current': np.array(currents),
                'data_points': len(potentials),
                'potential_range': [np.min(potentials), np.max(potentials)],
                'current_range': [np.min(currents), np.max(currents)],
                'filename': os.path.basename(file_path),
                'raw_data': df  # Keep raw data for cycle analysis
            }
            
            # Calculate metrics based on measurement type
            if measurement_type == 'cv':
                data['metrics'] = self._analyze_cv_data(potentials, currents, df)
            # elif measurement_type == 'ocv':
            #     data['metrics'] = self._calculate_ocv_metrics(potentials, currents)
            # elif measurement_type == 'ci':
            #     data['metrics'] = self._calculate_ci_metrics(potentials, currents)
            # else:
            #     data['metrics'] = self._calculate_general_electrochemical_metrics(potentials, currents)
            
            print(f"✅ Successfully parsed {measurement_type.upper()} data: {len(potentials)} points")
            
        except Exception as e:
            print(f"❌ Error parsing MPR file with BioLogic: {e}")
            return {}
        
        return data

    def _find_data_columns(self, df: pd.DataFrame) -> tuple:
        """Find potential and current columns in DataFrame"""
        potential_col = None
        current_col = None
        
        
        # Find potential column (usually 'Ewe/V' or similar)
        for col in df.columns:
            if 'Ewe' in col or 'potential' in col.lower() or 'E' in col:
                potential_col = col
                break
        
        # Find current column (usually 'I/A' or similar)
        for col in df.columns:
            if '<I>/mA' in col or 'current' in col.lower() or 'I' in col:
                current_col = col
                break
        
        return potential_col, current_col

    def _determine_measurement_type(self, filename: str) -> str:
        """Determine measurement type from filename"""
        filename_lower = filename.lower()
        
        if 'cv' in filename_lower:
            return 'cv'
        elif 'ocv' in filename_lower:
            return 'ocv'
        elif 'ci' in filename_lower or 'interrupt' in filename_lower:
            return 'ci'
        else:
            return 'unknown'

    def _analyze_cv_data(self, potentials: np.ndarray, currents: np.ndarray, df: pd.DataFrame) -> Dict:
        """
        Analyze CV data to extract potential at specific current (0.05 mA) from last cycle
        
        Args:
            potentials: Array of potential values
            currents: Array of current values  
            df: Raw DataFrame with cycle information
            
        Returns:
            Dictionary containing CV analysis metrics
        """
        print(f"🔍 Analyzing CV data for potential at 0.05 mA from last cycle")
        
        metrics = {}
        
        try:
            # Step 1: Find cycle column and get last cycle number
            cycle_col = self._find_cycle_column(df)
            if cycle_col is None:
                print("⚠️ No cycle column found, analyzing entire dataset")
                last_cycle_data = df
            else:
                last_cycle_num = df[cycle_col].max()
                print(f"📊 Last cycle number: {last_cycle_num}")
                
                # Step 2: Crop data to last cycle only
                last_cycle_data = df[df[cycle_col] == last_cycle_num].copy()
                print(f"📊 Last cycle data points: {len(last_cycle_data)}")
            
            # Step 3: Extract potential and current for last cycle
            potential_col, current_col = self._find_data_columns(last_cycle_data)
            
            if potential_col is None or current_col is None:
                print("⚠️ Could not find potential/current columns in last cycle data")
                return metrics
            
            last_cycle_potentials = last_cycle_data[potential_col].values
            last_cycle_currents = last_cycle_data[current_col].values
            
            # Step 4: Find potential at 0.05 mA during increasing current phase
            target_current = 0.05  # 0.05 mA in Amperes
            potential_at_target = self._find_potential_at_current(
                last_cycle_potentials, 
                last_cycle_currents, 
                target_current
            )
            
            if potential_at_target is not None:
                metrics['potential_at_0_05mA'] = potential_at_target
                print(f"✅ Found potential at 0.05 mA: {potential_at_target:.4f} V")
            else:
                print("⚠️ Could not find potential at 0.05 mA")
                metrics['potential_at_0_05mA'] = None
            
            # Additional basic metrics
            metrics.update({
                'last_cycle_points': len(last_cycle_data),
                'potential_range': np.max(last_cycle_potentials) - np.min(last_cycle_potentials),
                'current_range': np.max(last_cycle_currents) - np.min(last_cycle_currents),
                'max_current': np.max(np.abs(last_cycle_currents)),
                'min_current': np.min(np.abs(last_cycle_currents))
            })
            
        except Exception as e:
            print(f"❌ Error analyzing CV data: {e}")
            metrics['potential_at_0_05mA'] = None
        
        return metrics

    def _find_cycle_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find cycle number column in DataFrame"""
        cycle_candidates = ['cycle', 'Cycle', 'CYCLE', 'N_Cycle', 'cycle number']
        
        for col in df.columns:
            if any(candidate in col for candidate in cycle_candidates):
                print(f"📊 Found cycle column: {col}")
                return col
        
        return None

    def _find_potential_at_current(self, potentials: np.ndarray, currents: np.ndarray, target_current: float) -> Optional[float]:
        """
        Find potential at specific current using linear interpolation during increasing current phase
        
        Collects points around target current (0.04-0.06 mA range) and uses linear interpolation
        to find the exact potential at the target current (0.05 mA).
        
        Args:
            potentials: Array of potential values (V)
            currents: Array of current values (A)
            target_current: Target current value in Amperes (0.05 for 0.05 mA)
            
        Returns:
            Interpolated potential value at target current, or None if not found
        """
        try:
            # Convert target current to mA for easier handling
            target_current_mA = target_current * 1  # Convert A to mA
            currents_mA = currents * 1  # Convert all currents to mA
            
            print(f"🔍 Finding potential at {target_current_mA:.2f} mA using interpolation")
            
            # Define ranges around target current (0.05 mA)
            lower_range = [0.04, 0.05]  # 0.04 to 0.05 mA
            upper_range = [0.05, 0.06]  # 0.05 to 0.06 mA (extended from 0.08 to 0.06 for better accuracy)
            
            # Find increasing current phase first
            increasing_indices = []
            for i in range(1, len(currents_mA)):
                if currents_mA[i] > currents_mA[i-1]:  # Current is increasing
                    increasing_indices.append(i)
            
            if len(increasing_indices) == 0:
                print("⚠️ No increasing current phase found")
                return None
            
            # Filter data to increasing phase only
            increasing_currents = currents_mA[increasing_indices]
            increasing_potentials = potentials[increasing_indices]
            
            print(f"📊 Found {len(increasing_indices)} points in increasing current phase")
            
            # Collect points in lower range (0.04 to 0.05 mA)
            lower_mask = (increasing_currents >= lower_range[0]) & (increasing_currents <= lower_range[1])
            lower_currents = increasing_currents[lower_mask]
            lower_potentials = increasing_potentials[lower_mask]
            
            # Collect points in upper range (0.05 to 0.06 mA)  
            upper_mask = (increasing_currents >= upper_range[0]) & (increasing_currents <= upper_range[1])
            upper_currents = increasing_currents[upper_mask]
            upper_potentials = increasing_potentials[upper_mask]
            
            print(f"📊 Lower range points: {len(lower_currents)}, Upper range points: {len(upper_currents)}")
            
            # Select up to 10 points from each range
            if len(lower_currents) > 10:
                # Select evenly spaced points
                indices = np.linspace(0, len(lower_currents)-1, 10, dtype=int)
                lower_currents = lower_currents[indices]
                lower_potentials = lower_potentials[indices]
            
            if len(upper_currents) > 10:
                # Select evenly spaced points
                indices = np.linspace(0, len(upper_currents)-1, 10, dtype=int)
                upper_currents = upper_currents[indices]
                upper_potentials = upper_potentials[indices]
            
            # Combine both ranges for interpolation
            combined_currents = np.concatenate([lower_currents, upper_currents])
            combined_potentials = np.concatenate([lower_potentials, upper_potentials])
            
            if len(combined_currents) < 2:
                print(f"⚠️ Insufficient points for interpolation: {len(combined_currents)}")
                return None
            
            # Sort by current for proper interpolation
            sort_indices = np.argsort(combined_currents)
            combined_currents = combined_currents[sort_indices]
            combined_potentials = combined_potentials[sort_indices]
            
            print(f"📊 Using {len(combined_currents)} points for interpolation")
            print(f"📊 Current range: {combined_currents.min():.3f} to {combined_currents.max():.3f} mA")
            print(f"📊 Potential range: {combined_potentials.min():.4f} to {combined_potentials.max():.4f} V")
            
            # Check if target current is within the range
            if target_current_mA < combined_currents.min() or target_current_mA > combined_currents.max():
                print(f"⚠️ Target current {target_current_mA:.3f} mA outside interpolation range")
                print(f"    Available range: {combined_currents.min():.3f} to {combined_currents.max():.3f} mA")
                return None
            
            # Perform linear interpolation
            interpolated_potential = np.interp(target_current_mA, combined_currents, combined_potentials)
            
            print(f"✅ Interpolated potential at {target_current_mA:.2f} mA: {interpolated_potential:.4f} V")
            
            # Optional: Calculate R-squared for interpolation quality assessment
            try:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(combined_currents, combined_potentials)
                r_squared = r_value ** 2
                print(f"📊 Linear fit quality (R²): {r_squared:.4f}")
                if r_squared < 0.8:
                    print(f"⚠️ Warning: Low R² ({r_squared:.4f}) indicates poor linear fit")
            except ImportError:
                print("📊 scipy not available for R² calculation")
            
            return float(interpolated_potential)
            
        except Exception as e:
            print(f"❌ Error in interpolation-based potential finding: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _plot_interpolation_debug(self, currents_mA: np.ndarray, potentials: np.ndarray, 
                                 target_current_mA: float, interpolated_potential: float,
                                 save_path: str = None) -> None:
        """
        Create a debug plot showing the interpolation process (optional visualization)
        
        Args:
            currents_mA: Current values in mA
            potentials: Potential values in V
            target_current_mA: Target current in mA
            interpolated_potential: Interpolated potential result
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            
            # Plot all data points
            plt.scatter(currents_mA, potentials, alpha=0.6, s=20, label='Data points')
            
            # Plot interpolation line
            current_range = np.linspace(currents_mA.min(), currents_mA.max(), 100)
            interpolated_line = np.interp(current_range, currents_mA, potentials)
            plt.plot(current_range, interpolated_line, 'r--', alpha=0.8, label='Interpolation line')
            
            # Highlight target point
            plt.scatter([target_current_mA], [interpolated_potential], 
                       color='red', s=100, marker='x', linewidth=3, 
                       label=f'Target: {target_current_mA:.2f} mA → {interpolated_potential:.4f} V')
            
            plt.xlabel('Current (mA)')
            plt.ylabel('Potential (V)')
            plt.title('CV Interpolation for Potential at 0.05 mA')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Debug plot saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("📊 matplotlib not available for debug plotting")
        except Exception as e:
            print(f"⚠️ Error creating debug plot: {e}")

    # def _calculate_ocv_metrics(self, potentials: np.ndarray, currents: np.ndarray) -> Dict:
    #     """Calculate OCV-specific metrics"""
    #     metrics = {
    #         'ocv_stability': 1.0 / (np.std(potentials) + 1e-6),
    #         'ocv_range': np.max(potentials) - np.min(potentials),
    #         'final_ocv': potentials[-1],
    #         'ocv_drift': (potentials[-1] - potentials[0]) / len(potentials) if len(potentials) > 1 else 0
    #     }
    #     return metrics

    # def _calculate_ci_metrics(self, potentials: np.ndarray, currents: np.ndarray) -> Dict:
    #     """Calculate CI-specific metrics"""
    #     metrics = {
    #         'ci_stability': 1.0 / (np.std(currents) + 1e-6),
    #         'ci_range': np.max(currents) - np.min(currents),
    #         'final_current': currents[-1],
    #         'current_drift': (currents[-1] - currents[0]) / len(currents) if len(currents) > 1 else 0
    #     }
    #     return metrics

    # def _calculate_general_electrochemical_metrics(self, potentials: np.ndarray, currents: np.ndarray) -> Dict:
    #     """Calculate general electrochemical metrics"""
    #     metrics = {
    #         'max_current': np.max(np.abs(currents)),
    #         'min_current': np.min(np.abs(currents)),
    #         'current_ratio': np.max(np.abs(currents)) / np.min(np.abs(currents)) if np.min(np.abs(currents)) > 0 else 0,
    #         'potential_window': np.max(potentials) - np.min(potentials),
    #         'current_stability': 1.0 / (np.std(currents) / np.mean(np.abs(currents))) if np.mean(np.abs(currents)) > 0 else 0,
    #     }
    #     return metrics

    def _convert_to_json_serializable(self, obj):
        """
        Convert numpy types and other non-JSON-serializable objects to JSON-serializable Python types.
        
        Args:
            obj: Object to convert (can be nested dict, list, numpy types, etc.)
            
        Returns:
            JSON-serializable version of the object
        """
        import numpy as np
        
        if obj is None:
            return None
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to lists
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        else:
            return obj  # Return as-is for native Python types (int, float, str, bool)

    def process_and_save_cv_result(self, experiment_id: str) -> List[SECCMResult]:
        """
        Complete workflow: Parse CV data, check soundness, extract metrics, and save to database.
        
        Args:
            experiment_id: Experiment ID (e.g., "250915_EXP_001_ITO_0079")
            
        Returns:
            List of SECCMResult objects
        """
        print(f"\n🚀 Processing CV results for experiment: {experiment_id}")
        
        # Get experiment from database

        if isinstance(experiment_id, int):
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
        else:
            experiment = session.query(Experiment).filter_by(exp_id=experiment_id).first()
        if not experiment:
            print(f"❌ Experiment {experiment_id} not found")
            return []
        
        # Get all SECCM results (CV files)
        results = session.query(Result).filter_by(
            experiment_id=experiment.id,
            result_type='seccm_data'
        ).all()
        
        if not results:
            print(f"❌ No SECCM results found for experiment {experiment_id}")
            return []
        
        seccm_results = []
        
        # Process each CV file
        for result in results:
            if not result.file_path or not os.path.exists(result.file_path):
                print(f"⚠️ Result file not found: {result.file_path}")
                continue
            
            # Only process CV files (.mpr)
            if not result.file_path.endswith('.mpr'):
                print(f"⏭️ Skipping non-MPR file: {result.file_path}")
                continue
            
            # Determine if this is a CV file
            filename = os.path.basename(result.file_path).lower()
            if '_03_cv_' not in filename and 'cv' not in filename:
                print(f"⏭️ Skipping non-CV file: {result.file_path}")
                continue
            
            print(f"\n📊 Processing CV file: {os.path.basename(result.file_path)}")
            
            # Parse the MPR file (includes soundness checking)
            parsed_data = self._parse_mpr_file(result.file_path)
            
            if not parsed_data:
                print(f"❌ Failed to parse file")
                continue
            
            # Extract metrics (convert numpy types to JSON-serializable types)
            soundness_status = parsed_data.get('soundness_status', 'unsound')
            rcp_ohm = self._convert_to_json_serializable(parsed_data.get('rcp_ohm'))
            soundness_reason = parsed_data.get('soundness_reason')
            measurement_type = parsed_data.get('type', 'cv')
            metrics = parsed_data.get('metrics', {})
            potential_at_0_05mA = self._convert_to_json_serializable(metrics.get('potential_at_0_05mA'))
            
            # Prepare preprocessed data (convert numpy types to JSON-serializable types)
            preprocessed_data = {
                'potential_range': self._convert_to_json_serializable(parsed_data.get('potential_range')),
                'current_range': self._convert_to_json_serializable(parsed_data.get('current_range')),
                'data_points': self._convert_to_json_serializable(parsed_data.get('data_points')),
                'metrics': self._convert_to_json_serializable(metrics)
            }
            
            # Save to database
            seccm_result = self.save_seccm_result_to_database(
                result_id=result.id,
                experiment_id=experiment.id,
                file_path=result.file_path,
                measurement_type=measurement_type,
                soundness_status=soundness_status,
                rcp_ohm=rcp_ohm,
                soundness_reason=soundness_reason,
                potential_at_0_05mA=potential_at_0_05mA,
                preprocessed_data=preprocessed_data
            )
            
            if seccm_result:
                seccm_results.append(seccm_result)
        
        print(f"\n✅ Processed {len(seccm_results)} CV files for experiment {experiment_id}")
        return seccm_results
    
    def save_seccm_result_to_database(self, result_id: int, experiment_id: int, 
                                      file_path: str, measurement_type: str,
                                      soundness_status: str, rcp_ohm: Optional[float],
                                      soundness_reason: Optional[str],
                                      potential_at_0_05mA: Optional[float],
                                      preprocessed_data: Dict) -> Optional[SECCMResult]:
        """
        Save SECCM preprocessing results to the SECCMResult table.
        
        Args:
            result_id: ID from Result table
            experiment_id: Experiment ID
            file_path: Path to the CV .mpr file
            measurement_type: Type of measurement (cv/ocv/ci)
            soundness_status: 'sound' or 'unsound'
            rcp_ohm: Rcp/Ohm resistance value
            soundness_reason: Reason if unsound
            potential_at_0_05mA: Extracted potential value
            preprocessed_data: Other metrics
            
        Returns:
            SECCMResult object or None if failed
        """
        try:
            # Check if entry already exists
            existing = session.query(SECCMResult).filter_by(result_id=result_id).first()
            
            if existing:
                # Update existing entry
                print(f"📝 Updating existing SECCM result for result_id={result_id}")
                existing.soundness_status = soundness_status
                existing.rcp_ohm = rcp_ohm
                existing.soundness_reason = soundness_reason
                existing.soundness_checked_at = datetime.utcnow()
                existing.potential_at_0_05mA = potential_at_0_05mA
                existing.preprocessed_data = preprocessed_data
                existing.preprocessing_status = 'completed' if potential_at_0_05mA is not None else 'failed'
                existing.preprocessing_completed_at = datetime.utcnow()
                seccm_result = existing
            else:
                # Create new entry
                print(f"📝 Creating new SECCM result for result_id={result_id}")
                seccm_result = SECCMResult(
                    result_id=result_id,
                    experiment_id=experiment_id,
                    raw_data_path=file_path,
                    measurement_type=measurement_type,
                    soundness_status=soundness_status,
                    rcp_ohm=rcp_ohm,
                    soundness_reason=soundness_reason,
                    soundness_checked_at=datetime.utcnow(),
                    potential_at_0_05mA=potential_at_0_05mA,
                    preprocessed_data=preprocessed_data,
                    preprocessing_status='completed' if potential_at_0_05mA is not None else 'failed',
                    preprocessing_completed_at=datetime.utcnow()
                )
                session.add(seccm_result)
            
            session.commit()
            print(f"✅ Saved SECCM result to database (soundness: {soundness_status})")
            return seccm_result
            
        except Exception as e:
            print(f"❌ Error saving SECCM result to database: {e}")
            session.rollback()
            return None

    def update_experiment_index(self, experiment_id: str, optimization_targets: Dict[str, float]):
        """
        Update the experiment index with optimization target values.
        
        Args:
            experiment_id: Experiment ID
            optimization_targets: Dictionary of target values for optimization
        """
        print(f"📝 Updating experiment index for {experiment_id}")
        
        # Get experiment from database
        experiment = session.query(Experiment).filter_by(exp_id=experiment_id).first()
        if not experiment:
            print(f"❌ Experiment {experiment_id} not found")
            return
        
        # Update experiment metadata with optimization targets
        if experiment.eid_metadata is None:
            experiment.eid_metadata = {}
        
        experiment.eid_metadata['optimization_targets'] = optimization_targets
        experiment.eid_metadata['optimization_updated'] = datetime.now().isoformat()
        
        session.commit()
        print(f"✅ Updated experiment index with {len(optimization_targets)} optimization targets")
