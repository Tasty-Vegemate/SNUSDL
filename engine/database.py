from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, ForeignKey, LargeBinary, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
import configparser
from dataclasses import dataclass, asdict
from typing import Optional


"""
Example Usage with Separated EID Template and Experiment Management

# ===== STEP 1: CREATE EID TEMPLATES (Once) =====

from engine.database import (
    create_eid_template,
    create_experiment,
    create_dispense_process,
    create_heat_treatment_process,
    create_pre_heat_process,
    create_xrd_analysis
)

# Create process objects using convenience functions
standard_process_sequence = [
    create_dispense_process("High Conc Dispense", ch1_concentration="0.8", purge_time="25000"),
    create_pre_heat_process("Quick Warm Up", heating_time="900"),
    create_heat_treatment_process("High Temp Treatment", target_temp="450", holding_time="20")
]

# Create analyses dictionary
standard_analyses = {
    'XRD': {'enabled': True, 'PROFILE_NAME': 'High_Res'},
    'SECCM': {'enabled': True, 'PRE_PUMPING': '15'}
}

# Create EID template (creates both file and database record)
template = create_eid_template(
    template_name="STANDARD_ITO_PROCESS.EID",
    process_sequence=standard_process_sequence,
    analyses=standard_analyses,
    eid_directory="SUP/SNUSDL/EXP_PARAMETER",
    description="Standard ITO processing template"
)

# Create another template for high temperature processing
high_temp_process = [
    create_dispense_process("Prep", ch1_concentration="0.6"),
    create_heat_treatment_process("High Heat", target_temp="600", holding_time="30")
]

high_temp_template = create_eid_template(
    template_name="HIGH_TEMP_PROCESS.EID",
    process_sequence=high_temp_process,
    analyses={'XRD': {'enabled': True, 'PROFILE_NAME': 'High_Temp'}},
    eid_directory="SUP/SNUSDL/EXP_PARAMETER",
    description="High temperature processing template"
)

# ===== STEP 2: CREATE EXPERIMENTS (Many times) =====

# Experiment 1: Using standard template
exp1, processes1 = create_experiment(
    exp_id="EXP_001_SAMPLE_A",
    sample_name="ITO test on sample A",
    sid="SAMPLE_A_BATCH_2024_001",
    eid_template_name="STANDARD_ITO_PROCESS.EID"
)

# Experiment 2: Same template, different sample
exp2, processes2 = create_experiment(
    exp_id="EXP_002_SAMPLE_B", 
    sample_name="ITO test on sample B",
    sid="SAMPLE_B_BATCH_2024_001",
    eid_template_name="STANDARD_ITO_PROCESS.EID"  # Reuses same template
)

# Experiment 3: Different template
exp3, processes3 = create_experiment(
    exp_id="EXP_003_SAMPLE_C",
    sample_name="High temp test on sample C",
    sid="SAMPLE_C_BATCH_2024_001", 
    eid_template_name="HIGH_TEMP_PROCESS.EID"   # Different template
)

# ===== STEP 3: TEMPLATE MANAGEMENT =====

# List all available templates
templates = list_eid_templates()
print(f"Available templates: {[t.template_name for t in templates]}")

# Get specific template
template = get_eid_template("STANDARD_ITO_PROCESS.EID")
print(f"Template: {template.template_name}, Experiments using it: {len(template.experiments)}")

# Validate template exists
try:
    validate_eid_template("NONEXISTENT.EID")
except ValueError as e:
    print(f"Template validation failed: {e}")
"""

Base = declarative_base()

# ===== RELATIONAL DATABASE TABLES (ChemOS2.0 style) =====

class Device(Base):
    __tablename__ = "device"
    id = Column(Integer, primary_key=True)
    name = Column(String(64))           # "mfc", "hotplate", "furnace", "xrd", "seccm"
    type = Column(String(64))           # "dispensing", "heating", "characterization"
    manufacturing = Column(String(64))  # "custom", "PANalytical", etc.
    location = Column(String(64))       # "sdl_lab"
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships (parent side)
    processes = relationship("Process", backref="device", lazy=True)
    results = relationship("Result", backref="device", lazy=True)
    
    def __repr__(self):
        return f"<Device {self.name}>"

class EidTemplate(Base):
    __tablename__ = "eid_template"
    id = Column(Integer, primary_key=True)
    template_name = Column(String(64), unique=True)    # "STANDARD_ITO_PROCESS.EID"
    file_path = Column(String(500))                    # Full path to EID file
    process_config = Column(JSON)                      # Process sequence configuration
    analysis_config = Column(JSON)                     # Analysis configuration
    description = Column(String(200))                  # Human-readable description
    created_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationships (parent side)
    experiments = relationship("Experiment", backref="eid_template", lazy=True)
    
    def __repr__(self):
        return f"<EidTemplate {self.template_name}>"

class Experiment(Base):
    __tablename__ = "experiment"
    id = Column(Integer, primary_key=True)
    exp_id = Column(String(64), unique=True)    # From EID filename "ITO_001"
    sid = Column(String(64))                    # Sample ID
    sample_name = Column(String(200))           # Descriptive name
    eid_template_id = Column(Integer, ForeignKey("eid_template.id"))  # FK to EID template
    status = Column(String(64))                 # "pending", "running", "completed", "failed"
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Store original EID metadata as JSON (ChemOS style!)
    eid_metadata = Column(JSON)                 # Original EID parameters for reference
    
    # Relationships (parent side)
    processes = relationship("Process", backref="experiment", order_by="Process.sequence_order", lazy=True)
    
    def __repr__(self):
        return f"<Experiment {self.exp_id}>"

class Process(Base):
    __tablename__ = "process"
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey("experiment.id"))  # FK to experiment
    device_id = Column(Integer, ForeignKey("device.id"))          # FK to device
    
    process_type = Column(String(64))           # "DISPENSE", "PRE_HEAT", "FURNACE", etc.
    sequence_order = Column(Integer)            # 1, 2, 3... (order in EID sequence)
    
    # Device settings as JSON (ChemOS approach!)
    settings = Column(JSON)                     # All device parameters
    status = Column(String(64))                 # "pending", "running", "completed", "failed"
    timestamp_start = Column(DateTime)
    timestamp_end = Column(DateTime)
    
    # Relationships (parent side for results)
    results = relationship("Result", backref="process", lazy=True)
    
    def __repr__(self):
        return f"<Process {self.process_type} #{self.sequence_order}>"

class Result(Base):
    __tablename__ = "result"
    id = Column(Integer, primary_key=True)
    process_id = Column(Integer, ForeignKey("process.id"))        # FK to process
    device_id = Column(Integer, ForeignKey("device.id"))         # FK to device (direct link)
    experiment_id = Column(Integer, ForeignKey("experiment.id")) # FK to experiment (convenience)
    
    result_type = Column(String(64))            # "xrd_scan", "seccm_data", "log_file"
    file_path = Column(String(500))             # Path to result file
    file_data = Column(LargeBinary)             # Binary data (like ChemOS)
    result_metadata = Column(JSON)              # Parsed info, analysis results (renamed from metadata)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Result {self.result_type}>"

class SECCMResult(Base):
    __tablename__ = "seccm_result"
    id = Column(Integer, primary_key=True)
    result_id = Column(Integer, ForeignKey("result.id"))          # FK to Result table (CV file)
    experiment_id = Column(Integer, ForeignKey("experiment.id"))  # FK to experiment (convenience)
    
    # Raw data information
    raw_data_path = Column(String(500))         # Path to CV .mpr file
    measurement_type = Column(String(32))       # 'cv', 'ocv', 'ci'
    
    # Soundness checking
    soundness_status = Column(String(32), default='pending')  # 'pending', 'sound', 'unsound'
    rcp_ohm = Column(Float)                     # Rcp/Ohm resistance value
    soundness_reason = Column(String(500))      # Reason if unsound
    soundness_checked_at = Column(DateTime)     # When soundness was checked
    
    # Preprocessed results
    potential_at_0_05mA = Column(Float)         # Main optimization target
    preprocessed_data = Column(JSON)            # Other extracted metrics
    preprocessing_status = Column(String(32), default='pending')  # 'pending', 'completed', 'failed'
    preprocessing_completed_at = Column(DateTime)  # When preprocessing was done
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SECCMResult exp={self.experiment_id} status={self.soundness_status}>"

class BOCampaign(Base):
    __tablename__ = "bo_campaign"
    id = Column(Integer, primary_key=True)
    campaign_name = Column(String(64))                           # e.g., "251111_01"
    eid_template_name = Column(String(64))                       # e.g., "251111_01_Iter0001.EID"
    experiment_id = Column(Integer, ForeignKey("experiment.id")) # FK to experiment
    exp_id = Column(String(64))                                  # e.g., "251007_EXP_001_ITO_0079"
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    experiment = relationship("Experiment", backref="bo_campaigns")
    
    def __repr__(self):
        return f"<BOCampaign {self.campaign_name} exp={self.exp_id}>"

@dataclass
class DispenseProcess:
    TYPE: str = 'DISPENSE'
    NAME: str = 'dispense_name'
    USE: str = 'False'
    USE_CH1: str = 'True'
    USE_CH2: str = 'True'
    CONCERNTRATION_CH1: str = '0.5'
    CONCERNTRATION_CH2: str = '0.5'
    PURGE_TIME: str = '30000'
    DISPENSING_TIME: str = '5000'
    
    def to_dict(self):
        return asdict(self)

@dataclass
class PreHeatProcess:
    TYPE: str = 'PRE_HEAT'
    NAME: str = 'pre_heat_name'
    USE: str = 'False'
    HEATING_TIME: str = '1800'
    COOLING_TIME: str = '1'
    
    def to_dict(self):
        return asdict(self)

@dataclass
class HeatTreatmentProcess:
    TYPE: str = 'HEAT_TREATMENT'
    NAME: str = 'heat_treatment_name'
    USE: str = 'False'
    SCHEDULE_NO: str = '99'
    TARGET_TEMP: str = '300'
    INCREASE: str = '50'
    HOLDING_TIME: str = '10'
    COOLING_TIME: str = '5'
    
    def to_dict(self):
        return asdict(self)

@dataclass
class PressWorkProcess:
    TYPE: str = 'PRESS_WORK'
    NAME: str = 'PRC_0003'
    USE: str = 'False'
    PRESS_POSITION: str = '172.921'
    HOLDING_TIME: str = '4'
    
    def to_dict(self):
        return asdict(self)

@dataclass
class SECCMAnalysis:
    TYPE: str = 'SECCM'
    NAME: str = 'ANS_0000'
    USE: str = 'False'
    PRE_PUMPING: str = '10'
    
    def to_dict(self):
        return asdict(self)

@dataclass
class XRDAnalysis:
    TYPE: str = 'XRD'
    NAME: str = 'ANS_0001'
    USE: str = 'False'
    PROFILE_NAME: str = 'SNU_TEST'
    
    def to_dict(self):
        return asdict(self)

# Use SQLite for now, but can be changed to PostgreSQL/MySQL by editing the URL below
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/snusdl.db'))
engine = create_engine(f'sqlite:///{DB_PATH}')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# ===== DEVICE INITIALIZATION (ChemOS2.0 style) =====
def initialize_devices():
    """Initialize SDL devices in the database if they don't exist"""
    device_configs = [
        {"name": "mfc", "type": "dispensing", "manufacturing": "custom"},
        {"name": "hotplate", "type": "preheating", "manufacturing": "custom"},
        {"name": "furnace", "type": "calcination", "manufacturing": "custom"},
        {"name": "pressor", "type": "mechanical", "manufacturing": "custom"},
        {"name": "xrd", "type": "characterization", "manufacturing": "PANalytical"},
        {"name": "seccm", "type": "characterization", "manufacturing": "BioLogics"},
        {"name": "optimizer", "type": "bayesian_optimizer", "manufacturing": "sdl_team"}  # Software "device"
    ]
    
    for device_config in device_configs:
        # Check if device already exists
        existing_device = session.query(Device).filter_by(name=device_config["name"]).first()
        if not existing_device:
            new_device = Device(
                name=device_config["name"],
                type=device_config["type"],
                manufacturing=device_config["manufacturing"],
                location="sdl_lab",
                timestamp=datetime.utcnow()
            )
            session.add(new_device)
    
    session.commit()
    print("✓ Devices initialized")

# Initialize devices on import
initialize_devices()

# ===== EID TEMPLATE MANAGEMENT FUNCTIONS =====

def create_eid_template(template_name: str, process_sequence: list, analyses: dict, 
                       eid_directory: str, description: str = ""):
    """
    Create a new EID template file and database record.
    
    Args:
        template_name: Template filename (e.g., "STANDARD_ITO_PROCESS.EID")
        process_sequence: List of process configurations
        analyses: Dict of analysis configurations
        eid_directory: Directory to save EID file
        description: Human-readable description of the template
    
    Returns:
        EidTemplate object
    """
    # Check if template already exists
    existing_template = session.query(EidTemplate).filter_by(template_name=template_name).first()
    if existing_template:
        raise ValueError(f"EID template '{template_name}' already exists")
    
    # Map process types to their dataclass constructors
    process_classes = {
        'DISPENSE': DispenseProcess,
        'PRE_HEAT': PreHeatProcess,
        'HEAT_TREATMENT': HeatTreatmentProcess,
        'PRESS_WORK': PressWorkProcess
    }
    
    # Map analysis types to their dataclass constructors
    analysis_classes = {
        'SECCM': SECCMAnalysis,
        'XRD': XRDAnalysis
    }
    
    # Build the complete parameters dictionary
    parameters = {}
    process_count = len(process_sequence)
    analysis_count = len(analyses)
    parameters['SEQUENCE_LIST'] = {
        'PROCESS_COUNT': str(process_count),
        'ANALYSIS_COUNT': str(analysis_count)
    }
    
    # Create PROCESS sections in the specified order
    for i, process_config in enumerate(process_sequence):
        section_name = f"PROCESS_{i:02d}"
        
        # Handle both dataclass instances and dictionary configs
        if hasattr(process_config, 'to_dict'):
            process_data = process_config.to_dict()
            process_data['USE'] = 'True'
        else:
            process_type = process_config['type']
            custom_params = process_config.get('params', {})
            
            if process_type not in process_classes:
                raise ValueError(f"Unknown process type: {process_type}")
            
            process_instance = process_classes[process_type](**custom_params)
            process_instance.USE = 'True'
            process_data = process_instance.to_dict()
        
        parameters[section_name] = process_data
    
    # Create ANALYSIS sections
    analysis_index = 0
    for analysis_type, analysis_config in analyses.items():
        if analysis_type not in analysis_classes:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        section_name = f"ANALYSIS_{analysis_index:02d}"
        custom_params = {k: v for k, v in analysis_config.items() if k != 'enabled'}
        analysis_instance = analysis_classes[analysis_type](**custom_params)
        
        is_enabled = analysis_config.get('enabled', True)
        analysis_instance.USE = 'True' if is_enabled else 'False'
        
        parameters[section_name] = analysis_instance.to_dict()
        analysis_index += 1
    
    # Create physical EID file
    eid_filepath = os.path.join(eid_directory, template_name)
    
    config = configparser.ConfigParser()
    config.optionxform = str
    
    for section_name, section_data in parameters.items():
        config.add_section(section_name)
        for key, value in section_data.items():
            config.set(section_name, key, str(value))
    
    # Write EID file
    with open(eid_filepath, 'w', encoding='utf-8-sig') as f:
        config.write(f)
    
    # Create database record
    # Convert dataclass objects to JSON-serializable dictionaries
    serializable_process_config = []
    for process in process_sequence:
        if hasattr(process, 'to_dict'):
            serializable_process_config.append(process.to_dict())
        else:
            serializable_process_config.append(process)  # Already a dict
    
    eid_template = EidTemplate(
        template_name=template_name,
        file_path=eid_filepath,
        process_config=serializable_process_config,  # JSON-serializable
        analysis_config=analyses,                    # Already JSON-serializable
        description=description,
        created_date=datetime.utcnow()
    )
    
    session.add(eid_template)
    session.commit()
    
    print(f"✅ Created EID template: {template_name}")
    print(f"✅ Template file: {eid_filepath}")
    print(f"✅ Template ID: {eid_template.id}")
    
    return eid_template

def get_eid_template(template_name: str):
    """Get EID template by name"""
    return session.query(EidTemplate).filter_by(template_name=template_name).first()

def list_eid_templates():
    """List all available EID templates"""
    return session.query(EidTemplate).all()

def validate_eid_template(template_name: str):
    """Check if EID template exists"""
    template = get_eid_template(template_name)
    if not template:
        raise ValueError(f"EID template '{template_name}' not found")
    return template

def delete_eid_template(template_name: str, remove_file: bool = True):
    """
    Delete EID template (if no experiments are using it)
    
    Args:
        template_name: Template name to delete
        remove_file: Whether to also delete the physical EID file
    """
    template = validate_eid_template(template_name)
    
    # Check if any experiments are using this template
    experiment_count = len(template.experiments)
    if experiment_count > 0:
        raise ValueError(f"Cannot delete template '{template_name}': {experiment_count} experiments are using it")
    
    # Remove physical file if requested
    if remove_file and os.path.exists(template.file_path):
        os.remove(template.file_path)
        print(f"✅ Deleted EID file: {template.file_path}")
    
    # Remove from database
    session.delete(template)
    session.commit()
    
    print(f"✅ Deleted EID template: {template_name}")

# ===== SIMPLIFIED EXPERIMENT CREATION =====

def create_experiment(exp_id: str, sample_name: str, sid: str, eid_template_name: str):
    """
    Create a new experiment using an existing EID template.
    
    Args:
        exp_id: Unique experiment ID for database
        sample_name: Descriptive name of the experiment
        sid: Sample ID
        eid_template_name: Name of existing EID template
    
    Returns:
        tuple: (experiment_record, process_records)
    """
    # Check if exp_id already exists
    existing_experiment = session.query(Experiment).filter_by(exp_id=exp_id).first()
    if existing_experiment:
        raise ValueError(f"Experiment ID '{exp_id}' already exists in database")
    
    # Validate EID template exists
    eid_template = validate_eid_template(eid_template_name)
    
    # Create experiment record
    experiment = add_experiment(
        exp_id=exp_id,
        sample_name=sample_name,
        sid=sid,
        eid_template_id=eid_template.id,
        eid_metadata={}  # Could store experiment-specific metadata here
    )
    

    
    print(f"✅ Created experiment: {experiment.exp_id}")
    print(f"✅ Using EID template: {eid_template_name}")
    print(f"✅ Experiment ID: {experiment.id}")

    # Create Process records from template
    template = validate_eid_template(eid_template_name)
    process_records = []
    analysis_records = []

    # Map process types to devices
    process_device_map = {
        'DISPENSE': 'mfc',
        'PRE_HEAT': 'hotplate',
        'HEAT_TREATMENT': 'furnace',
        'PRESS_WORK': 'pressor',
        'XRD': 'xrd',
        'SECCM': 'seccm'
    }

    # Create Process records
    for i, process_config in enumerate(template.process_config):
        process_type = process_config['TYPE']
        device_name = process_device_map.get(process_type, 'unknown')

        process = add_process(
            experiment_id=experiment.id,
            device_name=device_name,
            process_type=process_type,
            sequence_order=i + 1,
            settings=process_config
        )
        process_records.append(process)

    for i, (analysis_type, analysis_config) in enumerate(template.analysis_config.items()):
        if analysis_config.get('enabled', True):
            analysis_process = add_process(
                experiment_id=experiment.id,
                device_name=analysis_type.lower(),
                process_type=analysis_type.upper(),
                sequence_order=len(template.process_config) +i + 1,
                settings=analysis_config
            )
            analysis_records.append(analysis_process)
    
    return experiment, process_records, analysis_records

# ===== UPDATED FUNCTIONS FOR RELATIONAL STRUCTURE =====

def add_experiment(exp_id, sample_name, sid, eid_template_id, eid_metadata=None):
    """Add experiment to the new relational structure with EID template reference"""
    exp = Experiment(
        exp_id=exp_id,
        sid=sid,
        sample_name=sample_name,
        eid_template_id=eid_template_id,  # FK to EID template
        eid_metadata=eid_metadata or {},
        status='pending'
    )
    session.add(exp)
    session.commit()
    return exp

def add_process(experiment_id, device_name, process_type, sequence_order, settings):
    """Add a process step to an experiment"""
    # Get device by name
    device = session.query(Device).filter_by(name=device_name).first()
    if not device:
        raise ValueError(f"Device '{device_name}' not found. Available devices: {[d.name for d in session.query(Device).all()]}")
    
    process = Process(
        experiment_id=experiment_id,
        device_id=device.id,
        process_type=process_type,
        sequence_order=sequence_order,
        settings=settings,
        status='pending'
    )
    session.add(process)
    session.commit()
    return process

def add_result(process_id, device_id, experiment_id, result_type, file_path=None, file_data=None, result_metadata=None):
    """Add a result from a process"""
    result = Result(
        process_id=process_id,
        device_id=device_id,
        experiment_id=experiment_id,
        result_type=result_type,
        file_path=file_path,
        file_data=file_data,
        result_metadata=result_metadata or {}
    )
    session.add(result)
    session.commit()
    return result

def update_process_status(process_id, status, timestamp_start=None, timestamp_end=None):
    """Update process status and timestamps"""
    process = session.query(Process).filter_by(id=process_id).first()
    if process:
        process.status = status
        if timestamp_start:
            process.timestamp_start = timestamp_start
        if timestamp_end:
            process.timestamp_end = timestamp_end
        session.commit()
        return process
    return None

def update_experiment_status(experiment_id, status):
    """Update experiment status by database ID"""
    exp = session.query(Experiment).filter_by(id=experiment_id).first()
    if exp:
        exp.status = status
        session.commit()
        return exp
    return None

# ===== BACKWARDS COMPATIBILITY FUNCTIONS =====

def update_result(exp_id, result, status='complete'):
    """Legacy function - for backwards compatibility"""
    exp = session.query(Experiment).filter_by(exp_id=exp_id).first()
    if exp:
        exp.status = status
        # You might want to store result as metadata or create Result entries
        session.commit()
        return exp
    return None

def get_all_experiments():
    """Get all experiments (updated for new structure)"""
    return session.query(Experiment).all()

def get_experiment_by_id(exp_id):
    """Get experiment by ID (updated for new structure)"""
    return session.query(Experiment).filter_by(exp_id=exp_id).first()

def update_seccm_potential_at_current(experiment_id, potential_value):
    """
    Update the potential_at_0_05mA value for a SECCM result.
    
    Args:
        experiment_id: Either experiment database ID (int) or experiment name (str)
        potential_value: New potential value (float)
    
    Returns:
        SECCMResult object if successful, None if not found
    """
    # Get experiment first
    if isinstance(experiment_id, int):
        experiment = session.query(Experiment).filter_by(id=experiment_id).first()
    else:
        experiment = session.query(Experiment).filter_by(exp_id=experiment_id).first()
    
    if not experiment:
        print(f"❌ Experiment {experiment_id} not found")
        return None
    
    # Find SECCM result for this experiment
    seccm_result = session.query(SECCMResult).filter_by(experiment_id=experiment.id).first()
    
    if not seccm_result:
        print(f"❌ No SECCM result found for experiment {experiment_id}")
        return None
    
    # Update the potential value
    old_value = seccm_result.potential_at_0_05mA
    seccm_result.potential_at_0_05mA = potential_value
    session.commit()
    
    print(f"✅ Updated experiment {experiment_id} potential_at_0_05mA: {old_value} → {potential_value}")
    return seccm_result

def batch_update_seccm_potentials(experiment_potentials):
    """
    Update potential_at_0_05mA values for multiple experiments.
    
    Args:
        experiment_potentials: Dictionary mapping experiment_id to potential_value
                              e.g., {17: 0.45, 18: 0.52, 19: 0.38, 20: 0.61, 21: 0.33}
    
    Returns:
        List of successfully updated SECCMResult objects
    """
    updated_results = []
    
    print("🚀 Batch updating SECCM potential values...")
    print("=" * 50)
    
    for exp_id, potential_value in experiment_potentials.items():
        result = update_seccm_potential_at_current(exp_id, potential_value)
        if result:
            updated_results.append(result)
    
    print(f"\n📊 Successfully updated {len(updated_results)} experiments")
    return updated_results


def update_seccm_soundness_status(experiment_id, soundness_status):
    """
    Update the soundness status for a SECCM result.
    
    Args:
        experiment_id: Either experiment database ID (int) or experiment name (str)
        soundness_status: New soundness status (str)
    
    Returns:
        SECCMResult object if successful, None if not found
    """
    # Get experiment first
    if isinstance(experiment_id, int):
        experiment = session.query(Experiment).filter_by(id=experiment_id).first()
    else:
        experiment = session.query(Experiment).filter_by(exp_id=experiment_id).first()
    
    if not experiment:
        print(f"❌ Experiment {experiment_id} not found")
        return None
    
    # Find SECCM result for this experiment
    seccm_result = session.query(SECCMResult).filter_by(experiment_id=experiment.id).first()
    
    if not seccm_result:
        print(f"❌ No SECCM result found for experiment {experiment_id}")
        return None
    
    # Update the potential value
    old_value = seccm_result.soundness_status
    seccm_result.soundness_status = soundness_status
    session.commit()
    
    print(f"✅ Updated experiment {experiment_id} soundness_status: {old_value} → {soundness_status}")
    return seccm_result

def batch_update_seccm_soundness_status(experiment_soundness_status):
    """
    Update soundness status for multiple experiments.
    
    Args:
        experiment_soundness_status: Dictionary mapping experiment_id to soundness_status
                              e.g., {17: 'sound', 18: 'sound', 19: 'sound', 20: 'sound', 21: 'sound'}
    
    Returns:
        List of successfully updated SECCMResult objects
    """
    updated_results = []
    
    print("🚀 Batch updating SECCM potential values...")
    print("=" * 50)
    
    for exp_id, soundness_status in experiment_soundness_status.items():
        result = update_seccm_soundness_status(exp_id, soundness_status)
        if result:
            updated_results.append(result)
    
    print(f"\n📊 Successfully updated {len(updated_results)} experiments")
    return updated_results

# ===== TEMPLATE-BASED QUERY HELPERS =====

def get_experiments_by_template(template_name: str):
    """Get all experiments using a specific EID template"""
    template = validate_eid_template(template_name)
    return template.experiments

def get_template_usage_stats():
    """Get usage statistics for all templates"""
    from sqlalchemy import func
    return session.query(
        EidTemplate.template_name,
        EidTemplate.description,
        func.count(Experiment.id).label('experiment_count'),
        func.max(Experiment.timestamp).label('latest_experiment')
    ).outerjoin(Experiment).group_by(EidTemplate.id).all()

def get_experiment_with_template(exp_id):
    """Get experiment with its template information"""
    exp = session.query(Experiment).filter_by(exp_id=exp_id).first()
    if exp:
        return {
            'experiment': exp,
            'template': exp.eid_template,
            'template_file': exp.eid_template.file_path if exp.eid_template else None,
            'processes': exp.processes,
            'results': [result for process in exp.processes for result in process.results]
        }
    return None

# ===== UPDATED QUERY HELPERS =====

def get_experiment_with_processes(exp_id):
    """Get experiment with all its processes (updated for new structure)"""
    exp = session.query(Experiment).filter_by(exp_id=exp_id).first()
    if exp:
        return {
            'experiment': exp,
            'template': exp.eid_template,
            'processes': exp.processes,
            'results': [result for process in exp.processes for result in process.results]
        }
    return None

def get_device_by_name(device_name):
    """Get device by name"""
    return session.query(Device).filter_by(name=device_name).first()

def get_processes_by_device(device_name):
    """Get all processes for a specific device"""
    device = session.query(Device).filter_by(name=device_name).first()
    return device.processes if device else []

def get_results_by_type(result_type):
    """Get all results of a specific type (e.g., 'xrd_scan')"""
    return session.query(Result).filter_by(result_type=result_type).all()

# ===== SAMPLE-BASED QUERY HELPERS =====

def get_experiments_by_sample(sid):
    """Get all experiments for a specific sample ID"""
    return session.query(Experiment).filter_by(sid=sid).all()

def get_sample_history(sid):
    """Get complete experimental history for a sample"""
    experiments = get_experiments_by_sample(sid)
    return {
        'sample_id': sid,
        'total_experiments': len(experiments),
        'experiments': [
            {
                'exp_id': exp.exp_id,
                'sample_name': exp.sample_name,
                'status': exp.status,
                'timestamp': exp.timestamp,
                'process_count': len(exp.processes),
                'processes': [
                    {
                        'type': p.process_type,
                        'device': p.device.name,
                        'sequence': p.sequence_order,
                        'status': p.status,
                        'settings': p.settings
                    } for p in exp.processes
                ]
            } for exp in experiments
        ]
    }

def get_samples_with_experiment_counts():
    """Get all unique sample IDs with their experiment counts"""
    from sqlalchemy import func
    return session.query(
        Experiment.sid,
        func.count(Experiment.id).label('experiment_count'),
        func.max(Experiment.timestamp).label('latest_experiment')
    ).group_by(Experiment.sid).all()

def compare_experiments_on_sample(sid, exp_id_1, exp_id_2):
    """Compare two experiments performed on the same sample"""
    exp1 = session.query(Experiment).filter_by(sid=sid, exp_id=exp_id_1).first()
    exp2 = session.query(Experiment).filter_by(sid=sid, exp_id=exp_id_2).first()
    
    if not exp1 or not exp2:
        return None
    
    return {
        'sample_id': sid,
        'experiment_1': {
            'exp_id': exp1.exp_id,
            'sample_name': exp1.sample_name,
            'processes': [(p.process_type, p.settings) for p in exp1.processes]
        },
        'experiment_2': {
            'exp_id': exp2.exp_id, 
            'sample_name': exp2.sample_name,
            'processes': [(p.process_type, p.settings) for p in exp2.processes]
        }
    }

def parse_eid_file(filepath):
    config = configparser.ConfigParser()
    config.optionxform = str  # preserve case
    with open(filepath, encoding='utf-8-sig') as f:
        config.read_file(f)
    params = {section: dict(config.items(section)) for section in config.sections()}
    return params

def add_experiment_from_eid(filepath, sample_name, sid):
    exp_id = os.path.splitext(os.path.basename(filepath))[0]
    params = parse_eid_file(filepath)
    return add_experiment(
        exp_id=exp_id,
        sample_name=sample_name,
        sid=sid,
        eid_template_id=None, # This will need to be updated to use EidTemplate
        eid_metadata=params
    )

def update_process_settings(experiment_id: str, process_type: str, new_settings: dict):
    """
    Update process settings in the database
    
    Args:
        experiment_id: Experiment ID (e.g., "250915_EXP_001_ITO_79")
        process_type: Type of process (e.g., "SECCM", "DISPENSE")
        new_settings: Dictionary of settings to update (e.g., {"USE": "True"})
    """
    print("\nDEBUG: Updating process settings")
    print(f"Input new_settings: {new_settings}")
    
    # Get experiment
    experiment = session.query(Experiment).filter_by(exp_id=experiment_id).first()
    if not experiment:
        print(f"❌ Experiment {experiment_id} not found")
        return
    
    # Get process
    process = session.query(Process).filter_by(
        experiment_id=experiment.id,
        process_type=process_type
    ).first()
    
    if not process:
        print(f"❌ Process {process_type} not found for experiment {experiment_id}")
        return
    
    print(f"Current settings before update: {process.settings}")
    
    # Replace settings entirely instead of updating
    process.settings = new_settings
    
    # Save to database
    session.commit()
    
    print(f"\nFinal settings after update: {process.settings}")
    return process

def check_process_settings(experiment_id: str):
    """Show current settings for all processes in an experiment"""
    experiment = session.query(Experiment).filter_by(exp_id=experiment_id).first()
    if not experiment:
        print(f"❌ Experiment {experiment_id} not found")
        return
    
    processes = session.query(Process).filter_by(experiment_id=experiment.id).all()
    print(f"\n📋 Process settings for experiment {experiment_id}:")
    for process in processes:
        print(f"\n{process.process_type}:")
        print(process.settings)

def check_actual_process(exp_id: str, process_type: str):
    """Debug function to show exactly what's in the database"""
    experiment = session.query(Experiment).filter_by(exp_id=exp_id).first()
    if experiment:
        process = session.query(Process).filter_by(
            experiment_id=experiment.id,
            process_type=process_type
        ).first()
        if process:
            print(f"\nProcess {process_type} (ID: {process.id}):")
            print(f"Settings type: {type(process.settings)}")
            print(f"Settings content: {process.settings}")
            print(f"USE value: {process.settings.get('USE')}")
            print(f"USE type: {type(process.settings.get('USE'))}")

# ===== DEPRECATED FUNCTION =====
# The old create_sdl_experiment function has been replaced with the separated approach.
# Use create_eid_template() + create_experiment() instead.
#
# OLD WAY (deprecated):
# create_sdl_experiment(exp_id, sample_name, sid, process_sequence, analyses, eid_directory, eid_file)
#
# NEW WAY (recommended):
# Step 1: Create template (once)
# template = create_eid_template("TEMPLATE.EID", process_sequence, analyses, eid_directory)
#
# Step 2: Create experiments (many times)
# exp1, processes1 = create_experiment("EXP_001", "Sample A", "SID_A", "TEMPLATE.EID")
# exp2, processes2 = create_experiment("EXP_002", "Sample B", "SID_B", "TEMPLATE.EID")

# def create_sdl_experiment(...):  # DEPRECATED - DO NOT USE

# ===== CONVENIENCE FUNCTIONS FOR CREATING PROCESS CONFIGURATIONS =====

def create_dispense_process(name: str = "Custom Dispense", 
                          ch1_concentration: str = "0.5", 
                          ch2_concentration: str = "0.5",
                          purge_time: str = "30000",
                          dispensing_time: str = "5000") -> DispenseProcess:
    """Create a dispense process with common parameters"""
    return DispenseProcess(
        NAME=name,
        CONCERNTRATION_CH1=ch1_concentration,
        CONCERNTRATION_CH2=ch2_concentration,
        PURGE_TIME=purge_time,
        DISPENSING_TIME=dispensing_time
    )

def create_heat_treatment_process(name: str = "Custom Heat Treatment",
                                target_temp: str = "300",
                                holding_time: str = "10",
                                cooling_time: str = "5") -> HeatTreatmentProcess:
    """Create a heat treatment process with common parameters"""
    return HeatTreatmentProcess(
        NAME=name,
        TARGET_TEMP=target_temp,
        HOLDING_TIME=holding_time,
        COOLING_TIME=cooling_time
    )

def create_pre_heat_process(name: str = "Custom Pre-Heat",
                          heating_time: str = "1800") -> PreHeatProcess:
    """Create a pre-heat process with common parameters"""
    return PreHeatProcess(
        NAME=name,
        HEATING_TIME=heating_time
    )

def create_press_work_process(name: str = "Custom Press Work",
                            press_position: str = "172.921",
                            holding_time: str = "4") -> PressWorkProcess:
    """Create a press work process with common parameters"""
    return PressWorkProcess(
        NAME=name,
        PRESS_POSITION=press_position,
        HOLDING_TIME=holding_time
    )

def create_xrd_analysis(name: str = "Custom XRD",
                       profile_name: str = "SNU_TEST") -> XRDAnalysis:
    """Create an XRD analysis with common parameters"""
    return XRDAnalysis(
        NAME=name,
        PROFILE_NAME=profile_name
    )

def create_seccm_analysis(name: str = "Custom SECCM",
                         pre_pumping: str = "10") -> SECCMAnalysis:
    """Create a SECCM analysis with common parameters"""
    return SECCMAnalysis(
        NAME=name,
        PRE_PUMPING=pre_pumping
    )

def get_next_campaign_number(date_str: str) -> str:
    """
    Get the next available campaign number for a given date.
    
    Args:
        date_str: Date string in YYMMDD format (e.g., "251111")
        
    Returns:
        Two-digit campaign number string (e.g., "01", "02", "03")
    """
    # Query existing campaigns for this date
    existing_campaigns = session.query(BOCampaign).filter(
        BOCampaign.campaign_name.like(f"{date_str}_%")
    ).all()
    
    if not existing_campaigns:
        return "01"  # First campaign of the day
    
    # Extract campaign numbers and find the highest
    campaign_numbers = []
    for campaign in existing_campaigns:
        try:
            # Extract number from campaign_name like "251111_03"
            campaign_num = int(campaign.campaign_name.split('_')[1])
            campaign_numbers.append(campaign_num)
        except (IndexError, ValueError):
            continue
    
    if campaign_numbers:
        next_num = max(campaign_numbers) + 1
    else:
        next_num = 1
    
    return f"{next_num:02d}"

def add_bo_campaign_record(campaign_name: str, eid_template_name: str, 
                          experiment_id: int, exp_id: str) -> 'BOCampaign':
    """
    Add a record to track Bayesian Optimization campaign relationships.
    
    Args:
        campaign_name: Campaign name (e.g., "251111_01")
        eid_template_name: EID template name (e.g., "251111_01_Iter0001.EID")
        experiment_id: Database ID of experiment
        exp_id: Experiment ID string
        
    Returns:
        BOCampaign record
    """
    bo_campaign = BOCampaign(
        campaign_name=campaign_name,
        eid_template_name=eid_template_name,
        experiment_id=experiment_id,
        exp_id=exp_id
    )
    
    session.add(bo_campaign)
    session.commit()
    
    return bo_campaign

def get_bo_campaigns_by_date(date_str: str) -> list:
    """
    Get all BO campaigns for a specific date.
    
    Args:
        date_str: Date string in YYMMDD format
        
    Returns:
        List of BOCampaign records
    """
    return session.query(BOCampaign).filter(
        BOCampaign.campaign_name.like(f"{date_str}_%")
    ).all()

def get_bo_campaign_summary(campaign_name: str) -> dict:
    """
    Get summary of a specific BO campaign.
    
    Args:
        campaign_name: Campaign name (e.g., "251111_01")
        
    Returns:
        Dictionary with campaign statistics
    """
    campaigns = session.query(BOCampaign).filter_by(campaign_name=campaign_name).all()
    
    if not campaigns:
        return {'error': f'Campaign {campaign_name} not found'}
    
    return {
        'campaign_name': campaign_name,
        'total_experiments': len(campaigns),
        'eid_templates': list(set([c.eid_template_name for c in campaigns])),
        'experiment_ids': [c.experiment_id for c in campaigns],
        'exp_ids': [c.exp_id for c in campaigns],
        'created_at': campaigns[0].created_at,
        'latest_update': max([c.created_at for c in campaigns])
    }

def add_eid_from_experiment(exp_id, eid_name):
    print("placeholder")