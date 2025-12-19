import random
from typing import Dict, Any
from copy import deepcopy
from config_basic import BASIC_DEFAULT
from config_process import PROC_DEFAULT


# unified configuration dictionary 
CFG = deepcopy(BASIC_DEFAULT)
CFG.update(PROC_DEFAULT)

""" Simulation settings """

# Simulation time settings
SIM_TIME = 3 * 24 * 60  # (unit: minutes)

# Logging and visualization settings
EVENT_LOGGING = True  # Event logging enable/disable flag
DETAILED_STATS_ENABLED = True  # Detailed statistics display flag

# Visualization flags
GANTT_CHART_ENABLED = True  # Gantt chart visualization enable/disable flag
VIS_STAT_ENABLED = False  # Statistical graphs visualization enable/disable flag
SHOW_GANTT_DEBUG = False  # 기본값은 False로 설정


""" Demand / Customer-related settings """

demand_cfg = CFG["demand"]

def NUM_PATIENTS_PER_ORDER() -> int:
    """Number of patients per order from demand config."""
    return random.randint(
        demand_cfg["num_patients_per_order_min"],
        demand_cfg["num_patients_per_order_max"],
    )

def NUM_ITEMS_PER_PATIENT() -> int:
    """Number of items per patient from demand config."""
    return random.randint(
        demand_cfg["num_items_per_patient_min"],
        demand_cfg["num_items_per_patient_max"],
    )

# Customer order cycle (time between new orders, in minutes)
CUST_ORDER_CYCLE = demand_cfg["order_cycle_min"]

# Order due-date 
ORDER_DUE_DATE = demand_cfg.get("order_due_date_min", 7 * 24 * 60)


""" Policy settings """

# Maximum items in one pallet (job splitting policy)
PALLET_SIZE_LIMIT = demand_cfg["pallet_size_limit"]

# Number of defective items to collect for rework
POLICY_NUM_DEFECT_PER_JOB = demand_cfg["policy_num_defect_per_job"]

# Policy for placing rework jobs in queue
POLICY_REPROC_SEQ_IN_QUEUE = demand_cfg["policy_reproc_seq_in_queue"]

# Policy for dividing orders into jobs: "EQUAL_SPLIT" or "MAX_PER_JOB"
POLICY_ORDER_TO_JOB = demand_cfg["policy_order_to_job"]

# job dispatch policy
POLICY_DISPATCH_FROM_QUEUE = "FIFO"



""" Factory platform/post-process configuration """

# Platform cleaning (machine-based)
FACTORY_PLATFORM_CLEAN = CFG["platform_clean"]
NUM_PLATFORMS = FACTORY_PLATFORM_CLEAN.get("initial_platforms")

# Pre-processing before print
FACTORY_PREPROC = CFG["preproc"]

# Automated post-processing (AMR + automatic washers/dryer/UV)
FACTORY_AUTO_POST = CFG["auto_post"]

# Printing process configuration
FACTORY_PRINT = CFG["print"]

# Manual transport (workers carrying platforms)
FACTORY_MANUAL_MOVE = CFG["manual_move"]

# Manual post-processing (human-operated machines)
FACTORY_MANUAL_POST = CFG["manual_post"]

# Manual finishing operations (support removal, finishing, painting)
FACTORY_MANUAL_OPS = CFG["manual_ops"]

# Stacker guard / capacity constraints
FACTORY_STACKER_GUARD = CFG["stacker_guard"]

# Material / resin properties
FACTORY_MATERIAL = CFG["material"]

