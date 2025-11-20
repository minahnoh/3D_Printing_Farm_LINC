import random


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


""" Process settings """

# Maximum items in one pallet
PALLET_SIZE_LIMIT = 5

# Process time settings (in minutes)
PROC_TIME_BUILD = 180  # Process time for build (unit: minutes)
PROC_TIME_WASH = 120  # Process time for wash (unit: minutes)
PROC_TIME_DRY = 120  # Process time for dry (unit: minutes)
PROC_TIME_INSPECT = 120  # Process time for inspect per item (unit: minutes)

# Machine settings
NUM_MACHINES_BUILD = 3  # Number of 3D print machines
NUM_MACHINES_WASH = 1  # Number of wash machines
NUM_MACHINES_DRY = 1  # Number of dry machines
CAPACITY_MACHINE_BUILD = 1  # Job capacity for build machines
CAPACITY_MACHINE_WASH = 2  # Job capacity for wash machines
CAPACITY_MACHINE_DRY = 2  # Job capacity for dry machines

# Process settings
DEFECT_RATE_PROC_BUILD = 0.2  # 5% defect rate in build process
NUM_WORKERS_IN_INSPECT = 5  # Number of workers in inspection process


""" Policy settings """
# Number of defective items to collect for rework
POLICY_NUM_DEFECT_PER_JOB = 3
# Policy for placing rework jobs in queue
POLICY_REPROC_SEQ_IN_QUEUE = "QUEUE_LAST"
# Policy for extracting jobs from queue
POLICY_DISPATCH_FROM_QUEUE = "FIFO"
# Policy for dividing orders into jobs: "EQUAL_SPLIT" or "MAX_PER_JOB"
POLICY_ORDER_TO_JOB = "MAX_PER_JOB"


""" Customer settings """

# Number of patients per order


def NUM_PATIENTS_PER_ORDER(): return random.randint(
    3, 3)

# Number of items per patient


def NUM_ITEMS_PER_PATIENT(): return random.randint(
    5, 5)


# Customer settings
CUST_ORDER_CYCLE = 1 * 24 * 60  # Customer order cycle (1 week in minutes)
# Order settings
ORDER_DUE_DATE = 7 * 24 * 60  # Order due date (1 week in minutes)

""" Factory platform/post-process configuration """
# Physical number of reusable build platforms
NUM_PLATFORMS = 5  # 필요하면 바꾸기

FACTORY_PLATFORM_CLEAN = {
    "washers": 1,
    "wash_time_min": 60,
    "initial_platforms": NUM_PLATFORMS,
}

FACTORY_PREPROC = {
    "servers": 1,
    "healing_time_per_platform_min": 120,
    "placement_time_per_platform_min": 10,
    "support_time_per_platform_min": 30,
}

FACTORY_AUTO_POST = {
    "washers_m1": 1,
    "washers_m2": 1,
    "dryers": 1,
    "uv_units": 1,
    "amr_count": 0,
    "amr_speed_m_per_s": 1.0,
    "amr_load_min": 1.0,
    "amr_unload_min": 1.0,
    "t_wash1_min": 30.0,
    "t_wash2_min": 30.0,
    "t_dry_min": 30.0,
    "t_uv_min": 30.0,
    "dist_m": {},
    "defect_rate_wash": 0.0,
    "defect_rate_dry": 0.0,
    "defect_rate_uv": 0.0,
}

FACTORY_PRINT = {
    "t_print_per_platform_min": 300.0,
    "defect_rate": 0.0,
    "printers": [
        {
            "platform_capacity": 8,
            "mtbf_days": None,
            "mttr_days": None,
        }
    ],
    "max_parts_per_platform": 8,
    "install_parallel": False,
    "install_time_min": 0.0,
}

FACTORY_MANUAL_OPS = {
    "workers": 2,
    "support_time_per_platform_min": 60,
    "support_time_per_part_min": 5,
    "finish_time_per_part_min": 10,
    "paint_time_per_part_min": 10,
    "move_platform_to_support_min": 0.0,
    "move_support_to_finish_min": 0.0,
    "move_finish_to_paint_min": 0.0,
    "move_paint_to_storage_min": 0.0,
    "work_shift": {
        "start_hhmm": "09:00",
        "end_hhmm": "18:00",
        "work_cycle_days": 7,
        "workdays_per_cycle": 5,
    },
}

FACTORY_STACKER_GUARD = {
    "enabled": False,
    "max_platforms": 0,
}

FACTORY_MATERIAL = {
    "avg_part_mass_kg": 0.1,
    "max_parts_per_platform": 8,
    "resin_price_per_kg": 100.0,
}


def build_factory_param_dict():
    """
    Build a parameter dictionary P for the Factory-like platform pipeline.
    This P will be passed into the Factory class.
    """

    P = {
        "flow_mode": "automated",  # or "manual"
        "platform_clean": FACTORY_PLATFORM_CLEAN,
        "preproc": FACTORY_PREPROC,
        "auto_post": FACTORY_AUTO_POST,
        "print": FACTORY_PRINT,
        "manual_ops": FACTORY_MANUAL_OPS,
        "stacker_guard": FACTORY_STACKER_GUARD,
        "material": FACTORY_MATERIAL,
        "_horizon_minutes": SIM_TIME,
        "viz_max_platforms": 0,
        "viz_max_parts_per_platform": 0,
        "viz_max_events": 0,
        "demand": {"initial_platforms": NUM_PLATFORMS},
    }
    return P