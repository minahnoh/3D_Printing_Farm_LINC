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