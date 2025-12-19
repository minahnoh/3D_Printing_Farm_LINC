from typing import Dict, Any

"""
Process-level and equipment-level simulation parameters.

This file splits parameters into:
  - MACHINE_PROC_DEFAULT : machine/automatic operations
  - HUMAN_PROC_DEFAULT   : human/manual operations
  - COMMON_PROC_DEFAULT  : neutral/common parameters (demand, material, cost, etc.)

All of them are then merged into a single PROC_DEFAULT dict.
"""

# Common process parameters (shared between human-based and machine-based flows)
COMMON_PROC_DEFAULT: Dict[str, Any] = {

    # Demand / Platform settings
    "demand": {
        # Order arrival pattern
        "order_cycle_min": 3 * 24 * 60,

        # Order size: patients and items per patient
        "num_patients_per_order_min": 1,
        "num_patients_per_order_max": 1,
        "num_items_per_patient_min": 50,
        "num_items_per_patient_max": 50,

        # Order due date
        "order_due_date_min": 7 * 24 * 60,
        "order_due_date_max": 7 * 24 * 60,

        # Job splitting & pallet policy
        "pallet_size_limit": 50,                        # PALLET_SIZE_LIMIT
        "policy_order_to_job": "MAX_PER_JOB",           # POLICY_ORDER_TO_JOB
        "policy_num_defect_per_job": 3,                 # POLICY_NUM_DEFECT_PER_JOB
        "policy_reproc_seq_in_queue": "QUEUE_LAST",     # POLICY_REPROC_SEQ_IN_QUEUE

        # setting for platform
        "install_time_min": 40,
        "install_parallel": True,
    },

    # Material / Resin properties
    "material": {
        "resin_price_per_kg": 50_000,
        "avg_part_mass_kg": 0.12,
        "max_parts_per_platform": 90,
    },

    # Cost structure
    "cost": {
        "wage_per_hour_krw": 23_000,

        "equipment": {
            "printer": 400_000_000,
            "washer": 50_000_000,
            "dryer": 10_000_000,
            "uv": 20_000_000,
            "amr": 200_000_000,
            "preproc_server": 5_000_000,
            "platform_washer": 30_000_000,
        },

        "depreciation_years": 5,
        "overhead_krw_per_month": 20_000_000,
    },

 
    # Stacker capacity constraints
    "stacker_guard": {
        "enabled": True,
        "max_platforms": 0,
    },

    # "automated": AMR Transport (24/7)
    # "manual": Manual Transport (Shift-based)
    "flow_mode": "manual", #default : "automated"
}


# Machine-based / automated processes (printers, AMRs, automatic washers, etc.)
MACHINE_PROC_DEFAULT: Dict[str, Any] = {

    # 3D Printing process parameters
    "print": {
        "t_print_per_platform_min": 1400,  # Total print time per platform
        "max_parts_per_platform": 90,
        "printer_count": 2,
        "defect_rate": 0.0,
        "install_parallel": True,
        "install_time_min": 40,

        # Individual printer-specific configurations
        "printers": [
            {"name": f"PR-{i+1}", "platform_capacity": 90, "defect_rate": 0.08}
            for i in range(6)
        ],
        
        # Printer breakdown / maintenance
        "breakdown": {
            "enabled": False,          # breakdown on:True / off:False
            "mtbf_min": 3 * 24 * 60,  # mean time breakdown failure
            "mttr_min": 4 * 60,       # mean time to repair (min)
        },

        "maintenance": {
            "enabled": False,          # maintenance on/off
            "start_hhmm": "02:00",    # start time
            "duration_min": 60,       # maintenance duration time
            "cycle_days": 1,          
        },

    },

    # Pre-processing before printing (machine-based servers)
    "preproc": {
        "servers": 2,
        "healing_time_per_platform_min": 120,
        "placement_time_per_platform_min": 10,
        "support_generation_time_min": 30,
    },

    # Automated Post-processing (AMR + equipment)
    "auto_post": {
        "amr_count": 2,
        "amr_speed_m_per_s": 1/60, # 0.1
        "amr_load_min": 20,
        "amr_unload_min": 20,

        # Distances between processing stations
        "dist_m": {
            "printer_to_wash1": 10,
            "wash1_to_wash2": 10,
            "wash_to_dry": 10,
            "dry_to_uv": 10,
            "uv_to_stacker": 10,
            "support_to_platform_wash": 10,
            "platform_wash_to_new": 10,
        },

        # Machine counts and processing times
        "washers_m1": 1,
        "washers_m2": 1,
        "t_wash1_min": 20,
        "t_wash2_min": 20,
        "dryers": 1,
        "t_dry_min": 20,
        "uv_units": 1,
        "t_uv_min": 20,

        "t_platform_install_min": 20,
        "t_platform_remove_min": 20,

        # Process-specific defect rates
        "defect_rate_wash": 0.005,  # 0.005 
        "defect_rate_dry": 0.0,     # 0.0
        "defect_rate_uv": 0.0,      # 0.0
    },

    # Platform cleaning process (machine-based)
    "platform_clean": {
        "washers": 1,
        "wash_time_min": 60,
        "initial_platforms": 20,
    },
}



# Human-based / manual processes (manual post-processing, transport, finishing)
HUMAN_PROC_DEFAULT: Dict[str, Any] = {
   
    # Manual Post-processing (human-operated machines)
    "manual_post": {
        "t_platform_install_min": 20,
        "t_platform_remove_min": 20,
        "human_unload_min": 20,
        "to_washer_travel_min": 10,
        "to_wash2_travel_min": 10,
        "t_wash1_min": 30,
        "t_wash2_min": 30,
        "to_dryer_travel_min": 10,
        "t_dry_min": 30,
        "to_stacker_travel_min": 10,
        "to_platform_wash_travel_min": 10,
        "to_newplatform_travel_min": 10,
        "t_uv_min": 30,

        # defect rate per process
        "defect_rate_wash": 0.0,
        "defect_rate_dry": 0.0,
        "defect_rate_uv": 0.0,
    },

    # Manual transport (workers carrying platforms)
    "manual_move": {
        "workers": 2,
        "speed_m_per_s": 0.1,

        "dist_m": {
            "printer_to_wash1": 10,
            "wash1_to_wash2": 10,
            "wash_to_dry": 10,
            "uv_to_stacker": 10,
            "support_to_platform_wash": 10,
            "platform_wash_to_new": 10,
        },

        "work_shift": {
            "start_hhmm": "09:00",
            "end_hhmm": "18:00",
            "work_cycle_days": 7,
            "workdays_per_cycle": 5,
        },

        "prioritize_printer": True,
    },

    # Manual finishing operations (support removal, finishing, painting)
    "manual_ops": {
        "support_removal_time_min": 20,
        "support_time_per_part_min": 2,
        "finish_time_per_part_min": 2,
        "paint_time_per_part_min": 2,

        "move_platform_to_support_min": 1,
        "move_support_to_finish_min": 1,
        "move_finish_to_paint_min": 1,
        "move_paint_to_storage_min": 1,

        "work_shift": {
            "start_hhmm": "09:00",
            "end_hhmm": "18:00",
            "work_cycle_days": 7,
            "workdays_per_cycle": 5,
        },

        "workers": 3,
    },
}


def _merge_dict(base: dict, updates: dict) -> dict:
    """Recursively merges nested dictionaries and is used to combine COMMON, MACHINE, and HUMAN process defaults into one."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value
    return base



# Final merged process config
PROC_DEFAULT: Dict[str, Any] = {}
_merge_dict(PROC_DEFAULT, COMMON_PROC_DEFAULT)
_merge_dict(PROC_DEFAULT, MACHINE_PROC_DEFAULT)
_merge_dict(PROC_DEFAULT, HUMAN_PROC_DEFAULT)
