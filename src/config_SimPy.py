import random
from typing import Dict, Any
from copy import deepcopy
from config_basic import BASIC_DEFAULT
from config_process import PROC_DEFAULT


# unified configuration dictionary
CFG = deepcopy(BASIC_DEFAULT)
CFG.update(PROC_DEFAULT)

# Logging / Visualization Flags 
EVENT_LOGGING = True
VIS_STAT_ENABLED = False
GANTT_CHART_ENABLED = True


""" Simulation settings """

# Simulation time settings
SIM_TIME = 3 * 24 * 60  # (unit: minutes)

# Printer settings
FACTORY_PRINT = CFG["print"]

# Pre-processing (before printing)
FACTORY_PREPROC = CFG["preproc"]

# Automated post-processing (wash, dry, UV, AMR)
FACTORY_AUTO_POST = CFG["auto_post"]

# Platform cleaning
FACTORY_PLATFORM_CLEAN = CFG["platform_clean"]

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

# =========================================================
# Demand / Customer-related settings  (필수: Customer/Manager가 사용)
# =========================================================

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

# Order due-date (분 단위)
ORDER_DUE_DATE = demand_cfg.get("order_due_date_min", 7 * 24 * 60)

# =========================================================
# Policy settings  (필수: manager.py가 import 함)
# =========================================================

PALLET_SIZE_LIMIT = demand_cfg["pallet_size_limit"]
POLICY_ORDER_TO_JOB = demand_cfg["policy_order_to_job"]

# 필요하면 같이 노출(다른 파일에서 import 가능성 있음)
POLICY_NUM_DEFECT_PER_JOB = demand_cfg.get("policy_num_defect_per_job")
POLICY_REPROC_SEQ_IN_QUEUE = demand_cfg.get("policy_reproc_seq_in_queue")

# job dispatch policy (현재 고정)
POLICY_DISPATCH_FROM_QUEUE = "FIFO"


# =========================================================
# Web override helpers (for FastAPI UI)
# =========================================================

def deep_merge_inplace(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge src into dst (in-place)."""
    if not src:
        return dst
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge_inplace(dst[k], v)
        else:
            dst[k] = v
    return dst


def apply_web_overrides_to_cfg(web_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Map FastAPI form payload into CFG keys and merge in-place.

    Returns the overrides dict that was applied (pruned of None).
    """
    if not isinstance(web_cfg, dict):
        return {}

    demand = web_cfg.get("demand", {}) or {}
    material = web_cfg.get("material", {}) or {}
    printing = web_cfg.get("printing", {}) or {}
    preprocess = web_cfg.get("preprocess", {}) or {}
    platform_clean = web_cfg.get("platform_clean", {}) or {}
    auto_post_common = web_cfg.get("auto_post_common", {}) or {}
    auto_post_amr = web_cfg.get("auto_post_amr", {}) or {}
    manual_transport = web_cfg.get("manual_transport", {}) or {}
    manual_post = web_cfg.get("manual_post", {}) or {}
    stacker = web_cfg.get("stacker", {}) or {}
    cost = web_cfg.get("cost", {}) or {}

    overrides: Dict[str, Any] = {
        # ---- demand ----
        "demand": {
            "order_cycle_min": demand.get("order_cycle_min"),
            "num_patients_per_order_min": demand.get("patients_min"),
            "num_patients_per_order_max": demand.get("patients_max"),
            "num_items_per_patient_min": demand.get("items_min"),
            "num_items_per_patient_max": demand.get("items_max"),
        },

        # ---- material ----
        "material": {
            "avg_part_mass_kg": (float(material.get("part_weight_g")) / 1000.0) if material.get("part_weight_g") is not None else None,
            "resin_price_per_kg": material.get("resin_cost_per_kg"),
            "max_parts_per_platform": material.get("pallet_size_limit"),
        },
        # pallet policy는 코드베이스에서 demand에 있음
        "demand__pallet_size_limit": material.get("pallet_size_limit"),

        # ---- printing ----
        "print": {
            "printer_count": printing.get("printer_count"),
            "t_print_per_platform_min": printing.get("print_time_min"),
            "defect_rate": printing.get("defect_rate"),
            "breakdown": {
                "enabled": printing.get("breakdown_enabled"),
                "mtbf_min": printing.get("mtbf_min"),
                "mttr_min": printing.get("mttr_min"),
            },
            "maintenance": {
                "enabled": True if (printing.get("maintenance_cycle_h") or printing.get("maintenance_duration_min")) else None,
                "cycle_days": (float(printing.get("maintenance_cycle_h")) / 24.0) if printing.get("maintenance_cycle_h") is not None else None,
                "duration_min": printing.get("maintenance_duration_min"),
            },
        },

        # ---- preproc ----
        "preproc": {
            "healing_time_per_platform_min": preprocess.get("healing_time"),
            "placement_time_per_platform_min": preprocess.get("placement_time"),
            "support_generation_time_min": preprocess.get("support_time"),
        },

        # ---- platform cleaning ----
        "platform_clean": {
            "washers": platform_clean.get("washer_count"),
            "wash_time_min": platform_clean.get("platform_clean_time"),
            "initial_platforms": material.get("initial_platforms"),
        },

        # ---- auto post ----
        "auto_post": {
            "washers_m1": auto_post_common.get("wash1_count"),
            "t_wash1_min": auto_post_common.get("wash1_time"),
            "washers_m2": auto_post_common.get("wash2_count"),
            "t_wash2_min": auto_post_common.get("wash2_time"),
            "dryers": auto_post_common.get("dry_count"),
            "t_dry_min": auto_post_common.get("dry_time"),
            "uv_units": auto_post_common.get("uv_count"),
            "t_uv_min": auto_post_common.get("uv_time"),
            "defect_rate_wash": auto_post_common.get("defect_wash1"),
            "defect_rate_dry": auto_post_common.get("defect_dry"),
            "defect_rate_uv": auto_post_common.get("defect_uv"),
            "amr_count": auto_post_amr.get("amr_count"),
            "amr_speed_m_per_s": auto_post_amr.get("amr_speed"),
            "amr_load_min": auto_post_amr.get("load_time"),
            "amr_unload_min": auto_post_amr.get("unload_time"),
            "dist_m": {
                "printer_to_wash1": auto_post_amr.get("dist_printer_w1"),
                "wash1_to_wash2": auto_post_amr.get("dist_w1_w2"),
                "wash_to_dry": auto_post_amr.get("dist_w2_dry"),
                "dry_to_uv": auto_post_amr.get("dist_dry_uv"),
            },
        },

        # ---- manual transport ----
        "manual_move": {
            "workers": manual_transport.get("worker_count"),
            "speed_m_per_s": manual_transport.get("walk_speed"),
            "dist_m": {
                "printer_to_wash1": manual_transport.get("dist_printer_w1"),
                "wash1_to_wash2": manual_transport.get("dist_w1_w2"),
                "wash_to_dry": manual_transport.get("dist_w2_dry"),
                "dry_to_uv": manual_transport.get("dist_dry_uv"),
            },
        },

        # ---- manual ops ----
        "manual_ops": {
            "support_removal_time_min": manual_post.get("support_remove_time"),
            "finish_time_per_part_min": manual_post.get("finish_time"),
            "paint_time_per_part_min": manual_post.get("paint_time"),
            "move_platform_to_support_min": manual_post.get("move_time"),
            "move_support_to_finish_min": manual_post.get("move_time"),
            "move_finish_to_paint_min": manual_post.get("move_time"),
            "move_paint_to_storage_min": manual_post.get("move_time"),
            "workers": manual_transport.get("worker_count"),
            "work_shift": {
                "start_hhmm": manual_transport.get("shift_start"),
                "end_hhmm": manual_transport.get("shift_end"),
                "work_cycle_days": manual_transport.get("workdays"),
            },
        },

        # ---- stacker guard ----
        "stacker_guard": {
            "enabled": stacker.get("enabled"),
            "max_platforms": stacker.get("max_wip"),
        },

        # ---- cost ----
        "cost": {
            "wage_per_hour_krw": cost.get("labor_cost_hour"),
            "equipment": {
                "printer": cost.get("printer_price"),
                "washer": cost.get("washer_price"),
                "dryer": cost.get("dryer_price"),
                "uv": cost.get("uv_price"),
                "amr": cost.get("amr_price"),
            },
            "depreciation_years": cost.get("depreciation_years"),
            "overhead_krw_per_month": cost.get("overhead_month"),
        },
    }

    # pallet_size_limit lives in CFG["demand"]
    pallet_lim = overrides.pop("demand__pallet_size_limit", None)
    if pallet_lim is not None:
        overrides.setdefault("demand", {})["pallet_size_limit"] = pallet_lim

    # prune None recursively
    def prune(obj):
        if isinstance(obj, dict):
            out = {}
            for kk, vv in obj.items():
                pv = prune(vv)
                if pv is None:
                    continue
                if isinstance(pv, dict) and len(pv) == 0:
                    continue
                out[kk] = pv
            return out
        return obj

    overrides = prune(overrides)

    # Apply to CFG (in-place) so FACTORY_* references stay valid
    deep_merge_inplace(CFG, overrides)

    return overrides
