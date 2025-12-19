import math
import random
from typing import Dict, Any


def amr_travel_time_min(dist_m, speed_m_per_s, load_min = 0.0, unload_min = 0.0):
    """
    Total Time (min) per AMR for a single travel = Travel Time + Loading/Unloading Time
    """
    travel_sec = dist_m / max(speed_m_per_s, 1e-6)
    travel_min = travel_sec / 60.0
    return travel_min + load_min + unload_min


def manual_travel_time_min(dist_m, speed_m_per_s):
    """
    Calculating Worker Travel Time While Carrying a Platform (min)
    """
    travel_sec = dist_m / max(speed_m_per_s, 1e-6)
    return travel_sec / 60.0


def sample_stage_defects(num_good, defect_rate):
    """
    Sampling defective items at one process
    """
    if num_good <= 0 or defect_rate <= 0.0:
        return 0
    expected = num_good * defect_rate
    return int(round(expected))


def build_stacker_payload(job, platform_id: str, env_now: float) -> Dict[str, Any]:
    """
    Defining the stacker payload structure
    """
    n_parts = len(job.list_items)
    payload = {
        "job_id": job.id_job,
        "platform_id": platform_id,
        "n_parts": n_parts,
        "created_at": getattr(job, "start_time", None),
        "arrived_stacker_at": env_now,
        "good_parts_after_auto": n_parts,
        "scrapped_in_auto": 0,
        "scrapped_in_manual": 0,
        "is_scrapped_job": False,
    }
    return payload

def manual_travel_time_min(dist_m: float, speed_m_per_s: float) -> float:
    """manual travel time(min) based distance/speed"""
    if speed_m_per_s <= 0:
        return 0
    return (dist_m / speed_m_per_s) / 60.0
