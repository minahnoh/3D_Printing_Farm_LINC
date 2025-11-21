from typing import Dict, Any

"""
Basic global-level simulation parameters.
These parameters define units, simulation horizon, mode selection,
layout, and visualization limits. These are not tied to any specific
process or equipment.
"""

BASIC_DEFAULT: Dict[str, Any] = {
    
    # Unit System
    "units": {
        "currency": "KRW",
        "length": "m",
        "mass": "kg",
        "time": "min",
    },

    # Simulation Time Horizon
    "horizon": {
        "duration": "month",   # Options: "month" | "year" | "days"
        "days": None,
    },

    # Operation Flow Mode
    "flow_mode": "automated",  # Options: "automated" | "manual"


    # Factory Layout (used for visualization / structural references)
    "layout": {
        "factory_w_m": 40,
        "factory_h_m": 20,
        "printers": 2,
        "wash": 1,
        "dry": 1,
        "uv": 1,
        "stacker": 1,
        "legend": True,
    },

    # Visualization / logging limits
    "viz_max_platforms": 150,
    "viz_max_parts_per_platform": 20,
    "viz_max_events": 500_000,
}
