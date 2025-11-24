from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class KPI:
    """
    Container for simulation performance metrics (KPI) and helper methods
    to accumulate stats and convert them into a JSON-friendly dictionary.
    """

    # Production counts
    completed_parts: int = 0
    scrapped_parts: int = 0
    completed_platforms: int = 0
    started_parts: int = 0
    started_platforms: int = 0


    # Resource busy times (minutes)
    printer_busy_min: float = 0.0
    wash1_busy_min: float = 0.0
    wash2_busy_min: float = 0.0
    dry_busy_min: float = 0.0
    uv_busy_min: float = 0.0
    amr_travel_min: float = 0.0
    platform_wash_busy_min: float = 0.0
    preproc_busy_min: float = 0.0
    manual_busy_min: float = 0.0
    pre_manual_busy_min: float = 0.0

    # Material usage
    resin_used_kg: float = 0.0
    resin_cost_krw: float = 0.0

    # Instrumentation counters
    n_preproc_jobs: int = 0
    n_print_jobs: int = 0
    n_amr_moves: int = 0

    # Wait-time statistics
    wait_sum: Dict[str, float] = field(default_factory=dict)
    wait_max: Dict[str, float] = field(default_factory=dict)
    req_count: Dict[str, int] = field(default_factory=dict)

    # Lead time / WIP statistics
    sum_platform_lead_time_min: float = 0.0
    finished_platforms: int = 0
    max_stacker_wip: int = 0


    def add_wait(self, key: str, w: float) -> None:
        """
        Accumulate waiting time statistics for a given key.
        key: logical name of the queue or waiting state (e.g., 'queue_wash1').
        w  : waiting time in minutes.
        """
        if w is None:
            return
        self.wait_sum[key] = self.wait_sum.get(key, 0.0) + w
        self.wait_max[key] = max(self.wait_max.get(key, 0.0), w)
        self.req_count[key] = self.req_count.get(key, 0) + 1


    # Export helper
    def to_dict(self, horizon_min: float, caps: Dict[str, int]) -> Dict[str, Any]:
        """
        Convert the collected KPIs into a JSON-friendly dictionary.

        horizon_min : total simulation time (minutes).
        caps        : capacity information per resource type, e.g.
                      {
                        "printers": 6,
                        "wash_m1": 1,
                        "wash_m2": 1,
                        "dryers": 1,
                        "uv_units": 1,
                        "amr": 2,
                        "platform_washers": 1,
                        "preproc_servers": 2,
                        "manual_workers": 12,
                        "manual_movers": 2
                      }
        """
        def util(busy: float, cap: int) -> float:
            if horizon_min <= 0:
                return 0.0
            return busy / (horizon_min * max(cap, 1))

        completed_plats = (
            self.finished_platforms
            if self.finished_platforms > 0
            else self.completed_platforms
        )

        total_parts = self.completed_parts + self.scrapped_parts
        yield_final = (
            self.completed_parts / total_parts
            if total_parts > 0
            else None
        )

        avg_lead_time = (
            self.sum_platform_lead_time_min / self.finished_platforms
            if self.finished_platforms > 0
            else None
        )

        wait_stats = {
            k: {
                "avg_min": (self.wait_sum[k] / max(1, self.req_count.get(k, 0))),
                "max_min": self.wait_max.get(k, 0.0),
                "req": self.req_count.get(k, 0),
            }
            for k in self.wait_sum.keys()
        }

        return {
            "completed_parts": self.completed_parts,
            "scrapped_parts": self.scrapped_parts,
            "started_parts": self.started_parts,
            "started_platforms": self.started_platforms,
            "completed_platforms": completed_plats,
            "yield_final": yield_final,
            "utilization": {
                "printers":        util(self.printer_busy_min,       caps.get("printers", 1)),
                "wash_m1":         util(self.wash1_busy_min,         caps.get("wash_m1", 0)),
                "wash_m2":         util(self.wash2_busy_min,         caps.get("wash_m2", 0)),
                "dryers":          util(self.dry_busy_min,           caps.get("dryers", 1)),
                "uv_units":        util(self.uv_busy_min,            caps.get("uv_units", 1)),
                "amr":             util(self.amr_travel_min,         caps.get("amr", 1)),
                "platform_wash":   util(self.platform_wash_busy_min, caps.get("platform_washers", 1)),
                "preproc":         util(self.preproc_busy_min,       caps.get("preproc_servers", 1)),
                "manual_workers":  util(self.manual_busy_min,        caps.get("manual_workers", 1)),
                "manual_movers":   util(self.pre_manual_busy_min,    caps.get("manual_movers", 1)),
            },
            "resin_used_kg": round(self.resin_used_kg, 3),
            "resin_cost_krw": int(self.resin_cost_krw),
            "avg_platform_lead_time_min": avg_lead_time,
            "stacker_wip_end": self.max_stacker_wip,
            "_profiling": {
                "preproc_jobs": self.n_preproc_jobs,
                "print_jobs": self.n_print_jobs,
                "amr_moves": self.n_amr_moves,
            },
            "wait_stats": wait_stats,
        }
