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
    started_platforms: int = 0
    completed_platforms: int = 0
    started_parts: int = 0

    # Busy time accumulators (minutes)
    preproc_busy_min: float = 0.0
    print_busy_min: float = 0.0
    printer_busy_min: float = 0.0
    wash1_busy_min: float = 0.0
    wash2_busy_min: float = 0.0
    dry_busy_min: float = 0.0
    uv_busy_min: float = 0.0
    platform_wash_busy_min: float = 0.0
    manual_busy_min: float = 0.0
    pre_manual_busy_min: float = 0.0

    # Material usage
    resin_used_kg: float = 0.0
    resin_cost_krw: float = 0.0

    # Instrumentation counters
    n_preproc_jobs: int = 0
    n_print_jobs: int = 0
    n_amr_moves: int = 0
    amr_travel_min: float = 0.0

    # Wait-time statistics
    wait_sum: Dict[str, float] = field(default_factory=dict)
    wait_max: Dict[str, float] = field(default_factory=dict)
    req_count: Dict[str, int] = field(default_factory=dict)

    # Lead time / WIP statistics
    sum_platform_lead_time_min: float = 0.0
    finished_platforms: int = 0
    max_stacker_wip: int = 0

    # Extra metrics for web UI
    scrap_by_stage: Dict[str, int] = field(default_factory=dict)
    amr_route_counts: Dict[str, int] = field(default_factory=dict)
    stacker_wip_history: list = field(default_factory=list)

    def add_wait(self, key: str, w: float) -> None:
        """
        Accumulate waiting time statistics for a given key.
        key: logical name of the queue or waiting state (e.g., 'queue_wash1').
        w  : waiting time in minutes.
        """
        if w is None:
            return
        w = float(w)
        self.wait_sum[key] = self.wait_sum.get(key, 0.0) + w
        self.req_count[key] = self.req_count.get(key, 0) + 1
        self.wait_max[key] = max(self.wait_max.get(key, 0.0), w)

    def add_scrap(self, stage: str, count: int) -> None:
        """Accumulate scrap count by logical stage name."""
        if not stage:
            stage = "unknown"
        c = int(max(0, count or 0))
        if c <= 0:
            return
        self.scrap_by_stage[stage] = int(self.scrap_by_stage.get(stage, 0) + c)

    def add_amr_route(self, route_key: str, n: int = 1) -> None:
        """Accumulate AMR route traversal counts."""
        if not route_key:
            route_key = "unknown"
        self.amr_route_counts[route_key] = int(self.amr_route_counts.get(route_key, 0) + int(n or 0))

    def record_stacker_wip(self, t_min: float, wip: int) -> None:
        """Append stacker WIP time series for web plotting."""
        try:
            self.stacker_wip_history.append({"t": float(t_min), "wip": int(wip)})
        except Exception:
            pass

    def to_dict(self, horizon_min: float, caps: Dict[str, int]) -> Dict[str, Any]:
        """
        Convert KPI into a JSON-friendly dict.

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
            else 0.0
        )

        avg_lead_time = (
            self.sum_platform_lead_time_min / completed_plats
            if completed_plats > 0
            else None
        )

        wait_stats = {}
        for k, s in self.wait_sum.items():
            n = self.req_count.get(k, 0)
            wait_stats[k] = {
                "avg": (s / n) if n > 0 else None,
                "max": self.wait_max.get(k, None),
                "n": n,
            }

        return {
            "completed_parts": int(self.completed_parts),
            "scrapped_parts": int(self.scrapped_parts),
            "completed_platforms": int(self.completed_platforms),
            "finished_platforms": int(self.finished_platforms),
            "yield_final": round(float(yield_final), 4),
            "utilization": {
                "preproc":         util(self.preproc_busy_min,       caps.get("preproc_servers", 1)),
                "printers":        util(self.print_busy_min,         caps.get("printers", 1)),
                "wash_m1":         util(self.wash1_busy_min,         caps.get("wash_m1", 1)),
                "wash_m2":         util(self.wash2_busy_min,         caps.get("wash_m2", 1)),
                "dryers":          util(self.dry_busy_min,           caps.get("dryers", 1)),
                "uv_units":        util(self.uv_busy_min,            caps.get("uv_units", 1)),
                "platform_wash":   util(self.platform_wash_busy_min, caps.get("platform_washers", 1)),
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
            "scrap_by_stage": dict(self.scrap_by_stage),
            "amr_route_counts": dict(self.amr_route_counts),
            "stacker_wip_history": list(self.stacker_wip_history),
        }
