
# des_factory_web.py (enhanced DES: preproc, shifts, KPIs, UI)
# -------------------------------------------------------------
# FastAPI + SimPy DES (KRW/m/kg/min) with structured GUI and safe deep-merge.
# -------------------------------------------------------------

import random
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from copy import deepcopy
from collections import defaultdict

import simpy
from simpy.resources.resource import Preempted
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ------------------------------
# Defaults
# ------------------------------

DEFAULT: Dict[str, Any] = {
    "units": {"currency": "KRW", "length": "m", "mass": "kg", "time": "min"},
    "horizon": {"duration": "month", "days": None},  # month|year|days
    "demand": {
        "policy": "infinite",
        "platform_interarrival_min_min": 30,
        "platform_interarrival_max_min": 30,
        "initial_platforms": 20,
        "install_time_min": 40,
        "install_parallel": True
    },
    "material": {
        "resin_price_per_kg": 50_000,    # KRW/kg
        "avg_part_mass_kg": 0.12,
        "max_parts_per_platform": 90
    },
    "print": {
        "t_print_per_platform_min": 1_400,
        "max_parts_per_platform": 90,
        "printer_count": 6,
        "defect_rate": 0.08,
        "install_parallel": True,
        "install_time_min": 40,
        "printers": [
            {
                "name": "PR-1",
                "platform_capacity": 90,
                "defect_rate": 0.08
            },
            {
                "name": "PR-2",
                "platform_capacity": 90,
                "defect_rate": 0.08
            },
            {
                "name": "PR-3",
                "platform_capacity": 90,
                "defect_rate": 0.08
            },
            {
                "name": "PR-4",
                "platform_capacity": 90,
                "defect_rate": 0.08
            },
            {
                "name": "PR-5",
                "platform_capacity": 90,
                "defect_rate": 0.08
            },
            {
                "name": "PR-6",
                "platform_capacity": 90,
                "defect_rate": 0.08
            }
        ]
    },
    "flow_mode": "automated",  # automated | manual
    "preproc": {
        "servers": 2,
        "healing_time_per_platform_min": 120,   # 분/빌드플랫폼 (디폴트 2시간)
        "placement_time_per_platform_min": 10,  # 분/빌드플랫폼
        "support_time_per_platform_min": 30     # 분/빌드플랫폼 (전처리 서포트 생성)
    },
    "auto_post": {
        "amr_count": 2,
        "amr_speed_m_per_s": 0.1,
        "amr_load_min": 20,
        "amr_unload_min": 20,
        "dist_m": {
            "printer_to_wash": 10,
            "wash_to_dry": 10,
            "dry_to_uv": 10,
            "uv_to_stacker": 10,
            "support_to_platform_wash": 10,
            "platform_wash_to_new": 10
        },
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
        "defect_rate_wash": 0.005,
        "defect_rate_dry": 0.001,
        "defect_rate_uv": 0
    },
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
        "defect_rate_wash": 0.0,
        "defect_rate_dry": 0.0,
        "defect_rate_uv": 0.0
    },
    "manual_move": {
        "workers": 2,
        "speed_m_per_s": 0.1,
        "dist_m": {
            "printer_to_wash": 10,
            "wash1_to_wash2": 10,
            "wash_to_dry": 10,
            "uv_to_stacker": 10,
            "support_to_platform_wash": 10,
            "platform_wash_to_new": 10
        },
        "work_shift": {"start_hhmm":"09:00","end_hhmm":"18:00","work_cycle_days":7,"workdays_per_cycle":5},
        "prioritize_printer": True
    },
    "manual_ops": {
        "support_time_per_platform_min": 20,
        "support_time_per_part_min": 2,
        "finish_time_per_part_min": 2,
        "paint_time_per_part_min": 2,
        "move_platform_to_support_min": 1,
        "move_support_to_finish_min": 1,
        "move_finish_to_paint_min": 1,
        "move_paint_to_storage_min": 1,
        "work_shift": {"start_hhmm":"09:00","end_hhmm":"18:00","work_cycle_days":7,"workdays_per_cycle":5},
        "workers": 12
    },
    "platform_clean": {
        "washers": 1,
        "wash_time_min": 60,
        "initial_platforms": 20
    },
    "cost": {
        "wage_per_hour_krw": 23000,
        "equipment": {
            "printer": 400_000_000,
            "washer": 50_000_000,
            "dryer": 10_000_000,
            "uv": 20_000_000,
            "amr": 200_000_000,
            "preproc_server": 5_000_000,
            "platform_washer": 30_000_000
        },
        "depreciation_years": 5,
        "overhead_krw_per_month": 20_000_000
    },
    "stacker_guard": {
        "enabled": False,
        "max_platforms": 0
    },
    "layout": {
        "factory_w_m": 40, "factory_h_m": 20,
        "printers": 2, "wash": 1, "dry": 1, "uv": 1, "stacker": 1,
        "legend": True
    },
    "viz_max_platforms": 150,
    "viz_max_parts_per_platform": 20,
    "viz_max_events": 500000
}

# ------------------------------
# Helpers
# ------------------------------

def tri(a,m,b): return random.triangular(a,b,m)
def logn(mu,sigma): return random.lognormvariate(mu, sigma)
def minutes(days=0,hours=0,mins=0): return int(days*24*60 + hours*60 + mins)

def deep_merge(dst: dict, src: dict):
    """Recursively merge src into dst (dicts only)."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

def normalize_printer_count(P: Dict[str,Any]):
    """If print.printer_count is provided, resize the printers list using the first profile as template."""
    pr = P.get("print", {})
    n = pr.get("printer_count")
    if not n or n <= 0:
        return
    base_list = pr.get("printers") or []
    if not base_list:
        return
    tpl = deepcopy(base_list[0])
    # derive a clean prefix for naming
    name = str(tpl.get("name", "PRINTER"))
    prefix = name
    if "-" in name and name.split("-")[-1].isdigit():
        prefix = "-".join(name.split("-")[:-1])
    new_list = []
    for i in range(n):
        item = deepcopy(tpl)
        item["name"] = f"{prefix}-{i+1}"
        # apply global defect rate if provided
        if pr.get("defect_rate") is not None:
            item["defect_rate"] = float(pr["defect_rate"])
        # apply global platform capacity override if provided
        if pr.get("max_parts_per_platform") is not None:
            try:
                item["platform_capacity"] = int(pr["max_parts_per_platform"]) or item.get("platform_capacity", 0)
            except Exception:
                pass
        new_list.append(item)
    pr["printers"] = new_list

def apply_global_printer_defaults(P: Dict[str,Any]):
    pr = P.get("print", {})
    if pr.get("printers"):
        if pr.get("defect_rate") is not None:
            for prof in pr["printers"]:
                prof["defect_rate"] = float(pr["defect_rate"])
        if pr.get("max_parts_per_platform") is not None:
            try:
                cap = int(pr["max_parts_per_platform"]) 
                for prof in pr["printers"]:
                    prof["platform_capacity"] = cap
            except Exception:
                pass

# ------------------------------
# KPI
# ------------------------------

@dataclass
class KPI:
    completed_parts: int = 0
    scrapped_parts: int = 0
    completed_platforms: int = 0
    started_parts: int = 0
    started_platforms: int = 0
    printer_busy_min: float = 0.0
    wash1_busy_min: float = 0.0
    wash2_busy_min: float = 0.0
    dry_busy_min: float = 0.0
    uv_busy_min: float = 0.0
    amr_travel_min: float = 0.0
    platform_wash_busy_min: float = 0.0
    resin_used_kg: float = 0.0
    resin_cost_krw: float = 0.0
    preproc_busy_min: float = 0.0
    manual_busy_min: float = 0.0
    pre_manual_busy_min: float = 0.0
    # instrumentation
    n_preproc_jobs: int = 0
    n_print_jobs: int = 0
    n_amr_moves: int = 0
    # wait-time stats
    wait_sum: Dict[str, float] = field(default_factory=dict)
    wait_max: Dict[str, float] = field(default_factory=dict)
    req_count: Dict[str, int] = field(default_factory=dict)

    def add_wait(self, key: str, w: float):
        if w is None: return
        self.wait_sum[key] = self.wait_sum.get(key, 0.0) + w
        self.wait_max[key] = max(self.wait_max.get(key, 0.0), w)
        self.req_count[key] = self.req_count.get(key, 0) + 1
    sum_platform_lead_time_min: float = 0.0
    finished_platforms: int = 0
    max_stacker_wip: int = 0

    def to_dict(self, horizon_min: float, caps: Dict[str,int]) -> Dict[str,Any]:
        def util(busy, cap): return (busy/(horizon_min*max(cap,1))) if horizon_min>0 else 0.0
        return {
            "completed_parts": self.completed_parts,
            "scrapped_parts": self.scrapped_parts,
            "started_parts": self.started_parts,
            "started_platforms": self.started_platforms,
            "completed_platforms": self.finished_platforms if self.finished_platforms>0 else self.completed_platforms,
            "yield_final": (self.completed_parts/(self.completed_parts+self.scrapped_parts)) if (self.completed_parts+self.scrapped_parts)>0 else None,
            "utilization": {
                "printers": util(self.printer_busy_min, caps.get("printers",1)),
                "wash_m1": util(self.wash1_busy_min, caps.get("wash_m1",0)),
                "wash_m2": util(self.wash2_busy_min, caps.get("wash_m2",0)),
                "dryers": util(self.dry_busy_min, caps.get("dryers",1)),
                "uv_units": util(self.uv_busy_min, caps.get("uv_units",1)),
                "amr": util(self.amr_travel_min, caps.get("amr",1)),
                "platform_wash": util(self.platform_wash_busy_min, caps.get("platform_washers",1)),
                "preproc": util(self.preproc_busy_min, caps.get("preproc_servers",1)),
                "manual_workers": util(self.manual_busy_min, caps.get("manual_workers",1)),
                "manual_movers": util(self.pre_manual_busy_min, caps.get("manual_movers",1))
            },
            "resin_used_kg": round(self.resin_used_kg,3),
            "resin_cost_krw": int(self.resin_cost_krw),
            "avg_platform_lead_time_min": (self.sum_platform_lead_time_min/self.finished_platforms) if self.finished_platforms>0 else None,
            "stacker_wip_end": self.max_stacker_wip,
            "_profiling": {
                "preproc_jobs": self.n_preproc_jobs,
                "print_jobs": self.n_print_jobs,
                "amr_moves": self.n_amr_moves
            },
            "wait_stats": {
                k: {
                    "avg_min": (self.wait_sum[k]/max(1, self.req_count.get(k,0))),
                    "max_min": self.wait_max.get(k,0.0),
                    "req": self.req_count.get(k,0)
                } for k in self.wait_sum.keys()
            }
        }

# ------------------------------
# Factory
# ------------------------------

class Factory:
    def __init__(self, env: simpy.Environment, P: Dict[str,Any], kpi: KPI):
        self.env=env; self.P=P; self.kpi=kpi

        # Washers are split into Mode1 and Mode2 (must be pairs)
        cap_w1 = int(P["auto_post"].get("washers_m1", 1))
        cap_w2 = int(P["auto_post"].get("washers_m2", 1))
        self.washers_m1 = simpy.Resource(env, capacity=cap_w1)
        self.washers_m2 = simpy.Resource(env, capacity=cap_w2)
        self.dryers  = simpy.Resource(env, capacity=P["auto_post"]["dryers"])
        self.uv      = simpy.Resource(env, capacity=P["auto_post"]["uv_units"])
        self.preproc = simpy.Resource(env, capacity=P["preproc"]["servers"])
        self.pre_in = simpy.Store(env, capacity=9999)
        self.ready = simpy.Store(env, capacity=9999)
        # New platform supply is considered effectively infinite (no blocking on new platform availability)
        self.new_plat = None
        manual_mover_cap = int(P.get("manual_move",{}).get("workers", 2) or 0)
        if manual_mover_cap <= 0:
            manual_mover_cap = 1
        self.prioritize_printer = bool(P.get("manual_move",{}).get("prioritize_printer", False))
        self.manual_movers = simpy.PriorityResource(env, capacity=manual_mover_cap)
        self.manual_move_gate = simpy.PriorityResource(env, capacity=manual_mover_cap)

        self.printers = [simpy.Resource(env, capacity=1) for _ in P["print"]["printers"]]
        # Keep track of held maintenance requests to block printers
        self._printer_locks = [None for _ in P["print"]["printers"]]
        for i,prof in enumerate(P["print"]["printers"]):
            env.process(self._maintenance_toggle(i, prof))
            env.process(self._random_breakdowns(i, prof))

        # Large stacker buffer so upstream never blocks due to manual backlog
        self.stacker = simpy.Store(env, capacity=100_000)
        plat_cfg = P.get("platform_clean", {})
        self.platform_wash = simpy.Resource(env, capacity=max(1, int(plat_cfg.get("washers", 1) or 1)))
        self.platform_wash_time = float(plat_cfg.get("wash_time_min", 60) or 0)
        demand_init = int(P.get("demand", {}).get("initial_platforms", 0) or 0)
        clean_init = int(plat_cfg.get("initial_platforms", 0) or 0)
        initial_plat = max(demand_init, clean_init)
        if initial_plat <= 0:
            initial_plat = 1
        self.clean_platforms = simpy.Store(env, capacity=initial_plat)
        self._platform_tokens = []
        for i in range(initial_plat):
            token = {"id": f"PLAT-{i+1}", "cycle": 0}
            self._platform_tokens.append(token)
            self.clean_platforms.items.append(token)
        guard_cfg = P.get("stacker_guard", {})
        self.stacker_guard_enabled = bool(guard_cfg.get("enabled", False))
        try:
            limit = int(guard_cfg.get("max_platforms", 0) or 0)
        except Exception:
            limit = 0
        self.stacker_guard_limit = max(0, limit)
        self._stacker_guard_event = None
        self._active_prints = 0
        self.amr = simpy.Resource(env, capacity=P["auto_post"]["amr_count"])
        self.manual_workers = P["manual_ops"].get("workers", 2)
        # Manual worker pool (shared across support/finish/paint). Ensures one person cannot do multiple tasks simultaneously.
        self.manual_pool = simpy.PreemptiveResource(env, capacity=self.manual_workers)
        # Visualization trace buffer
        self.trace = []
        self._trace_limit = int(P.get('viz_max_platforms', 0) or 0)
        self._trace_part_cap = int(P.get('viz_max_parts_per_platform', 0) or 0)
        self._trace_event_cap = int(P.get('viz_max_events', 0) or 0)
        self._trace_seen = set()
        self._trace_order = []
        self._trace_part_allowed = defaultdict(set)
        self.horizon_minutes = int(P.get('_horizon_minutes', 0) or 0)
        self.support_gate = simpy.Resource(env, capacity=1)
        self._stacker_entries: Dict[str, Dict[str,Any]] = {}
        self.install_parallel = bool(P.get("print",{}).get("install_parallel", False))
        install_cap = len(self.printers) if self.install_parallel else 1
        self.install_gate = simpy.Resource(env, capacity=max(1, install_cap))
        self.install_time_min = float(P.get("print",{}).get("install_time_min", 0.0) or 0.0)
        amr_count = int(P["auto_post"].get("amr_count", 0) or 0)
        self._amr_available = list(range(amr_count))
        self._next_amr_idx = amr_count
        self._amr_idle_entries: Dict[str, Dict[str,float]] = {}
        for idx in self._amr_available:
            name = f"AMR-{idx+1}"
            entry = self._log(name, 'amr_idle', 0.0, 0.0)
            if entry is not None:
                self._amr_idle_entries[name] = entry
        amr_count = int(P["auto_post"].get("amr_count", 0) or 0)
        self._amr_available = list(range(amr_count))
        self._next_amr_idx = amr_count
        self._amr_idle_entries: Dict[str, Dict[str,float]] = {}
        for idx in self._amr_available:
            name = f"AMR-{idx+1}"
            entry = self._log(name, 'amr_idle', 0.0, 0.0)
            if entry is not None:
                self._amr_idle_entries[name] = entry

    def _trace_enabled(self, platform_id:str)->bool:
        limit = self._trace_limit
        is_part = '-P' in platform_id
        is_amr = '-AMR' in platform_id
        base = platform_id
        if is_part:
            base = platform_id.split('-P')[0]
        elif is_amr:
            base = platform_id.split('-AMR')[0]

        if is_part and self._trace_part_cap > 0:
            allow = self._trace_part_allowed[base]
            if platform_id not in allow:
                if len(allow) >= self._trace_part_cap:
                    return False
                allow.add(platform_id)
        elif is_part:
            self._trace_part_allowed[base].add(platform_id)

        if limit <= 0:
            self._trace_seen.add(base)
            return True
        if base in self._trace_seen:
            return True
        if len(self._trace_seen) >= limit:
            return False
        self._trace_seen.add(base)
        self._trace_order.append(base)
        return True

    def _stacker_guard_load(self) -> int:
        if not self.stacker_guard_enabled or self.stacker_guard_limit <= 0:
            return 0
        return len(self.stacker.items) + self._active_prints

    def _stacker_guard_wait(self):
        if not self.stacker_guard_enabled or self.stacker_guard_limit <= 0:
            return
        while True:
            if self._stacker_guard_load() < self.stacker_guard_limit:
                self._stacker_guard_inc_active()
                return
            wait_start = self.env.now
            if self._stacker_guard_event is None or self._stacker_guard_event.triggered:
                self._stacker_guard_event = self.env.event()
            yield self._stacker_guard_event
            if hasattr(self.kpi, 'add_wait'):
                self.kpi.add_wait('stacker_guard', self.env.now - wait_start)

    def _stacker_guard_notify(self):
        if self._stacker_guard_event and not self._stacker_guard_event.triggered:
            if self._stacker_guard_load() < self.stacker_guard_limit:
                self._stacker_guard_event.succeed()
                self._stacker_guard_event = None

    def _stacker_guard_inc_active(self):
        if self.stacker_guard_enabled and self.stacker_guard_limit > 0:
            self._active_prints += 1

    def _stacker_guard_dec_active(self):
        if self.stacker_guard_enabled and self.stacker_guard_limit > 0 and self._active_prints > 0:
            self._active_prints -= 1
            self._stacker_guard_notify()

    def _platform_clean_cycle(self, platform_id:str, token:Optional[Dict[str,Any]]):
        if not token:
            return
        automated = self.P.get("flow_mode", "automated") == "automated"
        env = self.env
        if automated:
            with self.amr.request() as amr_rq:
                t_req = env.now; yield amr_rq
                if hasattr(self.kpi, 'add_wait'):
                    self.kpi.add_wait('amr', env.now - t_req)
                start = env.now
                travel = self.amr_travel_min("support_to_platform_wash")
                if travel > 0:
                    yield env.timeout(travel)
                    self.kpi.amr_travel_min += travel
                    self.kpi.n_amr_moves += 1
                self._log(platform_id, 'to_platform_wash', start, env.now)
                self._log_amr(platform_id, 'to_platform_wash', start, env.now)
        else:
            while True:
                if not self._in_shift_move(int(env.now)):
                    yield env.timeout(self._time_to_next_shift_start_move(int(env.now)))
                    continue
                priority = self._move_priority('to_platform_wash')
                with self.manual_move_gate.request(priority=priority) as gate_rq:
                    yield gate_rq
                    if not self._in_shift_move(int(env.now)):
                        continue
                    with self.manual_movers.request(priority=priority) as wrq:
                        t_req = env.now
                        yield wrq
                        if hasattr(self.kpi, 'add_wait'):
                            self.kpi.add_wait('manual_movers', env.now - t_req)
                        move_start = env.now
                        travel = self.manual_route_min("support_to_platform_wash")
                        if travel > 0:
                            yield from self._work_with_shift_move(travel)
                        self._log(platform_id, 'to_platform_wash', move_start, env.now)
                        self._log_manual(platform_id, 'to_platform_wash', move_start, env.now)
                        break
                break

        queue = self._begin_stage(platform_id, 'platform_wash_queue')
        with self.platform_wash.request() as rq:
            t_req = env.now; yield rq
            self._finish_stage(queue)
            if hasattr(self.kpi, 'add_wait'):
                self.kpi.add_wait('platform_wash', env.now - t_req)
            proc = self._begin_stage(platform_id, 'platform_wash')
            wash_time = self.t_platform_wash()
            if wash_time > 0:
                yield env.timeout(wash_time)
                self.kpi.platform_wash_busy_min += wash_time
            self._finish_stage(proc)

        if automated:
            with self.amr.request() as amr_rq:
                t_req = env.now; yield amr_rq
                if hasattr(self.kpi, 'add_wait'):
                    self.kpi.add_wait('amr', env.now - t_req)
                start = env.now
                travel = self.amr_travel_min("platform_wash_to_new")
                if travel > 0:
                    yield env.timeout(travel)
                    self.kpi.amr_travel_min += travel
                    self.kpi.n_amr_moves += 1
                self._log(platform_id, 'to_new_platform', start, env.now)
                self._log_amr(platform_id, 'to_new_platform', start, env.now)
        else:
            while True:
                if not self._in_shift_move(int(env.now)):
                    yield env.timeout(self._time_to_next_shift_start_move(int(env.now)))
                    continue
                priority = self._move_priority('to_new_platform')
                with self.manual_move_gate.request(priority=priority) as gate_rq:
                    yield gate_rq
                    if not self._in_shift_move(int(env.now)):
                        continue
                    with self.manual_movers.request(priority=priority) as wrq:
                        t_req = env.now
                        yield wrq
                        if hasattr(self.kpi, 'add_wait'):
                            self.kpi.add_wait('manual_movers', env.now - t_req)
                        move_start = env.now
                        travel = self.manual_route_min("platform_wash_to_new")
                        if travel > 0:
                            yield from self._work_with_shift_move(travel)
                        self._log(platform_id, 'to_new_platform', move_start, env.now)
                        self._log_manual(platform_id, 'to_new_platform', move_start, env.now)
                        break
                break

        yield self.clean_platforms.put(token)

    def _log(self, platform_id:str, stage:str, t0:float, t1:float):
        try:
            if not self._trace_enabled(platform_id):
                return None
            if self._trace_event_cap and len(self.trace) >= self._trace_event_cap:
                return None
            entry = {"id": platform_id, "stage": stage, "t0": float(t0), "t1": float(t1)}
            self.trace.append(entry)
            return entry
        except Exception:
            return None

    def _log_amr(self, platform_id:str, stage:str, t0:float, t1:float):
        return self._log(f"{platform_id}-AMR", stage, t0, t1)

    def _log_manual(self, platform_id:str, stage:str, t0:float, t1:float):
        return self._log(f"{platform_id}-MOV", stage, t0, t1)

    def _begin_stage(self, platform_id:str, stage:str):
        horizon = float(self.horizon_minutes or (self.env.now + 1_000_000))
        entry = self._log(platform_id, stage, self.env.now, horizon)
        return entry

    def _finish_stage(self, entry:Optional[Dict[str,float]]):
        if entry is not None:
            entry['t1'] = float(self.env.now)

    def _amr_begin(self):
        if self._amr_available:
            idx = self._amr_available.pop(0)
        else:
            idx = self._next_amr_idx
            self._next_amr_idx += 1
        name = f"AMR-{idx+1}"
        idle_entry = self._amr_idle_entries.pop(name, None)
        if idle_entry:
            idle_entry['t1'] = float(self.env.now)
        return idx, name

    def _amr_end(self, idx:int, name:str):
        entry = self._log(name, 'amr_idle', self.env.now, self.env.now)
        if entry is not None:
            self._amr_idle_entries[name] = entry
        self._amr_available.append(idx)
        self._amr_available.sort()

    def _maintenance_toggle(self, idx:int, prof:Dict[str,Any]):
        every = prof.get("maintenance_every_days")
        dur = prof.get("maintenance_duration_days")
        start = prof.get("maintenance_start_day", 0)
        if every and dur:
            t0=minutes(days=start)
            if t0>0: yield self.env.timeout(t0)
            while True:
                # occupy the printer to simulate downtime
                if self._printer_locks[idx] is None:
                    req = self.printers[idx].request()
                    yield req
                    self._printer_locks[idx] = req
                yield self.env.timeout(minutes(days=dur))
                # release the lock
                if self._printer_locks[idx] is not None:
                    self.printers[idx].release(self._printer_locks[idx])
                    self._printer_locks[idx] = None
                yield self.env.timeout(minutes(days=every))

    def _random_breakdowns(self, idx:int, prof:Dict[str,Any]):
        mtbf=prof.get("mtbf_days"); mttr=prof.get("mttr_days")
        if not mtbf or not mttr: return
        # Simple exponential time-to-failure / fixed repair duration
        while True:
            # time to next failure (in minutes)
            ttf_days = random.expovariate(1.0/mtbf)
            yield self.env.timeout(minutes(days=ttf_days))
            # occupy the printer until repaired
            req = self.printers[idx].request()
            yield req
            yield self.env.timeout(minutes(days=mttr))
            self.printers[idx].release(req)

    # samples
    def t_preproc(self):
        pre = self.P["preproc"]
        return float(pre.get("healing_time_per_platform_min",120)) \
             + float(pre.get("placement_time_per_platform_min",10)) \
             + float(pre.get("support_time_per_platform_min",30))
    def amr_travel_min(self, key:str):
        ap=self.P["auto_post"]; dist=ap.get("dist_m",{}).get(key, ap.get("dist_m",{}).get("wash_to_dry",10)); v=ap.get("amr_speed_m_per_s",1.0)
        return (dist/v)/60.0 + ap["amr_load_min"] + ap["amr_unload_min"]
    def manual_move_speed(self):
        mm = self.P.get("manual_move", {})
        return max(1e-6, float(mm.get("speed_m_per_s", 2.0) or 0.0))
    def manual_route_min(self, key:str):
        mp = self.P.get("manual_post", {})
        overrides = {
            "printer_to_wash": mp.get("to_washer_travel_min"),
            "wash1_to_wash2": mp.get("to_wash2_travel_min"),
            "wash_to_dry": mp.get("to_dryer_travel_min"),
            "uv_to_stacker": mp.get("to_stacker_travel_min"),
            "support_to_platform_wash": mp.get("to_platform_wash_travel_min"),
            "platform_wash_to_new": mp.get("to_newplatform_travel_min")
        }
        dist_val = overrides.get(key)
        if dist_val is None:
            dist_val = self.P.get("manual_move", {}).get("dist_m", {}).get(key, 0.0)
        try:
            dist = float(dist_val or 0.0)
        except Exception:
            dist = 0.0
        speed = self.manual_move_speed()
        travel = (dist / speed) / 60.0 if dist > 0 else 0.0
        install = float(mp.get("t_platform_install_min", 0.0) or 0.0)
        remove = float(mp.get("t_platform_remove_min", 0.0) or 0.0)
        if key in ("support_to_platform_wash", "platform_wash_to_new"):
            install = 0.0
            remove = 0.0
        return travel + install + remove
    def t_wash1(self): return float(self.P["auto_post"].get("t_wash1_min",30.0))
    def t_wash2(self): return float(self.P["auto_post"].get("t_wash2_min",30.0))
    def t_dry(self):   return float(self.P["auto_post"].get("t_dry_min",30.0))
    def t_uv(self):    return float(self.P["auto_post"].get("t_uv_min",30.0))
    def t_platform_wash(self): return max(0.0, self.platform_wash_time)
    def interarrival(self): D=self.P["demand"]; return random.uniform(D["platform_interarrival_min_min"], D["platform_interarrival_max_min"])
    def fail(self, p): return random.random()<p

    def print_pipeline(self, item:Dict[str,Any]):
        created_at = item.get("created_at", self.env.now)
        name = item.get("platform", "PLAT")
        token = item.get("_platform_token")
        if self.stacker_guard_enabled and self.stacker_guard_limit > 0:
            yield from self._stacker_guard_wait()
        # seize a printer
        reqs = [res.request() for res in self.printers]
        pairs = list(zip(self.printers, reqs))
        anyof = simpy.events.AnyOf(self.env, [rq for _, rq in pairs])
        got = yield anyof  # dict: {event->value}
        pr_idx = next(i for i, (_, rq) in enumerate(pairs) if rq in got)
        # release/cancel all non-selected requests
        for i, (res, rq) in enumerate(pairs):
            if i == pr_idx:
                continue
            if rq.triggered:
                # it was granted simultaneously; release immediately
                res.release(rq)
            else:
                rq.cancel()
        req = pairs[pr_idx][1]
        prof = self.P["print"]["printers"][pr_idx]

        start_service = self.env.now
        wait = start_service - created_at
        if wait <= 0:
            wait = 1.0
        self._log(name, 'new_plat', created_at, created_at + wait)

        # parts per platform: global override (print.max_parts_per_platform) if provided, otherwise printer profile capacity
        g_parts = self.P["print"].get("max_parts_per_platform")
        if g_parts is not None:
            try:
                parts = int(g_parts)
            except Exception:
                parts = int(prof.get("platform_capacity", self.P.get("material",{}).get("max_parts_per_platform", 8)))
        else:
            parts = int(prof.get("platform_capacity", self.P.get("material",{}).get("max_parts_per_platform", 8)))
        mass_kg = parts * self.P["material"]["avg_part_mass_kg"]
        self.kpi.started_parts += parts
        self.kpi.started_platforms += 1

        install_time = self.install_time_min
        if install_time <= 0:
            if self.P["flow_mode"]=="automated":
                install_time = float(self.P["auto_post"].get("t_platform_install_min",30.0))
            else:
                install_time = float(self.P["manual_post"].get("t_platform_install_min",30.0))
        travel = min(install_time, max(0.0, min(5.0, install_time*0.25)))
        work = max(0.0, install_time - travel)

        with self.install_gate.request() as install_rq:
            yield install_rq
            if self.P["flow_mode"]=="automated":
                with self.amr.request() as amr_rq:
                    t0=self.env.now; yield amr_rq
                    if travel > 0:
                        start = self.env.now
                        yield self.env.timeout(travel)
                        self._log(name, 'to_printer', start, self.env.now)
                        self._log_amr(name, 'to_printer', start, self.env.now)
                    else:
                        start = self.env.now
                        end = start + 0.5
                        self._log(name, 'to_printer', start, end)
                        self._log_amr(name, 'to_printer', start, end)
                    if work > 0:
                        t_inst_start = self.env.now
                        yield self.env.timeout(work)
                        self._log(name, 'install', t_inst_start, self.env.now)
                    else:
                        self._log(name, 'install', self.env.now, self.env.now)
            else:
                if travel > 0:
                    while True:
                        if not self._in_shift_move(int(self.env.now)):
                            yield self.env.timeout(self._time_to_next_shift_start_move(int(self.env.now)))
                            continue
                        priority = self._move_priority('to_printer')
                        with self.manual_move_gate.request(priority=priority) as gate_rq:
                            yield gate_rq
                            if not self._in_shift_move(int(self.env.now)):
                                continue
                            with self.manual_movers.request(priority=priority) as mv_rq:
                                yield mv_rq
                                start = self.env.now
                                yield from self._work_with_shift_move(travel)
                                break
                    self._log(name, 'to_printer', start, self.env.now)
                    self._log_manual(name, 'to_printer', start, self.env.now)
                else:
                    start = self.env.now
                    end = start + 0.5
                    self._log(name, 'to_printer', start, end)
                    self._log_manual(name, 'to_printer', start, end)
                if work > 0:
                    while True:
                        with self.manual_pool.request(priority=1) as worker_rq:
                            try:
                                yield worker_rq
                            except simpy.Interrupt as intr:
                                if isinstance(intr.cause, Preempted):
                                    continue
                                raise
                            t_inst_start = self.env.now
                            try:
                                yield from self._work_with_shift(work)
                            except simpy.Interrupt as intr:
                                if isinstance(intr.cause, Preempted):
                                    continue
                                raise
                            self._log(name, 'install', t_inst_start, self.env.now)
                            self._log_manual(name, 'install', t_inst_start, self.env.now)
                            break
                else:
                    self._log(name, 'install', self.env.now, self.env.now)

        t_print = float(self.P["print"]["t_print_per_platform_min"]) if self.P.get("print",{}).get("t_print_per_platform_min") is not None else 300.0
        self.kpi.printer_busy_min += t_print
        t0p = self.env.now
        yield self.env.timeout(t_print)
        self._log(name, 'print', t0p, self.env.now)

        printer_released = False
        def release_printer_once():
            nonlocal printer_released
            if not printer_released:
                self.printers[pr_idx].release(req)
                printer_released = True

        # remove the finished platform from printer (printer blocked but not busy)
        if self.P["flow_mode"]=="automated":
            t_rem = float(self.P["auto_post"].get("t_platform_remove_min",30.0))
            t0r = self.env.now
            yield self.env.timeout(t_rem)
            self._log(name, 'remove', t0r, self.env.now)
        else:
            # Printer removal also proceeds 24/7 in manual mode
            t_rem = float(self.P["manual_post"].get("t_platform_remove_min",30.0))
            t0r = self.env.now
            yield self.env.timeout(t_rem)
            self._log(name, 'remove', t0r, self.env.now)

        if self.fail(prof.get("defect_rate", self.P.get("print",{}).get("defect_rate", 0.0))):
            release_printer_once()
            self.kpi.scrapped_parts += parts
            if token:
                yield self.clean_platforms.put(token)
                token = None
            self._stacker_guard_dec_active()
            return

        self.kpi.resin_used_kg += mass_kg
        self.kpi.resin_cost_krw += mass_kg * self.P["material"]["resin_price_per_kg"]

        self.kpi.n_print_jobs += 1
        if self.P["flow_mode"]=="automated":
            with self.amr.request() as rq: 
                t0=self.env.now; yield rq; 
                if hasattr(self.kpi,'add_wait'): self.kpi.add_wait('amr', self.env.now - t0)
                start = self.env.now
                release_printer_once()
                t=self.amr_travel_min("printer_to_wash"); self.kpi.amr_travel_min+=t; self.kpi.n_amr_moves+=1; yield self.env.timeout(t)
                self._log(name,'to_wash', start, self.env.now)
                self._log_amr(name,'to_wash', start, self.env.now)
            queue_w1 = self._begin_stage(name, 'wash1_queue')
            with self.washers_m1.request() as wash1_rq:
                t0=self.env.now; yield wash1_rq; 
                self._finish_stage(queue_w1)
                if hasattr(self.kpi,'add_wait'): self.kpi.add_wait('wash_m1', self.env.now - t0)
                proc_w1 = self._begin_stage(name, 'wash1')
                t=self.t_wash1(); self.kpi.wash1_busy_min+=t; yield self.env.timeout(t)
                self._finish_stage(proc_w1)
                hold_w1 = self._begin_stage(name, 'wash1_hold')
                if self.fail(self.P["auto_post"]["defect_rate_wash"]):
                    self._finish_stage(hold_w1)
                    self.kpi.scrapped_parts += parts
                    if token:
                        yield self.clean_platforms.put(token)
                        token = None
                    self._stacker_guard_dec_active()
                    return
                with self.amr.request() as amr_rq: 
                    t0=self.env.now; yield amr_rq; 
                    if hasattr(self.kpi,'add_wait'): self.kpi.add_wait('amr', self.env.now - t0)
                    start = self.env.now
                    self._finish_stage(hold_w1)
                    t=self.amr_travel_min("wash1_to_wash2"); self.kpi.amr_travel_min+=t; self.kpi.n_amr_moves+=1; yield self.env.timeout(t);
                    self._log(name,'to_wash2', start, self.env.now)
                    self._log_amr(name,'to_wash2', start, self.env.now)
            queue_w2 = self._begin_stage(name, 'wash2_queue')
            with self.washers_m2.request() as wash2_rq:
                t0=self.env.now; yield wash2_rq; 
                self._finish_stage(queue_w2)
                if hasattr(self.kpi,'add_wait'): self.kpi.add_wait('wash_m2', self.env.now - t0)
                proc_w2 = self._begin_stage(name, 'wash2')
                t=self.t_wash2(); self.kpi.wash2_busy_min+=t; yield self.env.timeout(t); 
                self._finish_stage(proc_w2)
                hold_w2 = self._begin_stage(name, 'wash2_hold')
                # move to dryer only after carrier removes the platform
                with self.amr.request() as amr_rq:
                    t0=self.env.now; yield amr_rq;
                    if hasattr(self.kpi,'add_wait'): self.kpi.add_wait('amr', self.env.now - t0)
                    start = self.env.now
                    self._finish_stage(hold_w2)
                    t=self.amr_travel_min("wash_to_dry"); self.kpi.amr_travel_min+=t; self.kpi.n_amr_moves+=1; yield self.env.timeout(t);
                    self._log(name,'to_dry', start, self.env.now)
                    self._log_amr(name,'to_dry', start, self.env.now)
            queue_dry = self._begin_stage(name, 'dry_queue')
            with self.dryers.request() as dry_rq:
                t0=self.env.now; yield dry_rq; 
                self._finish_stage(queue_dry)
                if hasattr(self.kpi,'add_wait'): self.kpi.add_wait('dryers', self.env.now - t0)
                proc_dry = self._begin_stage(name, 'dry')
                t=self.t_dry(); self.kpi.dry_busy_min+=t; yield self.env.timeout(t); 
                self._finish_stage(proc_dry)
                hold_dry = self._begin_stage(name, 'dry_hold')
                if self.fail(self.P["auto_post"].get("defect_rate_dry",0.0)):
                    self._finish_stage(hold_dry)
                    self.kpi.scrapped_parts += parts
                    if token:
                        yield self.clean_platforms.put(token)
                        token = None
                    self._stacker_guard_dec_active()
                    return
                with self.amr.request() as amr_rq: 
                    t0=self.env.now; yield amr_rq; 
                    if hasattr(self.kpi,'add_wait'): self.kpi.add_wait('amr', self.env.now - t0)
                    start = self.env.now
                    self._finish_stage(hold_dry)
                    t=self.amr_travel_min("dry_to_uv"); self.kpi.amr_travel_min+=t; self.kpi.n_amr_moves+=1; yield self.env.timeout(t);
                    self._log(name,'to_uv', start, self.env.now)
                    self._log_amr(name,'to_uv', start, self.env.now)
            queue_uv = self._begin_stage(name, 'uv_queue')
            with self.uv.request() as uv_rq:
                t0=self.env.now; yield uv_rq; 
                self._finish_stage(queue_uv)
                if hasattr(self.kpi,'add_wait'): self.kpi.add_wait('uv', self.env.now - t0)
                proc_uv = self._begin_stage(name, 'uv')
                t=self.t_uv(); self.kpi.uv_busy_min+=t; yield self.env.timeout(t); 
                self._finish_stage(proc_uv)
                hold_uv = self._begin_stage(name, 'uv_hold')
                if self.fail(self.P["auto_post"].get("defect_rate_uv",0.0)):
                    self._finish_stage(hold_uv)
                    self.kpi.scrapped_parts += parts
                    if token:
                        yield self.clean_platforms.put(token)
                        token = None
                    self._stacker_guard_dec_active()
                    return
                with self.amr.request() as amr_rq: 
                    t0=self.env.now; yield amr_rq; 
                    if hasattr(self.kpi,'add_wait'): self.kpi.add_wait('amr', self.env.now - t0)
                    start = self.env.now
                    self._finish_stage(hold_uv)
                    t=self.amr_travel_min("uv_to_stacker"); self.kpi.amr_travel_min+=t; self.kpi.n_amr_moves+=1; yield self.env.timeout(t);
                    self._log(name,'to_stacker', start, self.env.now)
                    self._log_amr(name,'to_stacker', start, self.env.now)
            t_back = self.amr_travel_min("printer_to_wash")
            if t_back > 0:
                ret_start = self.env.now
                yield self.env.timeout(t_back)
                self._log_amr(name,'amr_return', ret_start, self.env.now)
                self.kpi.amr_travel_min += t_back
                self.kpi.n_amr_moves += 1
            else:
                ret_start = self.env.now
                end = ret_start + 0.5
                self._log_amr(name,'amr_return', ret_start, end)
        else:
            mp=self.P["manual_post"]
            move_w12 = max(0.0, self.manual_route_min("wash1_to_wash2"))
            # Unload and move to washer require manual mover within shift
            while True:
                if not self._in_shift_move(int(self.env.now)):
                    yield self.env.timeout(self._time_to_next_shift_start_move(int(self.env.now)))
                    continue
                priority = self._move_priority('to_wash')
                with self.manual_move_gate.request(priority=priority) as gate_rq:
                    yield gate_rq
                    if not self._in_shift_move(int(self.env.now)):
                        continue
                    with self.manual_movers.request(priority=priority) as wrq:
                        t0 = self.env.now
                        yield wrq
                        if hasattr(self.kpi,'add_wait'):
                            self.kpi.add_wait('manual_movers', self.env.now - t0)
                        unload_time = float(mp.get("human_unload_min", 0.0) or 0.0)
                        if unload_time > 0:
                            t0u = self.env.now
                            yield from self._work_with_shift_move(unload_time)
                            self._log(name, 'unload', t0u, self.env.now)
                            self._log_manual(name, 'unload', t0u, self.env.now)
                        release_printer_once()
                        t0tw = self.env.now
                        yield from self._work_with_shift_move(self.manual_route_min("printer_to_wash"))
                        self._log(name, 'to_wash', t0tw, self.env.now)
                        self._log_manual(name, 'to_wash', t0tw, self.env.now)
                        break
            # Wash1 runs continuously once started (no shift gating)
            queue_w1 = self._begin_stage(name, 'wash1_queue')
            with self.washers_m1.request() as rq:
                t1=self.env.now; yield rq; 
                self._finish_stage(queue_w1)
                if hasattr(self.kpi,'add_wait'): self.kpi.add_wait('wash_m1', self.env.now - t1)
                proc_w1 = self._begin_stage(name, 'wash1')
                t=self.t_wash1(); t0w1=self.env.now
                yield self.env.timeout(t); 
                self._finish_stage(proc_w1)
                self.kpi.wash1_busy_min += t
                hold_w1 = self._begin_stage(name, 'wash1_hold')
                if self.fail(self.P["auto_post"]["defect_rate_wash"]):
                    self._finish_stage(hold_w1)
                    self.kpi.scrapped_parts+=parts
                    if token:
                        yield self.clean_platforms.put(token)
                        token = None
                    self._stacker_guard_dec_active()
                    return
                # Move from Wash1 to Wash2 (manual carriers should visibly travel)
                while True:
                    if not self._in_shift_move(int(self.env.now)):
                        yield self.env.timeout(self._time_to_next_shift_start_move(int(self.env.now)))
                        continue
                    priority = self._move_priority('to_wash2')
                    with self.manual_move_gate.request(priority=priority) as gate_rq:
                        yield gate_rq
                        if not self._in_shift_move(int(self.env.now)):
                            continue
                        with self.manual_movers.request(priority=priority) as wrq:
                            t0 = self.env.now
                            yield wrq
                            if hasattr(self.kpi,'add_wait'):
                                self.kpi.add_wait('manual_movers', self.env.now - t0)
                            start = self.env.now
                            travel = move_w12 if move_w12 > 0 else 0.01
                            yield from self._work_with_shift_move(travel)
                            end_time = self.env.now
                            self._finish_stage(hold_w1)
                            self._log(name, 'to_wash2', start, end_time)
                            self._log_manual(name, 'to_wash2', start, end_time)
                            break
                    break
            queue_w2 = self._begin_stage(name, 'wash2_queue')
            with self.washers_m2.request() as rq:
                t2=self.env.now; yield rq; 
                self._finish_stage(queue_w2)
                if hasattr(self.kpi,'add_wait'): self.kpi.add_wait('wash_m2', self.env.now - t2)
                proc_w2 = self._begin_stage(name, 'wash2')
                t=self.t_wash2(); t0w2=self.env.now
                yield self.env.timeout(t); 
                self._finish_stage(proc_w2)
                self.kpi.wash2_busy_min += t
                hold_w2 = self._begin_stage(name, 'wash2_hold')
                # Move to Dryer
                while True:
                    if not self._in_shift_move(int(self.env.now)):
                        yield self.env.timeout(self._time_to_next_shift_start_move(int(self.env.now)))
                        continue
                    priority = self._move_priority('to_dry')
                    with self.manual_move_gate.request(priority=priority) as gate_rq:
                        yield gate_rq
                        if not self._in_shift_move(int(self.env.now)):
                            continue
                        with self.manual_movers.request(priority=priority) as wrq:
                            t0 = self.env.now
                            yield wrq
                            if hasattr(self.kpi,'add_wait'):
                                self.kpi.add_wait('manual_movers', self.env.now - t0)
                            t0td = self.env.now
                            yield from self._work_with_shift_move(self.manual_route_min("wash_to_dry"))
                            self._finish_stage(hold_w2)
                            self._log(name, 'to_dry', t0td, self.env.now)
                            self._log_manual(name, 'to_dry', t0td, self.env.now)
                            break
                    break
            queue_dry = self._begin_stage(name, 'dry_queue')
            with self.dryers.request() as rq:
                t3=self.env.now; yield rq; 
                self._finish_stage(queue_dry)
                if hasattr(self.kpi,'add_wait'): self.kpi.add_wait('dryers', self.env.now - t3)
                proc_dry = self._begin_stage(name, 'dry')
                t=self.t_dry(); t0dry=self.env.now
                yield self.env.timeout(t); 
                self._finish_stage(proc_dry)
                self.kpi.dry_busy_min += t
                hold_dry = self._begin_stage(name, 'dry_hold')
                if self.fail(self.P["auto_post"].get("defect_rate_dry",0.0)):
                    self._finish_stage(hold_dry)
                    self.kpi.scrapped_parts+=parts
                    if token:
                        yield self.clean_platforms.put(token)
                        token = None
                    self._stacker_guard_dec_active()
                    return
                # travel to stacker
                while True:
                    if not self._in_shift_move(int(self.env.now)):
                        yield self.env.timeout(self._time_to_next_shift_start_move(int(self.env.now)))
                        continue
                    priority = self._move_priority('to_stacker')
                    with self.manual_move_gate.request(priority=priority) as gate_rq:
                        yield gate_rq
                        if not self._in_shift_move(int(self.env.now)):
                            continue
                        with self.manual_movers.request(priority=priority) as wrq:
                            t0 = self.env.now
                            yield wrq
                            if hasattr(self.kpi,'add_wait'):
                                self.kpi.add_wait('manual_movers', self.env.now - t0)
                            t0ts = self.env.now
                            yield from self._work_with_shift_move(self.manual_route_min("uv_to_stacker"))
                            self._finish_stage(hold_dry)
                            self._log(name, 'to_stacker', t0ts, self.env.now)
                            self._log_manual(name, 'to_stacker', t0ts, self.env.now)
                            ret_back = self.manual_route_min("printer_to_wash")
                            if ret_back > 0:
                                ret_start = self.env.now
                                yield from self._work_with_shift_move(ret_back)
                                self._log_manual(name, 'manual_return', ret_start, self.env.now)
                            break
                    break
            # UV if present
            if self.P["auto_post"]["uv_units"]>0:
                queue_uv = self._begin_stage(name, 'uv_queue')
                with self.uv.request() as rq:
                    t4=self.env.now; yield rq; 
                    self._finish_stage(queue_uv)
                    if hasattr(self.kpi,'add_wait'): self.kpi.add_wait('uv', self.env.now - t4)
                    proc_uv = self._begin_stage(name, 'uv')
                    t=self.t_uv(); t0uv=self.env.now
                    yield self.env.timeout(t); 
                    self._finish_stage(proc_uv)
                    self.kpi.uv_busy_min += t
                if self.fail(self.P["auto_post"].get("defect_rate_uv",0.0)):
                    self.kpi.scrapped_parts += parts
                    if token:
                        yield self.clean_platforms.put(token)
                        token = None
                    self._stacker_guard_dec_active()
                    return

        arrived = self.env.now
        payload = {
            "platform": name,
            "n_parts": parts,
            "created_at": created_at,
            "arrived_stacker_at": arrived,
            "_platform_token": item.get("_platform_token"),
            "job_id": item.get("job_id")
        }
        yield self.stacker.put(payload)
        self.kpi.max_stacker_wip = max(self.kpi.max_stacker_wip, len(self.stacker.items))
        entry = self._log(name, 'stacker', arrived, arrived)
        if entry is not None:
            self._stacker_entries[name] = entry
        self._stacker_guard_dec_active()
        release_printer_once()

    def preproc_worker(self):
        while True:
            item = yield self.pre_in.get()
            with self.preproc.request() as rq:
                t0=self.env.now; yield rq
                if hasattr(self.kpi,'add_wait'): self.kpi.add_wait('preproc', self.env.now - t0)
                t = self.t_preproc()
                self.kpi.preproc_busy_min += t
                yield self.env.timeout(t)
            token = yield self.clean_platforms.get()
            token["cycle"] += 1
            name = f"{token['id']}-R{token['cycle']}"
            item["_platform_token"] = token
            item.setdefault("job_id", item.get("platform") or name)
            item["platform"] = name
            yield self.ready.put(item)
            self.kpi.n_preproc_jobs += 1

    def infinite_feeder(self, horizon_min:int):
        i = 0
        parts = self.P["print"].get("max_parts_per_platform", self.P.get("material",{}).get("max_parts_per_platform", 8))
        while True:
            if horizon_min and self.env.now >= horizon_min:
                break
            token = yield self.clean_platforms.get()
            token["cycle"] += 1
            name = f"{token['id']}-R{token['cycle']}"
            i += 1
            item = {
                "job_id": f"JOB-{i}",
                "platform": name,
                "_platform_token": token,
                "n_parts": parts,
                "created_at": self.env.now
            }
            yield self.ready.put(item)

    def printer_dispatcher(self):
        while True:
            item = yield self.ready.get()
            self.env.process(self.print_pipeline(item))

    def arrivals(self):
        i=0
        while True:
            i+=1
            yield self.env.timeout(self.interarrival())
            job_id = f"JOB-{i}"
            parts = self.P["print"].get("max_parts_per_platform", self.P.get("material",{}).get("max_parts_per_platform", 8))
            created_at = self.env.now
            yield self.pre_in.put({"job_id": job_id, "n_parts": parts, "created_at": created_at})

    def _hhmm_to_min(self, s:str)->int:
        hh, mm = str(s).split(':')
        return int(hh)*60 + int(mm)

    def _shift_info(self, group:str):
        defaults = {"start_hhmm":"09:00","end_hhmm":"18:00","work_cycle_days":7,"workdays_per_cycle":7}
        ws = defaults.copy()
        ws.update(self.P.get(group, {}).get("work_shift", {}))
        start = self._hhmm_to_min(ws.get("start_hhmm", "09:00"))
        end = self._hhmm_to_min(ws.get("end_hhmm", "18:00"))
        cycle = int(ws.get("work_cycle_days", ws.get("work_cycle_len", 7)) or 7)
        if cycle <= 0:
            cycle = 7
        raw_pattern = ws.get("workdays_pattern")
        if raw_pattern is not None:
            pattern = [bool(x) for x in raw_pattern]
        else:
            workdays = ws.get("workdays_per_cycle")
            if workdays is None:
                workdays = ws.get("workdays_per_week")
            if workdays is None:
                workdays = ws.get("workdays_per_month")
            if workdays is None:
                workdays = cycle
            workdays = int(workdays or 0)
            workdays = max(0, min(workdays, cycle))
            pattern = [True]*workdays + [False]*(cycle - workdays)
        if len(pattern) < cycle:
            pattern += [False]*(cycle - len(pattern))
        elif len(pattern) > cycle:
            pattern = pattern[:cycle]
        if not any(pattern):
            pattern = [True]*cycle
        return start, end, cycle, pattern

    @staticmethod
    def _shift_state(now:int, cycle:int):
        minutes_per_day = 24*60
        day_min = now % minutes_per_day
        day_idx = (now // minutes_per_day) % cycle
        prev_idx = (day_idx - 1) % cycle
        return day_min, day_idx, prev_idx

    @staticmethod
    def _shift_active(day_min:int, start:int, end:int)->bool:
        if start == end:
            return True
        if start < end:
            return start <= day_min < end
        return day_min >= start or day_min < end

    @staticmethod
    def _time_until_current_shift_end(day_min:int, start:int, end:int)->int:
        minutes_per_day = 24*60
        if start == end:
            return minutes_per_day
        if start < end:
            return max(0, end - day_min)
        if day_min >= start:
            return (minutes_per_day - day_min) + end
        return max(0, end - day_min)

    @staticmethod
    def _time_to_shift_start_today(day_min:int, start:int, end:int, prev_active:bool)->Optional[int]:
        if start == end:
            return 0
        if start < end:
            if day_min < start:
                return start - day_min
            return None
        if prev_active and day_min < end:
            return None
        if day_min < start:
            return start - day_min
        return None

    def _time_to_next_shift_start_generic(self, now:int, start:int, end:int, cycle:int, pattern:list)->int:
        minutes_per_day = 24*60
        day_min, day_idx, prev_idx = self._shift_state(now, cycle)
        if pattern[day_idx]:
            offset = self._time_to_shift_start_today(day_min, start, end, pattern[prev_idx])
            if offset is not None and offset > 0:
                return offset
        total = max(0, minutes_per_day - day_min)
        for d in range(1, cycle+1):
            idx = (day_idx + d) % cycle
            if pattern[idx]:
                return total + start
            total += minutes_per_day
        return total + start

    def _time_until_shift_end_generic(self, now:int, start:int, end:int, cycle:int, pattern:list)->int:
        day_min, day_idx, prev_idx = self._shift_state(now, cycle)
        if pattern[day_idx] and self._shift_active(day_min, start, end):
            return self._time_until_current_shift_end(day_min, start, end)
        if start > end and pattern[prev_idx] and day_min < end:
            return max(0, end - day_min)
        return 0

    def _in_shift_generic(self, now:int, start:int, end:int, cycle:int, pattern:list)->bool:
        day_min, day_idx, prev_idx = self._shift_state(now, cycle)
        if pattern[day_idx] and self._shift_active(day_min, start, end):
            return True
        if start > end and pattern[prev_idx] and day_min < end:
            return True
        return False

    def _in_shift(self, now:int)->bool:
        start, end, cycle, pattern = self._shift_info('manual_ops')
        return self._in_shift_generic(now, start, end, cycle, pattern)

    def _time_to_next_shift_start(self, now:int)->int:
        start, end, cycle, pattern = self._shift_info('manual_ops')
        return self._time_to_next_shift_start_generic(now, start, end, cycle, pattern)

    def _time_until_shift_end(self, now:int)->int:
        start, end, cycle, pattern = self._shift_info('manual_ops')
        return self._time_until_shift_end_generic(now, start, end, cycle, pattern)

    def _work_with_shift(self, t:float):
        remaining = t
        while remaining > 0:
            if not self._in_shift(int(self.env.now)):
                wait = self._time_to_next_shift_start(int(self.env.now))
                yield self.env.timeout(wait)
                continue
            can = min(remaining, self._time_until_shift_end(int(self.env.now)))
            if can <= 0:
                wait = self._time_to_next_shift_start(int(self.env.now))
                yield self.env.timeout(wait)
                continue
            self.kpi.manual_busy_min += can
            yield self.env.timeout(can)
            remaining -= can

    def _in_shift_move(self, now:int)->bool:
        start, end, cycle, pattern = self._shift_info('manual_move')
        return self._in_shift_generic(now, start, end, cycle, pattern)

    def _time_to_next_shift_start_move(self, now:int)->int:
        start, end, cycle, pattern = self._shift_info('manual_move')
        return self._time_to_next_shift_start_generic(now, start, end, cycle, pattern)

    def _time_until_shift_end_move(self, now:int)->int:
        start, end, cycle, pattern = self._shift_info('manual_move')
        return self._time_until_shift_end_generic(now, start, end, cycle, pattern)

    def _work_with_shift_move(self, t:float):
        remaining = t
        while remaining > 0:
            if not self._in_shift_move(int(self.env.now)):
                wait = self._time_to_next_shift_start_move(int(self.env.now))
                yield self.env.timeout(wait)
                continue
            can = min(remaining, self._time_until_shift_end_move(int(self.env.now)))
            if can <= 0:
                wait = self._time_to_next_shift_start_move(int(self.env.now))
                yield self.env.timeout(wait)
                continue
            self.kpi.pre_manual_busy_min += can
            yield self.env.timeout(can)
            remaining -= can

    def _move_priority(self, task:str)->int:
        if not self.prioritize_printer:
            return 1
        if task == 'to_printer':
            return 0
        if task == 'return':
            return 3
        return 1

    def worker_manual_ops(self):
        ops=self.P["manual_ops"]
        while True:
            # respect shift before picking work
            if not self._in_shift(int(self.env.now)):
                yield self.env.timeout(self._time_to_next_shift_start(int(self.env.now)))
                continue
            with self.support_gate.request() as gate:
                yield gate
                item = yield self.stacker.get()
                self._stacker_guard_notify()
                parts = item["n_parts"]; created_at = item.get("created_at", self.env.now)
                plat = item.get("platform", f"PLAT-{random.randint(1,9999)}")
                arrived_stack = item.get("arrived_stacker_at", self.env.now)
                entry = self._stacker_entries.pop(plat, None)
                if entry is not None:
                    entry['t1'] = float(self.env.now)
                elif arrived_stack < self.env.now:
                    self._log(plat, 'stacker', arrived_stack, self.env.now)

                # 1) Platform-level support removal at 'support_plat' (platform arrives and gets prepared)
                t_plat = float(ops.get("support_time_per_platform_min",60))
                t0_arrival = arrived_stack
                if t0_arrival < self.env.now:
                    self._log(plat, 'support_plat', t0_arrival, self.env.now)

                move_plat = float(ops.get("move_platform_to_support_min", 0.0))
                if move_plat > 0 or t_plat > 0:
                    while True:
                        with self.manual_pool.request(priority=0, preempt=True) as escort:
                            try:
                                yield escort
                            except simpy.Interrupt as intr:
                                if isinstance(intr.cause, Preempted):
                                    continue
                                raise
                            if move_plat > 0:
                                t_move = self.env.now
                                try:
                                    yield from self._work_with_shift(move_plat)
                                except simpy.Interrupt as intr:
                                    if isinstance(intr.cause, Preempted):
                                        continue
                                    raise
                                self._log(plat, 'to_support_plat', t_move, self.env.now)
                            t_proc_plat = self.env.now
                            try:
                                yield from self._work_with_shift(t_plat)
                            except simpy.Interrupt as intr:
                                if isinstance(intr.cause, Preempted):
                                    continue
                                raise
                            self._log(plat, 'support_plat', t_proc_plat, self.env.now)
                            break
                else:
                    t_proc_plat = self.env.now
                    yield from self._work_with_shift(t_plat)
                    self._log(plat, 'support_plat', t_proc_plat, self.env.now)

                token = item.get("_platform_token")
                if token:
                    self.env.process(self._platform_clean_cycle(plat, token))
                    item["_platform_token"] = None
                    token = None

            # gate released; continue with part-level stages
            # 2) Split into part tokens at support, and process per stage with shared worker pool
            t_sup = float(ops.get("support_time_per_part_min",5))
            t_fin = float(ops.get("finish_time_per_part_min",10))
            t_pnt = float(ops.get("paint_time_per_part_min",10))

            def do_stage(stage:str, per_time:float, n_parts:int):
                jobs=[]
                # mark arrivals at stage for all parts (so tokens appear and wait)
                stage_arr = item.get("arrived_stacker_at", self.env.now)
                for i in range(1, n_parts+1):
                    pid=f"{plat}-P{i}"
                    # initial presence at station (zero-length; detailed wait logged inside)
                    self._log(pid, stage, stage_arr, stage_arr)
                    jobs.append(self.env.process(self._proc_part_stage(pid, stage, per_time)))
                return simpy.events.AllOf(self.env, jobs)

            # Run stages with barriers between them
            yield do_stage('support_part', t_sup, parts)
            yield do_stage('finish',  t_fin, parts)
            yield do_stage('paint',   t_pnt, parts)
            # All parts completed for this platform
            self.kpi.completed_parts += parts
            self.kpi.finished_platforms += 1
            self.kpi.sum_platform_lead_time_min += (self.env.now - created_at)

    def _proc_part_stage(self, part_id:str, stage:str, per_time:float):
        # Each part requests a manual worker, waits if none available, then works only in shift time
        while True:
            with self.manual_pool.request(priority=1) as rq:
                t_req = self.env.now
                try:
                    yield rq
                except simpy.Interrupt as intr:
                    if isinstance(intr.cause, Preempted):
                        continue
                    raise
                if hasattr(self.kpi,'add_wait'):
                    self.kpi.add_wait('manual_workers', self.env.now - t_req)
                if t_req < self.env.now:
                    self._log(part_id, stage, t_req, self.env.now)
                ops = self.P.get("manual_ops", {})
                move_stage = None
                move_time = 0.0
                if stage == 'finish':
                    move_stage = 'to_finish'
                    move_time = float(ops.get('move_support_to_finish_min', 0.0))
                elif stage == 'paint':
                    move_stage = 'to_paint'
                    move_time = float(ops.get('move_finish_to_paint_min', 0.0))
                if move_stage and move_time > 0:
                    t_move = self.env.now
                    try:
                        yield from self._work_with_shift(move_time)
                    except simpy.Interrupt as intr:
                        if isinstance(intr.cause, Preempted):
                            continue
                        raise
                    self._log(part_id, move_stage, t_move, self.env.now)
                t0 = self.env.now
                try:
                    yield from self._work_with_shift(per_time)
                except simpy.Interrupt as intr:
                    if isinstance(intr.cause, Preempted):
                        continue
                    raise
                self._log(part_id, stage, t0, self.env.now)
                if stage == 'paint':
                    move_store = float(ops.get('move_paint_to_storage_min', 0.0))
                    if move_store > 0:
                        t_move_out = self.env.now
                        try:
                            yield from self._work_with_shift(move_store)
                        except simpy.Interrupt as intr:
                            if isinstance(intr.cause, Preempted):
                                continue
                            raise
                        self._log(part_id, 'to_storage', t_move_out, self.env.now)
                    stay = max(1.0, (self.horizon_minutes - self.env.now) if self.horizon_minutes else 1000.0)
                    self._log(part_id, 'storage', self.env.now, self.env.now + stay)
                break

# ------------------------------
# Runner
# ------------------------------

def duration_to_minutes(duration:str, days:Optional[int])->int:
    d=duration.lower()
    if d=="month": return 30*24*60
    if d=="year":  return 365*24*60
    if d=="days":
        if not days or days<=0: raise ValueError("days must be positive for custom duration")
        return days*24*60
    raise ValueError("duration must be month|year|days")

def run_sim(params:Dict[str,Any], seed:int=42)->Dict[str,Any]:
    random.seed(seed)
    H = duration_to_minutes(params["horizon"]["duration"], params["horizon"]["days"])
    # normalize printers length and apply global defaults
    normalize_printer_count(params)
    apply_global_printer_defaults(params)
    params['_horizon_minutes'] = H
    env=simpy.Environment(); kpi=KPI(); fac=Factory(env, params, kpi)
    if params.get("demand",{}).get("policy","infinite").lower()=="infinite":
        # bypass arrivals/preproc; keep printers fed continuously
        env.process(fac.infinite_feeder(H))
        env.process(fac.printer_dispatcher())
    else:
        env.process(fac.arrivals())
        for _ in range(params.get("preproc",{}).get("servers", 1)):
            env.process(fac.preproc_worker())
        env.process(fac.printer_dispatcher())
    # spawn manual workers per parameter
    for _ in range(params.get("manual_ops",{}).get("workers", 2)):
        env.process(fac.worker_manual_ops())
    env.run(until=H)
    for entry in fac._stacker_entries.values():
        entry['t1'] = float(H)
    fac._stacker_entries.clear()
    cap_w1 = int(params["auto_post"].get("washers_m1",1))
    cap_w2 = int(params["auto_post"].get("washers_m2",1))
    caps={
        "printers":len(params["print"]["printers"]),
        "wash_m1":cap_w1,
        "wash_m2":cap_w2,
        "dryers":params["auto_post"]["dryers"],
        "uv_units":params["auto_post"]["uv_units"],
        "amr": max(1, int(params["auto_post"].get("amr_count", 0) or 0)),
        "platform_washers": params.get("platform_clean",{}).get("washers",1),
        "preproc_servers": params["preproc"]["servers"],
        "manual_workers": params.get("manual_ops",{}).get("workers",2),
        "manual_movers": params.get("manual_move",{}).get("workers",2)
    }
    out=kpi.to_dict(H, caps); out["horizon_minutes"]=H; out["seed"]=seed
    tr = getattr(fac, 'trace', [])
    # Fallback: if trace is empty (e.g., disabled or network hiccup), generate a synthetic trace
    if not tr:
        def amr_t(key):
            ap=params.get("auto_post",{})
            dist=(ap.get("dist_m",{}).get(key,0.0) or 0.0); v=(ap.get("amr_speed_m_per_s",1.0) or 1.0)
            return (dist/v)/60.0 + float(ap.get("amr_load_min",0.0)) + float(ap.get("amr_unload_min",0.0))
        def manual_t(key):
            mp=params.get('manual_post',{})
            overrides={
                'printer_to_wash': mp.get('to_washer_travel_min'),
                'wash1_to_wash2': mp.get('to_wash2_travel_min'),
                'wash_to_dry': mp.get('to_dryer_travel_min'),
                'uv_to_stacker': mp.get('to_stacker_travel_min'),
                'support_to_platform_wash': mp.get('to_platform_wash_travel_min'),
                'platform_wash_to_new': mp.get('to_newplatform_travel_min')
            }
            override=overrides.get(key)
            if override is not None and float(override) > 0:
                return float(override)
            mm=params.get('manual_move',{})
            dist_map=(mm.get('dist_m') or {})
            dist=dist_map.get(key)
            if dist is None:
                fallback={'wash1_to_wash2':'wash_to_dry','uv_to_stacker':'uv_to_stacker','wash_to_dry':'wash_to_dry','printer_to_wash':'printer_to_wash'}
                alt=fallback.get(key)
                dist=params.get('auto_post',{}).get('dist_m',{}).get(alt or key)
            dist=float(dist or 0.0)
            if dist <= 0:
                return 0.0
            speed=float(mm.get('speed_m_per_s',2.0) or 2.0)
            speed=max(speed, 1e-6)
            return (dist / speed) / 60.0
        flow_mode = params.get('flow_mode','automated')
        install = float(params.get('auto_post',{}).get('t_platform_install_min',30.0)) if flow_mode=='automated' else float(params.get('manual_post',{}).get('t_platform_install_min',30.0))
        remove  = float(params.get('auto_post',{}).get('t_platform_remove_min',30.0)) if flow_mode=='automated' else float(params.get('manual_post',{}).get('t_platform_remove_min',30.0))
        t_print = float(params.get('print',{}).get('t_print_per_platform_min',300.0))
        t_w1    = float(params.get('auto_post',{}).get('t_wash1_min',30.0))
        t_w2    = float(params.get('auto_post',{}).get('t_wash2_min',30.0))
        t_dry   = float(params.get('auto_post',{}).get('t_dry_min',30.0))
        t_uv    = float(params.get('auto_post',{}).get('t_uv_min',30.0))
        to_wash = amr_t('printer_to_wash') if flow_mode=='automated' else manual_t('printer_to_wash')
        to_wash2= amr_t('wash1_to_wash2')  if flow_mode=='automated' else manual_t('wash1_to_wash2')
        to_dry  = amr_t('wash_to_dry')     if flow_mode=='automated' else manual_t('wash_to_dry')
        to_uv   = amr_t('dry_to_uv')       if flow_mode=='automated' else 0.0
        to_stack= amr_t('uv_to_stacker')   if flow_mode=='automated' else manual_t('uv_to_stacker')
        to_plat_wash = amr_t('support_to_platform_wash') if flow_mode=='automated' else manual_t('support_to_platform_wash')
        t_plat_wash = float(params.get('platform_clean',{}).get('wash_time_min',60) or 0)
        to_new_plat = amr_t('platform_wash_to_new') if flow_mode=='automated' else manual_t('platform_wash_to_new')
        cycle = install + t_print + remove + to_wash + t_w1 + to_wash2 + t_w2 + to_dry + t_dry + to_uv + t_uv + to_stack
        cycle += to_plat_wash + t_plat_wash + to_new_plat
        n_pr = max(1, len(params.get('print',{}).get('printers', [])))
        viz_max = int(params.get('viz_max_platforms', 30) or 30)
        # staggered starts per printer
        tr = []
        start_gap = max(1.0, (t_print+install+remove)/n_pr)
        pid = 0
        t0 = 0.0
        while pid < viz_max and t0 < H:
            name = f"PLAT-{pid+1}"
            # sequential segments
            s=t0; tr.append({"id":name,"stage":"install","t0":s,"t1":s+install}); s+=install
            tr.append({"id":name,"stage":"print","t0":s,"t1":s+t_print}); s+=t_print
            tr.append({"id":name,"stage":"remove","t0":s,"t1":s+remove}); s+=remove
            tr.append({"id":name,"stage":"to_wash","t0":s,"t1":s+to_wash}); s+=to_wash
            tr.append({"id":name,"stage":"wash1","t0":s,"t1":s+t_w1}); s+=t_w1
            tr.append({"id":name,"stage":"to_wash2","t0":s,"t1":s+to_wash2}); s+=to_wash2
            tr.append({"id":name,"stage":"wash2","t0":s,"t1":s+t_w2}); s+=t_w2
            tr.append({"id":name,"stage":"to_dry","t0":s,"t1":s+to_dry}); s+=to_dry
            tr.append({"id":name,"stage":"dry","t0":s,"t1":s+t_dry}); s+=t_dry
            tr.append({"id":name,"stage":"to_uv","t0":s,"t1":s+to_uv}); s+=to_uv
            tr.append({"id":name,"stage":"uv","t0":s,"t1":s+t_uv}); s+=t_uv
            tr.append({"id":name,"stage":"to_stacker","t0":s,"t1":s+to_stack});
            s+=to_stack
            tr.append({"id":name,"stage":"to_platform_wash","t0":s,"t1":s+to_plat_wash}); s+=to_plat_wash
            tr.append({"id":name,"stage":"platform_wash","t0":s,"t1":s+t_plat_wash}); s+=t_plat_wash
            tr.append({"id":name,"stage":"to_new_platform","t0":s,"t1":s+to_new_plat});
            pid += 1; t0 += start_gap
        # clamp within horizon for UI safety
        tr = [{**e, "t1": min(H, e["t1"])} for e in tr if e["t0"] < H]
    out["trace"] = tr
    return out

# ------------------------------
# Costing
# ------------------------------

def compute_costs(params:Dict[str,Any], kpi:KPI, horizon_min:int, util:Dict[str,Any])->Dict[str,Any]:
    C = params.get("cost", {})
    wage = float(C.get("wage_per_hour_krw", 15000))
    eq = C.get("equipment", {})
    dep_years = float(C.get("depreciation_years", 5))
    overhead_month = float(C.get("overhead_krw_per_month", 20_000_000))
    flow_mode = params.get("flow_mode", "automated")

    # derive manual busy time from utilization if not present
    if getattr(kpi, 'manual_busy_min', 0) and kpi.manual_busy_min>0:
        manual_busy_min = kpi.manual_busy_min
    else:
        u = float(util.get('manual_workers', 0.0))
        cap = int(params.get('manual_ops',{}).get('workers', 1)) or 1
        manual_busy_min = u * horizon_min * cap
    if flow_mode == "manual":
        if getattr(kpi, 'pre_manual_busy_min', 0) and kpi.pre_manual_busy_min>0:
            mover_busy_min = kpi.pre_manual_busy_min
        else:
            u_move = float(util.get('manual_movers', 0.0))
            mover_cap = int(params.get('manual_move',{}).get('workers', 1)) or 1
            mover_busy_min = u_move * horizon_min * mover_cap
    else:
        mover_busy_min = 0.0
    labor = ((manual_busy_min + mover_busy_min)/60.0) * wage
    material = kpi.resin_cost_krw
    # equipment counts
    n_pr = len(params.get("print",{}).get("printers", []))
    n_ws = int(params.get("auto_post",{}).get("washers", 0))
    n_dr = int(params.get("auto_post",{}).get("dryers", 0))
    n_uv = int(params.get("auto_post",{}).get("uv_units", 0))
    n_amr= int(params.get("auto_post",{}).get("amr_count", 0)) if flow_mode != "manual" else 0
    n_pf = int(params.get("platform_clean",{}).get("washers", 0))
    n_pre= int(params.get("preproc",{}).get("servers", 0))
    capex = (n_pr*eq.get("printer",0) + n_ws*eq.get("washer",0) + n_dr*eq.get("dryer",0) +
             n_uv*eq.get("uv",0) + n_amr*eq.get("amr",0) + n_pf*eq.get("platform_washer",0) + n_pre*eq.get("preproc_server",0))
    horizon_years = horizon_min / (60*24*365)
    depreciation = (capex/dep_years) * horizon_years
    horizon_months = horizon_min / (60*24*30)
    overhead = overhead_month * horizon_months
    total = labor + material + depreciation + overhead
    unit = (total / kpi.completed_parts) if kpi.completed_parts>0 else None
    return {
        "labor_krw": int(labor),
        "material_krw": int(material),
        "depreciation_krw": int(depreciation),
        "overhead_krw": int(overhead),
        "total_krw": int(total),
        "unit_cost_krw": None if unit is None else int(unit)
    }

# ------------------------------
# Web App
# ------------------------------

app=FastAPI(title="3D Printing DES (enhanced)", version="1.3.0")

class SimRequest(BaseModel):
    params: Optional[Dict[str,Any]] = None
    seed: int = 42

@app.get("/", response_class=HTMLResponse)
def index(req: Request):
    html = """
<!doctype html>
<html data-theme="dark"><head><meta charset="utf-8"/>
<title>LINC-DES v3.1</title>
<script>
(function(){
  try{
    const saved = localStorage.getItem('linc-des-theme');
    if(saved){
      document.documentElement.setAttribute('data-theme', saved);
    }
  }catch(e){}
})();
</script>
<style>
:root{
  /* K-pop Demon Hunters inspired neon/cyberpunk palette */
  color-scheme: dark;
  --bg:#0b0f1f; --bg2:#0e1228;
  --bg-gradient: radial-gradient(1200px 800px at 20% 0%, #11163a 0%, #0b0f1f 60%), linear-gradient(180deg,#0e132b,#0b0f1f);
  --panel:rgba(20,26,58,0.6); --line:#1a2146;
  --text:#eaf2ff; --muted:#98a5c7;
  --primary:#8a2be2; --primary2:#00e5ff; --hot:#ff2bd6; --acid:#00ffa3;
  --control-bg:#0c1230; --control-border:#1d2750; --control-text:var(--text);
  --table-header:#12183a;
  --notice-bg:#1e293b; --notice-text:var(--text);
  --bar-bg:#141c3f; --shadow:0 10px 30px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.05);
  --canvas-surface:#1e293b; --canvas-surface-alt:#0f1424; --canvas-night:#1b1f2b; --canvas-text:#eaf2ff;
  --canvas-station:rgba(245,247,255,0.22); --canvas-station-night:rgba(236,240,255,0.18); --canvas-station-border:transparent;
  --layout-text:#eaf2ff;
  --layout-border:rgba(234,242,255,0.25);
  --layout-arrow:var(--primary2);
  --layout-step1:#18245a;
  --layout-step2:#1f2f6f;
  --layout-step3:#10384a;
  --layout-step4:#1a3f72;
  --layout-step5:#1b4a54;
  --layout-step6:#331b50;
  --layout-step7:#1f2e63;
  --cycle-pre:#264653;
  --cycle-print:#2a9d8f;
  --cycle-post:#e9c46a;
  --cycle-ops:#e76f51;
  --cycle-text:#eaf2ff;
  --toolbar-bg:rgba(12,18,48,0.85); --toolbar-border:#334155;
}
:root[data-theme="light"]{
  color-scheme: light;
  --bg:#f5f7ff; --bg2:#eef2ff;
  --bg-gradient: radial-gradient(1200px 800px at 20% 0%, #ffffff 0%, #eef2ff 65%), linear-gradient(180deg,#f8fbff,#e9efff);
  --panel:rgba(255,255,255,0.86); --line:#d2d9f5;
  --text:#1a2146; --muted:#5c6b91;
  --primary:#7b2cd6; --primary2:#2d8cff; --hot:#ff4fad; --acid:#00a87f;
  --control-bg:#ffffff; --control-border:#c3cae7; --control-text:#132043;
  --table-header:#e7ecff;
  --notice-bg:#f1f4ff; --notice-text:#1f2c5c;
  --bar-bg:#dbe3ff; --shadow:0 12px 26px rgba(110,130,180,0.18), inset 0 1px 0 rgba(255,255,255,0.75);
  --canvas-surface:#dde5ff; --canvas-surface-alt:#eef2ff; --canvas-night:#d1d4dc; --canvas-text:#20315f;
  --canvas-station:rgba(255,255,255,0.6); --canvas-station-night:rgba(245,246,252,0.5); --canvas-station-border:transparent;
  --layout-text:#132043;
  --layout-border:rgba(19,32,67,0.25);
  --layout-arrow:#0fb0ff;
  --layout-step1:#d8def9;
  --layout-step2:#c6d6ff;
  --layout-step3:#c0e8f0;
  --layout-step4:#cddbf9;
  --layout-step5:#c4eadb;
  --layout-step6:#d9c2f4;
  --layout-step7:#c8d6fd;
  --cycle-pre:#3a4f8a;
  --cycle-print:#3ea89d;
  --cycle-post:#f5cf82;
  --cycle-ops:#f18466;
  --cycle-text:#132043;
  --toolbar-bg:rgba(255,255,255,0.9); --toolbar-border:#c3cae7;
}
body{font-family:-apple-system,Inter,Segoe UI,Roboto,Helvetica,Arial,sans-serif;background:var(--bg-gradient);background-color:var(--bg);color:var(--text);margin:0;padding:24px;transition:background 0.4s ease,color 0.2s ease}
h1{margin:0 0 2px;background:linear-gradient(90deg,var(--primary),var(--primary2));-webkit-background-clip:text;background-clip:text;color:transparent}
.subtitle{color:var(--muted);margin:0 0 16px}
.wrap{max-width:1200px;margin:0 auto}
.brand{display:flex;align-items:center;gap:12px;margin-bottom:16px;justify-content:space-between}
.brand-actions{display:flex;align-items:center;gap:12px;flex-wrap:wrap;justify-content:flex-end}
.brand img{height:28px;filter:drop-shadow(0 0 8px rgba(0,229,255,.4))}
.card{background:var(--panel);backdrop-filter:blur(10px);border:1px solid var(--line);border-radius:14px;padding:16px;margin:12px 0;box-shadow:var(--shadow)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.grid-4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
label{font-size:12px;color:var(--muted);margin-top:6px;display:block}
input,select,textarea{width:100%;padding:10px 12px;background:var(--control-bg);border:1px solid var(--control-border);color:var(--control-text);border-radius:10px;box-sizing:border-box;transition:background 0.3s ease,color 0.2s ease,border 0.3s ease}
input:focus,select:focus,textarea:focus{outline:1px solid var(--primary2);box-shadow:0 0 0 3px rgba(0,229,255,0.15)}
small{color:var(--muted)}
button{background:linear-gradient(90deg,var(--hot),var(--primary2));color:#08121f;border:none;padding:10px 16px;border-radius:10px;cursor:pointer;font-weight:700;box-shadow:0 0 12px rgba(255,43,214,.3), 0 0 18px rgba(0,229,255,.25)}
button.mode-toggle{background:transparent;color:var(--text);border:1px solid var(--line);box-shadow:none;padding:8px 12px;font-weight:600;transition:background 0.2s ease,color 0.2s ease}
button.mode-toggle:hover{background:rgba(255,255,255,0.08)}
:root[data-theme="light"] button.mode-toggle:hover{background:rgba(0,0,0,0.08)}
table{border-collapse:collapse;width:100%} th,td{border:1px solid var(--line);padding:8px} th{background:var(--table-header);text-align:left;color:var(--text)}
.bar{height:10px;border-radius:6px;background:var(--bar-bg);position:relative;box-shadow:inset 0 0 8px rgba(0,229,255,.2)}
.bar>span{position:absolute;left:0;top:0;height:100%;border-radius:6px;box-shadow:0 0 8px rgba(0,229,255,.45)}
.kpi{display:grid;grid-template-columns:1fr 1fr;gap:16px}
</style>
</head>
<body>
<div class="wrap">
<div class="brand">
  <h1>LINC-DES v3.1</h1>
  <div class="brand-actions">
    <button id="themeToggle" type="button" class="mode-toggle">라이트 모드</button>
    <img src="https://lincsolution.com/wp-content/uploads/2025/04/logo_gnb_color.svg" alt="LINC" onerror="this.style.display='none'"/>
  </div>
</div>
<div class="subtitle">3D 프린팅 공정 시뮬레이션. 파라미터를 조정하고 자동화/비자동화를 비교하세요.</div>

<div class="card">
  <div class="grid">
    <div>
      <label>기간 (Horizon)</label>
      <select id="duration">
        <option value="month">1개월 (30일)</option>
        <option value="year">1년 (365일)</option>
        <option value="days">커스텀(일)</option>
      </select>
    </div>
    <div>
      <label>커스텀 일수 (duration=days일 때)</label>
      <input id="days" type="number" placeholder="예: 45"/>
    </div>
  </div>
  <div class="grid">
    <div>
      <label>비교 모드</label>
      <input type="text" value="후처리 자동화 vs 비자동화 결과를 동시에 표로 비교합니다" disabled />
    </div>
    <div>
      <label>Random Seed</label>
      <input id="seed" type="number" value="42"/>
      <small>난수 초기값입니다. 같은 Seed로 돌리면 같은 랜덤 시나리오가 재현되어 결과 비교가 쉬워요.</small>
    </div>
  </div>
  
</div>

<div class="card">
  <h3>① 수요/기간 설정</h3>
  <div class="grid">
    <div>
      <label>빌드플랫폼 도착 간격 최소 (분)</label>
      <input id="inter_min" type="number" step="0.1" value="30"/>
    </div>
    <div>
      <label>빌드플랫폼 도착 간격 최대 (분)</label>
      <input id="inter_max" type="number" step="0.1" value="30"/>
    </div>
    <div>
      <label>초기 공급 플랫폼 수 (무한 수요)</label>
      <input id="initial_platforms" type="number" value="20"/>
      <small>무한 수요 모드에서 미리 제공할 플랫폼 수입니다. 0이면 수요에 따라 동적 생성합니다.</small>
    </div>
  </div>
  <small>설명: 두 값 사이에서 '다음 도착까지의 시간(분)'을 균등분포로 매회 샘플링합니다. 간격이 짧을수록 수요가 큽니다.</small>
</div>

<div class="card">
  <h3>② 준비/소재 조건</h3>
  <div class="grid">
    <div>
      <label>파트 1개 무게 (kg)</label>
      <input id="avg_part_mass_kg" type="number" step="0.001" value="0.12"/>
    </div>
    <div>
      <label>레진 가격 (KRW/kg)</label>
      <input id="resin_price_per_kg" type="text" value="50,000" class="krw"/>
    </div>
  </div>
</div>

<div class="card">
  <h3>③ 출력(프린팅) 조건</h3>
  <div class="grid">
    <div><label>프린팅 시간 (플랫폼당, 분)</label>
      <input id="t_print" type="number" step="1" value="1400"/>
      <small>직접 계산한 프린팅 시간을 입력하세요. (빌드플랫폼 1회)</small>
    </div>
    <div><label>플랫폼 설치 시간 (분)</label>
      <input id="install_time" type="number" step="1" value="40"/>
      <small>AMR 또는 작업자가 신규 플랫폼을 프린터에 세팅하는 데 걸리는 시간.</small>
    </div>
    <div>
      <label>설치 병렬 허용</label>
      <label><input id="install_parallel" type="checkbox" checked/> AMR/작업자가 동시에 여러 플랫폼을 세팅</label>
    </div>
    <div>
      <label>빌드플랫폼 최대 탑재 파트 수 (ea)</label>
      <input id="max_parts_per_platform" type="number" value="90"/>
      <small>프린터별 플랫폼 용량이 다르면 프린터 프로필의 platform_capacity로 개별 설정 가능합니다.</small>
    </div>
    <div>
      <label>프린터 불량률 (%)</label>
      <input id="print_defect_rate" type="number" step="0.1" value="8.0"/>
      <small>모든 프린터에 적용되는 평균 불량률입니다.</small>
    </div>
    <div>
      <label>프린터 대수</label>
      <input id="printer_count" type="number" value="6"/>
      <small>프린터 목록은 첫 프로필을 템플릿으로 복제합니다. (상세는 Raw JSON에서 편집)</small>
    </div>
  </div>
  <small>빌드플랫폼: 3D 프린터에 들어가는 출력용 판형으로, 여러 파트를 배치해 한 번에 출력합니다.</small>
</div>

<div class="card">
  <h3>④ 전처리(데이터/배치/서포트/전송)</h3>
  <div class="grid">
    <div><label>전처리 서버 수 (ea)</label><input id="preproc_servers" type="number" value="2"/></div>
    <div><label>힐링 시간 (분/빌드플랫폼)</label><input id="heal_per_plat" type="number" step="1" value="120"/></div>
    <div><label>배치 시간 (분/빌드플랫폼)</label><input id="placement_per_plat" type="number" step="1" value="10"/></div>
    <div><label>서포트 생성 (분/빌드플랫폼)</label><input id="support_per_plat" type="number" step="1" value="30"/></div>
  </div>
  <small>서버 1대당 동시에 하나의 빌드플랫폼 데이터를 처리합니다. 전처리 시간 = 힐링(파트수×분) + 배치(분/빌드플랫폼) + 서포트 생성(분/빌드플랫폼).</small>
</div>

<div class="card">
  <h3>⑥ 후처리 공통 설정</h3>
  <div class="grid-4">
    <div><label>세척1 시간 (분)</label><input id="cw1" type="number" step="1" value="20"/></div>
    <div><label>세척2 시간 (분)</label><input id="cw2" type="number" step="1" value="20"/></div>
    <div><label>건조 시간 (분)</label><input id="cdry" type="number" step="1" value="20"/></div>
    <div><label>UV 시간 (분)</label><input id="cuv" type="number" step="1" value="20"/></div>
    <div><label>불량률 (세척/건조/UV)</label>
      <input id="c_def_w" type="number" step="0.0001" value="0.005"/>
      <input id="c_def_d" type="number" step="0.0001" value="0.001"/>
      <input id="c_def_u" type="number" step="0.0001" value="0.0"/>
    </div>
  </div>
  <small>자동/비자동 모두 동일 장비/가동시간/불량률을 사용합니다. 차이는 수거/이동 방식(AMR vs 인력)만 적용됩니다.</small>
</div>

<div class="card">
  <h3>⑦ 후처리 조건 (자동화 모드)</h3>
  <div class="grid-4">
    <div><label>AMR 대수 (ea)</label><input id="amr_count" type="number" value="2"/></div>
    <div><label>AMR 속도 (m/s)</label><input id="amr_speed" type="number" step="0.1" value="0.1"/></div>
    <div><label>AMR 로드/언로드 (분)</label><input id="amr_load" type="number" step="1" value="20"/><input id="amr_unload" type="number" step="1" value="20"/></div>
    <div><label>세척1 기수 (ea)</label><input id="washers_m1" type="number" value="1"/></div>
    <div><label>세척2 기수 (ea)</label><input id="washers_m2" type="number" value="1"/></div>
    <div><label>건조기 수 (ea)</label><input id="dryers" type="number" value="1"/></div>
    <div><label>UV 수 (ea)</label><input id="uv_units" type="number" value="1"/></div>
    <div><label>거리 (m)</label>
      <input id="d_pw" type="number" value="10"/>
      <input id="d_wd" type="number" value="10"/>
      <input id="d_du" type="number" value="10"/>
      <input id="d_us" type="number" value="10"/>
    </div>

    <div><label>플랫폼 장착/제거 시간 (분)</label>
      <input id="auto_install" type="number" step="1" value="20"/>
      <input id="auto_remove" type="number" step="1" value="20"/>
    </div>
    <div><label>플랫폼 세척기 (ea)</label><input id="platform_washers" type="number" value="1"/></div>
    <div><label>플랫폼 세척 시간 (분)</label><input id="platform_wash_time" type="number" step="1" value="60"/></div>
    <div><label>초기 플랫폼 수 (세척 루프)</label><input id="platform_initial_count" type="number" value="20"/></div>
  </div>
</div>

<div class="card">
  <h3>⑧ 후처리 조건 (비자동화 모드)</h3>
  <div class="grid">
    <div><label>빌드플랫폼 수거 (분/빌드플랫폼)</label><input id="human_unload_min" type="number" step="0.1" value="20"/></div>
    <div><label>세척기로 이동 (분)</label><input id="to_washer" type="number" step="0.1" value="10"/></div>
    <div><label>세척1→2 이동 (분)</label><input id="to_wash2" type="number" step="0.1" value="10"/></div>
    <div><label>건조기로 이동 (분)</label><input id="to_dryer" type="number" step="0.1" value="10"/></div>
    <div><label>스택커로 이동 (분)</label><input id="to_stacker" type="number" step="0.1" value="10"/></div>
    <div><label>플랫폼 장착/제거 시간 (분)</label>
      <input id="man_install" type="number" step="1" value="20"/>
      <input id="man_remove" type="number" step="1" value="20"/>
    </div>
  </div>
</div>

<div class="card">
  <h3>⑨ 수동 이송/장비운전 인력(비자동화)</h3>
  <div class="grid-3">
    <div><label>이송/운전 인력 수 (ea)</label><input id="mv_workers" type="number" value="2"/></div>
    <div><label>이송 속도 (m/s)</label><input id="mv_speed" type="number" step="0.1" value="0.1"/></div>
    <div><label>근무 시작(HH:MM)</label><input id="mv_ws_start" type="text" value="09:00"/></div>
    <div><label>근무 종료(HH:MM)</label><input id="mv_ws_end" type="text" value="18:00"/></div>
    <div><label>주기당 근무일수 (기본 5/7)</label><input id="mv_ws_days" type="number" value="5"/></div>
    <div><label>프린터 우선 이송</label><input id="mv_prioritize" type="checkbox" checked/></div>
  </div>
</div>

<div class="card">
  <h3>⑩ 수작업 공정(스택커 이후)</h3>
  <div class="grid-3">
    <div><label>서포트 제거 (분/빌드플랫폼)</label><input id="sup_plat" type="number" step="1" value="20"/></div>
    <div><label>서포트 제거 (분/파트)</label><input id="sup_part" type="number" step="1" value="2"/></div>
    <div><label>사상 (분/파트)</label><input id="fin" type="number" step="1" value="2"/></div>
    <div><label>페인팅 (분/파트)</label><input id="paint" type="number" step="1" value="2"/></div>
  </div>
  <div class="grid-4">
    <div><label>스택커→플랫폼 서포트 이동 (분/빌드플랫폼)</label><input id="move_plat_support" type="number" step="0.1" value="1"/></div>
    <div><label>플랫폼 서포트→사상 이동 (분/파트)</label><input id="move_support_finish" type="number" step="0.1" value="1"/></div>
    <div><label>사상→페인팅 이동 (분/파트)</label><input id="move_finish_paint" type="number" step="0.1" value="1"/></div>
    <div><label>페인팅→보관 이동 (분/파트)</label><input id="move_paint_storage" type="number" step="0.1" value="1"/></div>
  </div>
  <div class="grid-3">
    <div><label>작업 시간 (시작~끝, 주당 근무일) / 작업자 수</label>
      <input id="ws_start" type="text" value="09:00"/>
      <input id="ws_end" type="text" value="18:00"/>
      <input id="ws_days" type="number" value="5"/>
      <input id="workers" type="number" value="12"/>
    </div>
  </div>
</div>

<div class="card">
  <h3>⑪ 스택커 병목 제어</h3>
  <div class="grid">
    <div>
      <label><input id="stacker_guard_enabled" type="checkbox"/> 스택커 적체 시 출력 중단</label>
      <small>스택커 대기 플랫폼이 기준 이상이면 신규 출력 착수를 일시 중단합니다.</small>
    </div>
    <div>
      <label>허용 스택커 대기 플랫폼 수 (ea)</label>
      <input id="stacker_guard_limit" type="number" value="0"/>
      <small>0이면 제한 없이 계속 출력합니다.</small>
    </div>
  </div>
</div>

<div class="card">
  <h3>⑫ 비용/단가 설정</h3>
  <div class="grid-3">
    <div>
      <label>작업자 임율 (KRW/시간)</label>
      <input id="cost_wage" type="text" value="23,000" class="krw"/>
    </div>
    <div>
      <label>감가상각 (년)</label>
      <input id="dep_years" type="number" step="0.5" value="5"/>
    </div>
    <div>
      <label>인프라/고정비 (KRW/월)</label>
      <input id="overhead" type="text" value="20,000,000" class="krw"/>
    </div>
    <div>
      <label>장비 단가 (프린터, KRW)</label>
      <input id="c_printer" type="text" value="400,000,000" class="krw"/>
    </div>
    <div>
      <label>장비 단가 (세척, KRW)</label>
      <input id="c_washer" type="text" value="50,000,000" class="krw"/>
    </div>
    <div>
      <label>장비 단가 (건조, KRW)</label>
      <input id="c_dryer" type="text" value="10,000,000" class="krw"/>
    </div>
    <div>
      <label>장비 단가 (UV, KRW)</label>
      <input id="c_uv" type="text" value="20,000,000" class="krw"/>
    </div>
    <div>
      <label>장비 단가 (AMR, KRW)</label>
      <input id="c_amr" type="text" value="200,000,000" class="krw"/>
    </div>
    <div>
      <label>장비 단가 (전처리 서버, KRW)</label>
      <input id="c_pre" type="text" value="5,000,000" class="krw"/>
    </div>
    <div>
      <label>장비 단가 (플랫폼 세척기, KRW)</label>
      <input id="c_platform_wash" type="text" value="30,000,000" class="krw"/>
    </div>
  </div>
  <small>단가는 예시값입니다. 각 설비 대수×단가를 감가상각하여 기간에 해당하는 비용으로 환산합니다.</small>
</div>

<div class="card">
  <h3>Raw JSON 파라미터 (고급 사용자)</h3>
  <textarea id="raw" rows="12">{}</textarea>
  <div style="margin-top:8px"><button onclick="run()">시뮬레이션 실행</button></div>
</div>

<div id="out"></div>

<script>
const THEME_STORAGE_KEY = 'linc-des-theme';
function applyTheme(theme, skipStore){
  const next = theme === 'light' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  if(document.body){
    document.body.setAttribute('data-theme', next);
  }
  if(!skipStore){
    try{ localStorage.setItem(THEME_STORAGE_KEY, next); }catch(e){}
  }
  const toggle = document.getElementById('themeToggle');
  if(toggle){
    toggle.textContent = next === 'light' ? '다크 모드' : '라이트 모드';
  }
  redrawAnimations();
}
function initThemeToggle(){
  let saved = null;
  try{ saved = localStorage.getItem(THEME_STORAGE_KEY); }catch(e){}
  applyTheme(saved === 'light' ? 'light' : 'dark', true);
  const toggle = document.getElementById('themeToggle');
  if(toggle){
    toggle.addEventListener('click', ()=>{
      const current = (document.body && document.body.getAttribute('data-theme')) || document.documentElement.getAttribute('data-theme') || 'dark';
      const next = current === 'light' ? 'dark' : 'light';
      applyTheme(next, false);
    });
  }
}
function currentThemeVars(){
  const root = getComputedStyle(document.documentElement);
  const read = (name, fallback)=>{
    const val = root.getPropertyValue(name);
    return val && val.trim() ? val.trim() : fallback;
  };
  return {
    text: read('--text', '#1a2146'),
    muted: read('--muted', '#5c6b91'),
    accent: read('--primary2', '#2d8cff'),
    noticeBg: read('--notice-bg', '#f1f4ff'),
    noticeText: read('--notice-text', '#1f2c5c'),
    line: read('--line', '#d2d9f5'),
    controlBg: read('--control-bg', '#ffffff'),
    controlText: read('--control-text', '#132043'),
    controlBorder: read('--control-border', '#c3cae7'),
    barBg: read('--bar-bg', '#dbe3ff'),
    canvasSurface: read('--canvas-surface', '#dde5ff'),
    canvasSurfaceAlt: read('--canvas-surface-alt', '#f6f8ff'),
    nightBg: read('--canvas-night', '#d1d4dc'),
    canvasText: read('--canvas-text', '#20315f'),
    canvasStation: read('--canvas-station', '#c3d4ff'),
    canvasStationNight: read('--canvas-station-night', '#aebff7'),
    canvasStationBorder: read('--canvas-station-border', 'rgba(28,48,102,0.25)'),
    toolbarBg: read('--toolbar-bg', 'rgba(255,255,255,0.9)'),
    toolbarBorder: read('--toolbar-border', '#c3cae7')
  };
}
function redrawAnimations(){
  const cache = window.__renderFactoryCache;
  if(!cache) return;
  const entries = Object.entries(cache);
  if(!entries.length) return;
  entries.forEach(([containerId, payload])=>{
    if(!payload || !payload.trace) return;
    renderFactory2D(containerId, payload.trace, payload.opts || {});
  });
}
initThemeToggle();

function barHtml(p){
  const pct = Math.max(0, Math.min(100, p));
  let color = pct<60? '#00ffa3' : (pct<85? '#00e5ff':'#ff2bd6');
  return `<div class="bar"><span style="width:${pct}%;background:${color}"></span></div>`;
}
function svgRect(x,y,w,h,fill,stroke,label,textColor) {
  return `<g><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" ry="12" fill="${fill}" stroke="${stroke}"></rect>
  <text x="${x+12}" y="${y+22}" font-size="12" fill="${textColor}" font-weight="600">${label}</text></g>`;
}

function layoutSvgHtml(P, util){
  const Np = (P.print && P.print.printer_count) || (P.print && P.print.printers ? P.print.printers.length : 2);
  const W = 360;
  const x0 = 20, y0 = 20, bw = 220, bh = 36, gap = 24;
  let y = y0;
  const blocks = [];
  const arrowColor = 'var(--layout-arrow)';
  const markerDef = `<defs><marker id="layoutArrow" markerWidth="10" markerHeight="10" refX="10" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="${arrowColor}" /></marker></defs>`;
  const arrowDown = (x,y)=>`<line x1="${x}" y1="${y}" x2="${x}" y2="${y+gap-6}" stroke="${arrowColor}" stroke-width="2" marker-end="url(#layoutArrow)" />`;
  const arrows = [];
  const stepFills = [
    'var(--layout-step1)',
    'var(--layout-step2)',
    'var(--layout-step3)',
    'var(--layout-step4)',
    'var(--layout-step5)',
    'var(--layout-step6)',
    'var(--layout-step7)'
  ];
  const textColor = 'var(--layout-text)';
  const strokeColor = 'var(--layout-border)';
  function add(label, fillIdx){
    const fill = stepFills[fillIdx % stepFills.length];
    blocks.push(svgRect(x0, y, bw, bh, fill, strokeColor, label, textColor));
    arrows.push(arrowDown(x0+bw/2, y+bh));
    y += bh + gap;
  }
  add(`전처리 x${(P.preproc&&P.preproc.servers)||2}`, 0);
  add(`프린터 x${Np}`, 1);
  add(`세척1 x${(P.auto_post&&P.auto_post.washers_m1)||1} / 세척2 x${(P.auto_post&&P.auto_post.washers_m2)||1}`, 2);
  add(`플랫폼 세척 x${(P.platform_clean&&P.platform_clean.washers)||1}`, 3);
  add(`건조 x${(P.auto_post&&P.auto_post.dryers)||2}`, 4);
  add(`UV x${(P.auto_post&&P.auto_post.uv_units)||1}`, 5);
  add('스택커', 6);
  // Util bars stacked below
  const uKeys = ['preproc','printers','wash_m1','wash_m2','platform_wash','dryers','uv_units','manual_workers','amr'];
  const names = ['전처리','프린터','세척1','세척2','플랫폼 세척','건조','UV경화','작업자','AMR'];
  let gy = y + 10; let utilG = '';
  for(let i=0;i<uKeys.length;i++){
    const k = uKeys[i]; const p = util[k]||0; const label = names[i];
    const w = Math.max(0,Math.min(100,p*100)); const c=w<60?'#00ffa3':(w<85?'#00e5ff':'#ff2bd6');
    utilG += `<g transform="translate(${x0},${gy})"><text x="0" y="-2" font-size="11" fill="var(--layout-text)">${label} ${(p*100).toFixed(0)}%</text><g transform="translate(0,2)"><rect x="0" y="0" width="${w*2}" height="10" fill="${c}" rx="5"></rect></g></g>`;
    gy += 24;
  }
  const stepsBottom = y - gap;
  const utilBottom = utilG ? (gy + 10) : stepsBottom;
  const H = Math.max(540, utilBottom + 40, stepsBottom + 40);
  return `<div style="overflow-x:auto"><svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}">${markerDef}${blocks.join('')}${arrows.join('')}${utilG}</svg></div>`;
}

function parseKRW(val){ if(val==null) return NaN; const s = String(val).replace(/,/g,'').trim(); const n = parseFloat(s); return isNaN(n)? NaN : n; }
function attachKRWFormat(){
  const els = document.querySelectorAll('input.krw');
  els.forEach(el=>{
    el.addEventListener('blur', ()=>{
      const n = parseKRW(el.value); if(!isNaN(n)) el.value = n.toLocaleString('ko-KR');
    });
  });
}

async function run(){
  attachKRWFormat();
  let P;
  try { P = eval('(' + document.getElementById('raw').value + ')'); }
  catch(e){ try{ P = JSON.parse(document.getElementById('raw').value); }catch(e2){ alert('Raw JSON 파싱 오류'); return; } }

  P = P || {};
  // show progress immediately
  document.getElementById('out').innerHTML = '<div class="card"><b>시뮬레이션 실행 중...</b> (기본 30일 × 자동화/비자동화 비교로 1~2분 걸릴 수 있어요)</div>';
  P.horizon = { duration: document.getElementById('duration').value,
                 days: (document.getElementById('duration').value==='days'? parseInt(document.getElementById('days').value||'0'): null) };
  P.demand = P.demand || {};
  P.material = P.material || {};
  P.print = P.print || {};
  P.auto_post = P.auto_post || {};
  P.manual_post = P.manual_post || {};
  P.manual_ops = P.manual_ops || {};
  P.stacker_guard = P.stacker_guard || {};

  P.demand.platform_interarrival_min_min = parseFloat(document.getElementById('inter_min').value);
  P.demand.platform_interarrival_max_min = parseFloat(document.getElementById('inter_max').value);
  P.demand.initial_platforms = parseInt(document.getElementById('initial_platforms').value||'0');
  P.demand.install_time_min = parseFloat(document.getElementById('install_time').value||'60');
  P.demand.install_parallel = document.getElementById('install_parallel').checked;
  P.material.avg_part_mass_kg = parseFloat(document.getElementById('avg_part_mass_kg').value);
  {
    const rp = parseKRW(document.getElementById('resin_price_per_kg').value);
    if(!isNaN(rp)) P.material.resin_price_per_kg = rp;
  }
  P.print.t_print_per_platform_min = parseFloat(document.getElementById('t_print').value);
  P.print.max_parts_per_platform = parseInt(document.getElementById('max_parts_per_platform').value);
  P.material.max_parts_per_platform = P.print.max_parts_per_platform;
  P.print.printer_count = parseInt(document.getElementById('printer_count').value);
  const pdr = parseFloat(document.getElementById('print_defect_rate').value);
  if(!isNaN(pdr)) { P.print.defect_rate = pdr/100.0; }
  // preproc
  P.preproc = P.preproc || {};
  P.preproc.servers = parseInt(document.getElementById('preproc_servers').value);
  P.preproc.healing_time_per_platform_min = parseFloat(document.getElementById('heal_per_plat').value);
  P.preproc.placement_time_per_platform_min = parseFloat(document.getElementById('placement_per_plat').value);
      P.preproc.support_time_per_platform_min = parseFloat(document.getElementById('support_per_plat').value);
  P.print.install_time_min = parseFloat(document.getElementById('install_time').value||'60');
  P.print.install_parallel = document.getElementById('install_parallel').checked;
  P.auto_post.amr_count = parseInt(document.getElementById('amr_count').value);
  P.auto_post.amr_speed_m_per_s = parseFloat(document.getElementById('amr_speed').value);
  P.auto_post.amr_load_min = parseFloat(document.getElementById('amr_load').value);
  P.auto_post.amr_unload_min = parseFloat(document.getElementById('amr_unload').value);
  P.auto_post.washers_m1 = parseInt(document.getElementById('washers_m1').value);
  P.auto_post.washers_m2 = parseInt(document.getElementById('washers_m2').value);
  P.auto_post.dryers = parseInt(document.getElementById('dryers').value);
  P.auto_post.uv_units = parseInt(document.getElementById('uv_units').value);
  const dist_pw = parseFloat(document.getElementById('d_pw').value||'0');
  const dist_wd = parseFloat(document.getElementById('d_wd').value||'0');
  const dist_du = parseFloat(document.getElementById('d_du').value||'0');
  const dist_us = parseFloat(document.getElementById('d_us').value||'0');
  P.auto_post.dist_m = {
    printer_to_wash: dist_pw,
    wash_to_dry: dist_wd,
    dry_to_uv: dist_du,
    uv_to_stacker: dist_us,
    support_to_platform_wash: dist_pw,
    platform_wash_to_new: dist_pw
  };
  // common defect rates (apply to both automated/manual since same equipment)
  P.auto_post.defect_rate_wash = parseFloat(document.getElementById('c_def_w').value);
  P.auto_post.defect_rate_dry = parseFloat(document.getElementById('c_def_d').value);
  P.auto_post.defect_rate_uv = parseFloat(document.getElementById('c_def_u').value);
  P.auto_post.t_platform_install_min = parseFloat(document.getElementById('auto_install').value);
  P.auto_post.t_platform_remove_min  = parseFloat(document.getElementById('auto_remove').value);
  // common post-process times (used by both automated/manual)
  const c_w1 = parseFloat(document.getElementById('cw1').value);
  const c_w2 = parseFloat(document.getElementById('cw2').value);
  const c_d  = parseFloat(document.getElementById('cdry').value);
  const c_uv = parseFloat(document.getElementById('cuv').value);
  P.auto_post.t_wash1_min = c_w1;
  P.auto_post.t_wash2_min = c_w2;
  P.auto_post.t_dry_min   = c_d;
  P.auto_post.t_uv_min    = c_uv;

  // manual (uses common process times and defect rates; only unload/travel and install/remove differ)
  P.manual_post.t_platform_install_min = parseFloat(document.getElementById('man_install').value);
  P.manual_post.t_platform_remove_min  = parseFloat(document.getElementById('man_remove').value);
  P.manual_post.human_unload_min = parseFloat(document.getElementById('human_unload_min').value);
  P.manual_post.to_washer_travel_min = parseFloat(document.getElementById('to_washer').value||'0');
  P.manual_post.to_wash2_travel_min = parseFloat(document.getElementById('to_wash2').value||'0');
  P.manual_post.to_dryer_travel_min = parseFloat(document.getElementById('to_dryer').value||'0');
  P.manual_post.to_stacker_travel_min = parseFloat(document.getElementById('to_stacker').value||'0');
  P.manual_post.to_platform_wash_travel_min = P.manual_post.to_washer_travel_min;
  P.manual_post.to_newplatform_travel_min = P.manual_post.to_stacker_travel_min;
  P.manual_ops.work_shift = {
    start_hhmm: document.getElementById('ws_start').value,
    end_hhmm: document.getElementById('ws_end').value,
    work_cycle_days: 7,
    workdays_per_cycle: parseInt(document.getElementById('ws_days').value||'5')
  };
  P.manual_ops.workers = parseInt(document.getElementById('workers').value);
  P.manual_ops.support_time_per_platform_min = parseFloat(document.getElementById('sup_plat').value) || 0;
  P.manual_ops.support_time_per_part_min = parseFloat(document.getElementById('sup_part').value) || 0;
  P.manual_ops.finish_time_per_part_min = parseFloat(document.getElementById('fin').value) || 0;
  P.manual_ops.paint_time_per_part_min = parseFloat(document.getElementById('paint').value) || 0;
  P.manual_ops.move_platform_to_support_min = parseFloat(document.getElementById('move_plat_support').value) || 0;
  P.manual_ops.move_support_to_finish_min = parseFloat(document.getElementById('move_support_finish').value) || 0;
  P.manual_ops.move_finish_to_paint_min = parseFloat(document.getElementById('move_finish_paint').value) || 0;
  P.manual_ops.move_paint_to_storage_min = parseFloat(document.getElementById('move_paint_storage').value) || 0;
  // manual movers (pre-stacker human ops)
  P.manual_move = P.manual_move || {};
  P.manual_move.workers = parseInt(document.getElementById('mv_workers').value||'2');
  P.manual_move.speed_m_per_s = parseFloat(document.getElementById('mv_speed').value||'2');
  P.manual_move.work_shift = {
    start_hhmm: document.getElementById('mv_ws_start').value||'09:00',
    end_hhmm: document.getElementById('mv_ws_end').value||'18:00',
    work_cycle_days: 7,
    workdays_per_cycle: parseInt(document.getElementById('mv_ws_days').value||'5')
  };
  P.manual_move.prioritize_printer = !!document.getElementById('mv_prioritize').checked;
  const manualDist = P.manual_move.dist_m || {};
  P.manual_move.dist_m = {
    printer_to_wash: manualDist.printer_to_wash ?? dist_pw,
    wash1_to_wash2: manualDist.wash1_to_wash2 ?? dist_wd,
    wash_to_dry: manualDist.wash_to_dry ?? dist_wd,
    uv_to_stacker: manualDist.uv_to_stacker ?? dist_us,
    support_to_platform_wash: manualDist.support_to_platform_wash ?? dist_pw,
    platform_wash_to_new: manualDist.platform_wash_to_new ?? dist_pw
  };
  P.platform_clean = P.platform_clean || {};
  P.platform_clean.washers = parseInt(document.getElementById('platform_washers').value||'1');
  P.platform_clean.wash_time_min = parseFloat(document.getElementById('platform_wash_time').value||'60');
  const platInitInput = document.getElementById('platform_initial_count');
  const platInitVal = parseInt((platInitInput && platInitInput.value) || document.getElementById('initial_platforms').value||'10');
  P.platform_clean.initial_platforms = platInitVal;
  P.stacker_guard.enabled = !!document.getElementById('stacker_guard_enabled').checked;
  const guardLimit = parseInt(document.getElementById('stacker_guard_limit').value||'0');
  P.stacker_guard.max_platforms = isNaN(guardLimit)? 0 : guardLimit;
  const TRACE_PLATFORM_CAP_DEFAULT = 150;
  const TRACE_PART_CAP_DEFAULT = 20;
  const TRACE_EVENT_CAP_DEFAULT = 500000;
  if (P.viz_max_platforms == null) P.viz_max_platforms = TRACE_PLATFORM_CAP_DEFAULT;
  if (P.viz_max_parts_per_platform == null) P.viz_max_parts_per_platform = TRACE_PART_CAP_DEFAULT;
  if (P.viz_max_events == null) P.viz_max_events = TRACE_EVENT_CAP_DEFAULT;

  // costs
  P.cost = P.cost || {};
  P.cost.wage_per_hour_krw = parseKRW(document.getElementById('cost_wage').value);
  P.cost.depreciation_years = parseFloat(document.getElementById('dep_years').value);
  P.cost.overhead_krw_per_month = parseKRW(document.getElementById('overhead').value);
  P.cost.equipment = {
    printer: parseKRW(document.getElementById('c_printer').value),
    washer: parseKRW(document.getElementById('c_washer').value),
    dryer: parseKRW(document.getElementById('c_dryer').value),
    uv: parseKRW(document.getElementById('c_uv').value),
    amr: parseKRW(document.getElementById('c_amr').value),
    preproc_server: parseKRW(document.getElementById('c_pre').value),
    platform_washer: parseKRW(document.getElementById('c_platform_wash').value)
  };
  try{
    const snapshot = JSON.parse(JSON.stringify(P));
    window.__LAST_PARAMS__ = snapshot;
    const rawBox = document.getElementById('raw');
    if(rawBox) rawBox.value = JSON.stringify(snapshot, null, 2);
  }catch(err){ console.warn('Unable to snapshot params', err); }

  const payload = { params: P, seed: parseInt(document.getElementById('seed').value||'42') };
  let data;
  try{
    const res = await fetch('/simulate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    data = await res.json();
  }catch(err){
    console.error(err);
    alert('네트워크 또는 서버 오류가 발생했습니다. 콘솔을 확인해주세요.');
    return;
  }
  if(!data.ok) { alert('에러: '+data.error); return; }

  const a = data.results.automated; const m = data.results.manual;
  const utilA = a.utilization, utilM = m? m.utilization : null;
  const layoutA = layoutSvgHtml(P, utilA);
  const layoutM = m? layoutSvgHtml(P, utilM) : '';
  function tbl(r){
    const totalParts = (r.started_parts && r.started_parts>0)? r.started_parts : (r.completed_parts + r.scrapped_parts);
    const scrapDisplay = (()=>{
      if(!totalParts || totalParts===0) return r.scrapped_parts.toLocaleString();
      const pct = (r.scrapped_parts/totalParts)*100;
      return `${r.scrapped_parts.toLocaleString()} / ${totalParts.toLocaleString()} (${pct.toFixed(2)}%)`;
    })();
    return `
      <table>
        <tr><th>지표</th><th>값</th></tr>
        <tr><td>전체 진행된 파트 수</td><td>${totalParts? totalParts.toLocaleString() : '-'}</td></tr>
        <tr><td>완료 파트 수</td><td>${r.completed_parts}</td></tr>
        <tr><td>스크랩 수</td><td>${scrapDisplay}</td></tr>
        <tr><td>최종 수율</td><td>${r.yield_final===null?'-':r.yield_final.toFixed(3)}</td></tr>
        <tr><td>완료 빌드플랫폼 수</td><td>${r.completed_platforms}</td></tr>
        <tr><td>평균 리드타임 (분)</td><td>${r.avg_platform_lead_time_min===null?'-':r.avg_platform_lead_time_min.toFixed(1)}</td></tr>
        <tr><td>레진 사용 (kg)</td><td>${r.resin_used_kg}</td></tr>
        <tr><td>시뮬레이션 기간 (분)</td><td>${r.horizon_minutes}</td></tr>
      </table>`;
  }
  function utilTbl(u){
    const label = {
      printers:'프린터', wash_m1:'세척1', wash_m2:'세척2', dryers:'건조기', uv_units:'UV경화', platform_wash:'플랫폼 세척', amr:'AMR', preproc:'전처리', manual_workers:'수작업'
    };
    return `<table>
      <tr><th>자원</th><th>가동률</th><th></th></tr>
      ${Object.keys(u).map(k=>{ const p=u[k]*100; const name = label[k]||k; return `<tr><td>${name}</td><td>${p.toFixed(1)}%</td><td>${barHtml(p)}</td></tr>`; }).join('')}
    </table>`;
  }
  function costTbl(c){
    return `<table>
      <tr><th>비용 항목</th><th>금액(KRW)</th></tr>
      <tr><td>노동비</td><td>${c.labor_krw.toLocaleString()}</td></tr>
      <tr><td>소재비</td><td>${c.material_krw.toLocaleString()}</td></tr>
      <tr><td>감가상각</td><td>${c.depreciation_krw.toLocaleString()}</td></tr>
      <tr><td>인프라/고정비</td><td>${c.overhead_krw.toLocaleString()}</td></tr>
      <tr><td><b>총비용</b></td><td><b>${c.total_krw.toLocaleString()}</b></td></tr>
      <tr><td><b>단가(파트당)</b></td><td><b>${c.unit_cost_krw===null?'-':c.unit_cost_krw.toLocaleString()}</b></td></tr>
    </table>`;
  }
  function waitTbl(w){
    const label = {printer:'프린터', wash_m1:'세척1', wash_m2:'세척2', dryers:'건조기', uv:'UV경화', amr:'AMR', preproc:'전처리', manual_movers:'이송/운전 인력', platform_wash:'플랫폼 세척', stacker_guard:'스택커 가드'};
    const keys = Object.keys(w||{});
    if(keys.length===0) return '<small>대기시간 데이터 없음</small>';
    return `<table>
      <tr><th>자원</th><th>평균 대기(분)</th><th>최대 대기(분)</th><th>요청수</th></tr>
      ${keys.map(k=>`<tr><td>${label[k]||k}</td><td>${(w[k].avg_min||0).toFixed(1)}</td><td>${(w[k].max_min||0).toFixed(1)}</td><td>${w[k].req}</td></tr>`).join('')}
    </table>`;
  }

  function simpyExtraHtml(simpy){
    if(!simpy) return '';
    const k = simpy.kpi || {};
    const scrap = simpy.scrap_by_stage || {};
    const routes = simpy.amr_route_counts || {};
    const wipHist = simpy.stacker_wip_history || [];
    const rowKV = (obj)=> Object.keys(obj||{}).map(key=>`<tr><td>${key}</td><td>${obj[key]}</td></tr>`).join('');
    const scrapRows = Object.keys(scrap||{}).length? rowKV(scrap) : '<tr><td colspan="2"><small>데이터 없음</small></td></tr>';
    const routeRows = Object.keys(routes||{}).length? rowKV(routes) : '<tr><td colspan="2"><small>데이터 없음</small></td></tr>';
    const lastWip = (wipHist && wipHist.length)? wipHist[wipHist.length-1] : null;
    const wipSummary = lastWip ? (typeof lastWip==='object' ? JSON.stringify(lastWip) : String(lastWip)) : '데이터 없음';
    return `
      <div class="card" style="margin-top:14px">
        <h3>추가 결과 (SimPy 코드 기반)</h3>
        <p style="margin-top:6px;color:var(--muted);font-size:13px">기존 웹 애니메이션/디자인은 유지하고, 내 코드(main_SimPy/KPI/logger)에서만 나오는 결과를 덧붙여 표시합니다.</p>
        <div class="grid2">
          <div>
            <h4>Scrap by Stage</h4>
            <table>
              <tr><th>Stage</th><th>Scrap</th></tr>
              ${scrapRows}
            </table>
          </div>
          <div>
            <h4>AMR Route Counts</h4>
            <table>
              <tr><th>Route</th><th>Count</th></tr>
              ${routeRows}
            </table>
          </div>
        </div>
        <h4 style="margin-top:12px">Stacker WIP (Latest)</h4>
        <div style="padding:10px;border:1px solid var(--line);border-radius:10px;background:var(--card2);font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;font-size:12px;white-space:pre-wrap">${wipSummary}</div>
      </div>
    `;
  }

  // 2D Canvas animation -----------------------------------------
  function renderFactory2D(containerId, trace, opts){
    opts = opts || {};
    const theme = currentThemeVars();
    const fontScale = 1.3;
    const fontPx = (size, weight='normal')=>{
      const px = Math.round(size * fontScale);
      return `${weight && weight!=='normal' ? weight + ' ' : ''}${px}px system-ui`;
    };
    const partRadius = 2.4;
    const amrRadius = 4;
    const moverRadius = 4;
    const platformRadius = 5;
    const stageDim = {
      new_plat:{w:78,h:24},
      printers:{w:96,h:28},
      platform_wash:{w:86,h:24},
      wash1:{w:82,h:24},
      wash2:{w:82,h:24},
      dry:{w:86,h:24},
      uv:{w:86,h:24},
      stacker:{w:96,h:28},
      support_plat:{w:86,h:24},
      support_part:{w:86,h:24},
      finish:{w:82,h:24},
      paint:{w:82,h:24},
      storage:{w:90,h:26}
    };
    const tokenRadius = (token)=>{
      if(!token) return platformRadius;
      if(token.isPart) return partRadius;
      if(token.isAmr) return amrRadius;
      if(token.isMover) return moverRadius;
      return platformRadius;
    };
    const automated = !!opts.automated;
    const horizonMinutes = Number(opts.horizonMinutes || 0);
    const container = document.getElementById(containerId);
    if(!container){
      if(window.__renderFactoryCache){
        delete window.__renderFactoryCache[containerId];
      }
      return;
    }
    const W = container.clientWidth || 1000;
    const H = Math.max(700, container.clientHeight || 700);
    container.style.minHeight = `${H}px`;
    container.style.width = '100%';
    const hostCard = container.closest('.card');
    if(hostCard){
      hostCard.style.overflow = 'visible';
    }
    if(!trace || trace.length===0){
      container.innerHTML = `<div style="padding:8px;color:${theme.noticeText};background:${theme.noticeBg};border-radius:8px;border:1px solid ${theme.line}">애니메이션을 위한 트레이스를 찾을 수 없습니다. 시뮬레이션을 다시 실행해 주세요.</div>`;
      if(window.__renderFactoryCache){
        delete window.__renderFactoryCache[containerId];
      }
      return;
    }
    window.__renderFactoryCache = window.__renderFactoryCache || {};
    window.__renderFactoryCache[containerId] = {trace, opts: Object.assign({}, opts)};
    container.innerHTML = `
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap;background:${theme.toolbarBg};border:1px solid ${theme.toolbarBorder};border-radius:10px;padding:10px 12px">
        <button id="${containerId}_play" style="padding:6px 10px;border-radius:6px;border:1px solid ${theme.toolbarBorder};background:${theme.accent};color:#fff">Play</button>
        <button id="${containerId}_pause" style="padding:6px 10px;border-radius:6px;border:1px solid ${theme.toolbarBorder};background:${theme.controlBg};color:${theme.controlText}">Pause</button>
        <button id="${containerId}_restart" style="padding:6px 10px;border-radius:6px;border:1px solid ${theme.toolbarBorder};background:${theme.controlBg};color:${theme.controlText}">Restart</button>
        <label style="color:${theme.text};margin-left:6px">Speed</label>
        <select id="${containerId}_speed" style="background:${theme.controlBg};color:${theme.controlText};border:1px solid ${theme.controlBorder};border-radius:6px;padding:4px">
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="2">2x</option>
          <option value="5">5x</option>
        </select>
        <label style="color:${theme.text};margin-left:12px">Playback</label>
        <select id="${containerId}_duration" style="background:${theme.controlBg};color:${theme.controlText};border:1px solid ${theme.controlBorder};border-radius:6px;padding:4px">
          <option value="60000" selected>1분</option>
          <option value="120000">2분</option>
          <option value="300000">5분</option>
        </select>
      </div>
      <canvas width="${W}" height="${H}"></canvas>`;
    const canvas = container.querySelector('canvas');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    ctx.scale(dpr, dpr);
    // stations (x,y) laid out to avoid overlap (two rows)
    const centerY = H * 0.55;
    const processSpacing = Math.min(130, H * 0.18);
    let manualSpacing = Math.min(130, H * 0.20);

    const newPlatLeft = 60;
    const newPlatCenter = newPlatLeft + stageDim.new_plat.w / 2;
    const gapCount = 5;
    const marginRight = 60;
    const maxRight = W - stageDim.storage.w/2 - marginRight;
    let baseSpacing = (maxRight - newPlatCenter) / gapCount;
    const minGap = 170;
    if(baseSpacing < minGap) baseSpacing = minGap;
    if(newPlatCenter + baseSpacing * gapCount > maxRight){
      baseSpacing = Math.max((maxRight - newPlatCenter) / gapCount, minGap * 0.7);
    }
    const columns = Array.from({length: gapCount + 1}, (_, i)=> newPlatCenter + baseSpacing * i);
    const printersCenter = columns[1];
    const processCenter = columns[2];
    let stackerCenter = columns[3];
    let supportColumnX = columns[4];
    let storageCenterX = columns[5];

    const desiredGap = baseSpacing;
    if(stackerCenter - processCenter < desiredGap){
      stackerCenter = processCenter + desiredGap;
    }
    if(supportColumnX - stackerCenter < desiredGap){
      supportColumnX = stackerCenter + desiredGap;
    }
    if(storageCenterX - supportColumnX < desiredGap){
      storageCenterX = supportColumnX + desiredGap;
    }
    if(storageCenterX > maxRight){
      const adjust = storageCenterX - maxRight;
      storageCenterX = maxRight;
      supportColumnX -= adjust;
      stackerCenter -= adjust;
    }

    const printersGapY = Math.min(processSpacing * 0.6, 90);
    let wash1Y = centerY - processSpacing * 1.5;
    if(wash1Y < 140){
      const diff = 140 - wash1Y;
      wash1Y += diff;
    }
    let wash2Y = wash1Y + processSpacing;
    let dryY = wash2Y + processSpacing;
    let uvY = dryY + processSpacing;
    const maxBottom = H - 80;
    if(uvY + stageDim.uv.h/2 > maxBottom){
      const shift = uvY + stageDim.uv.h/2 - maxBottom;
      wash1Y -= shift;
      wash2Y -= shift;
      dryY -= shift;
      uvY -= shift;
    }
    const printersY = (wash1Y + wash2Y) / 2;
    let platformWashY = (dryY + uvY) / 2;
    if(platformWashY <= printersY + 20){
      platformWashY = printersY + 20;
    }

    const manualStart = centerY - manualSpacing * 1.5;
    let supportTopY = manualStart;
    let supportPartY = supportTopY + manualSpacing;
    let finishY = supportPartY + manualSpacing;
    let paintY = finishY + manualSpacing;
    if(supportTopY < 140){
      const shift = 140 - supportTopY;
      supportTopY += shift;
      supportPartY += shift;
      finishY += shift;
      paintY += shift;
    }
    const manualBottom = paintY + stageDim.paint.h/2;
    if(manualBottom > maxBottom){
      const shift = manualBottom - maxBottom;
      supportTopY -= shift;
      supportPartY -= shift;
      finishY -= shift;
      paintY -= shift;
    }
    const storageCenterY = centerY;

    const pos = {
      new_plat:[newPlatCenter, centerY],
      printers:[printersCenter, printersY],
      platform_wash:[printersCenter, platformWashY],
      wash1:[processCenter, wash1Y],
      wash2:[processCenter, wash2Y],
      dry:[processCenter, dryY],
      uv:[processCenter, uvY],
      stacker:[stackerCenter, centerY],
      support_plat:[supportColumnX, supportTopY],
      support_part:[supportColumnX, supportPartY],
      finish:[supportColumnX, finishY],
      paint:[supportColumnX, paintY],
      storage:[storageCenterX, storageCenterY]
    };
    const labelPos = { x: (newPlatCenter + stackerCenter) / 2, y: 36 };
    const workShift = (P.manual_move && P.manual_move.work_shift) || {start_hhmm:'09:00', end_hhmm:'18:00', work_cycle_days:7, workdays_per_cycle:5};
    const cycleDays = Number(workShift.work_cycle_days || 7);
    const workDays = Number(workShift.workdays_per_cycle || cycleDays);
    const weekdays = ['월요일','화요일','수요일','목요일','금요일','토요일','일요일'];
    const rainbowStations = new Set(['printers','platform_wash','wash1','wash2','dry','uv','stacker']);
    const rainbowColors = ['#ff6b6b','#f7c948','#4ade80','#38bdf8','#818cf8','#f472b6'];
    let busyStations = new Set();
    // preprocess trace
    const byId = {}; (trace||[]).forEach(e=>{ (byId[e.id]=byId[e.id]||[]).push(e); });
    const pickTail = (arr, n)=> arr.slice(Math.max(0, arr.length - n));
    const allIds = Object.keys(byId);
    const rawTokens = allIds.map((id)=>{
      const segs = byId[id].sort((a,b)=>a.t0-b.t0);
      const isPart = id.includes('-P');
      const isAmr = id.endsWith('-AMR');
      const isMover = id.endsWith('-MOV');
      const isCarrier = isAmr || isMover;
      const baseId = isPart ? id.split('-P')[0] : (isCarrier ? id.replace(/-(AMR|MOV)$/,'') : id);
      const color = isPart ? '#22c55e' : (isAmr ? '#fde047' : (isMover ? '#3b82f6' : '#f97316'));
      const isAmrIdle = id.startsWith('AMR-') && !isAmr;
      return {id, color, segs, isPart, isAmr, isMover, isCarrier, baseId, isAmrIdle};
    });
    const maxPlat = Number(P?.viz_max_platforms||0);
    const maxPart = Number(P?.viz_max_parts_per_platform||0);
    const basePlatformTokens = rawTokens.filter(t=> t.id.startsWith('PLAT-') && !t.isCarrier && !t.isPart);
    const baseIdsOrdered = basePlatformTokens.map(t=>t.id);
    const selectedBaseIds = (maxPlat && maxPlat>0)? pickTail(baseIdsOrdered, maxPlat) : baseIdsOrdered;
    const selectedBaseSet = new Set(selectedBaseIds);
    const baseTokenMap = new Map(basePlatformTokens.map(t=>[t.id, t]));
    const partByBase = new Map();
    rawTokens.filter(t=>t.isPart && selectedBaseSet.has(t.baseId)).forEach(t=>{
      if(!partByBase.has(t.baseId)) partByBase.set(t.baseId, []);
      partByBase.get(t.baseId).push(t);
    });
    const moverByBase = new Map();
    rawTokens.filter(t=>t.isMover && selectedBaseSet.has(t.baseId)).forEach(t=>{
      if(!moverByBase.has(t.baseId)) moverByBase.set(t.baseId, []);
      moverByBase.get(t.baseId).push(t);
    });
    const amrByBase = new Map();
    rawTokens.filter(t=>t.isAmr && selectedBaseSet.has(t.baseId)).forEach(t=>{
      if(!amrByBase.has(t.baseId)) amrByBase.set(t.baseId, []);
      amrByBase.get(t.baseId).push(t);
    });
    const tokens = [];
    const seenIds = new Set();
    const pushToken = (token)=>{
      if(!token || seenIds.has(token.id)) return;
      tokens.push(token);
      seenIds.add(token.id);
    };
    selectedBaseIds.forEach(baseId=>{
      pushToken(baseTokenMap.get(baseId));
      const parts = partByBase.get(baseId) || [];
      const chosenParts = (maxPart && maxPart>0)? pickTail(parts, maxPart) : parts;
      chosenParts.forEach(pushToken);
      (moverByBase.get(baseId) || []).forEach(pushToken);
      (amrByBase.get(baseId) || []).forEach(pushToken);
    });
    rawTokens.filter(t=>t.isAmrIdle).forEach(pushToken);
    rawTokens.filter(t=>!t.id.startsWith('PLAT-') && !t.isPart && !t.isCarrier && !t.isAmrIdle).forEach(pushToken);
    // time scale
    let simT0=Infinity, simT1=0; tokens.forEach(t=>t.segs.forEach(s=>{ simT0=Math.min(simT0,s.t0); simT1=Math.max(simT1,s.t1);}));
    // 목표 전체 재생시간을 선택(1/2/5분). 전체 시뮬 분(simT1-simT0)을 targetTotalMs에 매핑
    let targetTotalMs = 60000; // 기본 1분
    function recomputeScale(){
      const span = Math.max(1, (simT1 - simT0));
      minToMsBase = targetTotalMs / span; // 1 sim min 당 ms
    }
    let minToMsBase = 1; recomputeScale();
    let speed = 1;
    let playing = false;
    let tPrev = performance.now();
    let simNow = simT0;
    let rainbowPhase = 0;
    // controls
    container.querySelector(`#${containerId}_play`).onclick = ()=>{ playing = true; tPrev = performance.now(); };
    container.querySelector(`#${containerId}_pause`).onclick = ()=>{ playing = false; };
    container.querySelector(`#${containerId}_restart`).onclick = ()=>{ simNow = simT0; rainbowPhase = 0; tPrev = performance.now(); };
    container.querySelector(`#${containerId}_speed`).onchange = (e)=>{ speed = parseFloat(e.target.value||'1')||1; };
    container.querySelector(`#${containerId}_duration`).onchange = (e)=>{ targetTotalMs = parseInt(e.target.value||'60000'); recomputeScale(); };
    const moveMap = {
      to_printer:['new_plat','printers'],
      to_wash:['printers','wash1'],
      to_wash2:['wash1','wash2'],
      to_dry:['wash2','dry'],
      to_uv:['dry','uv'],
      to_stacker:['uv','stacker'],
      to_platform_wash:['support_plat','platform_wash'],
      to_new_platform:['platform_wash','new_plat'],
      to_support_plat:['stacker','support_plat'],
      to_finish:['support_part','finish'],
      to_paint:['finish','paint'],
      to_storage:['paint','storage'],
      amr_return:['stacker','new_plat'],
      manual_return:['stacker','wash1']
    };
    const normalizeStage = (st)=>{
      if(!st) return '';
      const queueSuffix = '_queue';
      const holdSuffix = '_hold';
      let base = st;
      if(base.endsWith(queueSuffix)) base = base.slice(0, -queueSuffix.length);
      if(base.endsWith(holdSuffix)) base = base.slice(0, -holdSuffix.length);
      return base;
    };
    const stagePos = st => {
      const base = normalizeStage(st);
      return (base==='install'||base==='print'||base==='remove'||base==='unload')? pos.printers : pos[base]||pos.printers;
    };
    function lerp(a,b,u){ return [a[0]+(b[0]-a[0])*u, a[1]+(b[1]-a[1])*u]; }

    function draw(){
      const now = performance.now();
      let deltaSim = 0;
      if(playing){
        const deltaMs = now - tPrev;
        deltaSim = (deltaMs)/minToMsBase*speed;
        simNow += deltaSim;
        rainbowPhase += (deltaMs/1000) * (1.8 * speed);
      }
      tPrev = now;
      // bg: day/night (work shift) visualization
      const minutesPerDay = 24*60;
      function inShift(min){
        try{
          const ws = (P.manual_ops && P.manual_ops.work_shift) || {start_hhmm:'09:00', end_hhmm:'18:00', workdays_per_month:22};
          const hhmm = s=>{ const [h,m]=String(s).split(':'); return (+h)*60 + (+m); };
          const start = hhmm(ws.start_hhmm), end = hhmm(ws.end_hhmm);
          const dayMin = Math.floor(min % minutesPerDay);
          const totalSimDays = horizonMinutes>0 ? Math.max(1, Math.ceil(horizonMinutes / minutesPerDay)) : 30;
          const dayIdxAbs = Math.floor(min / minutesPerDay);
          const dayIdx = totalSimDays ? (dayIdxAbs % totalSimDays) : dayIdxAbs;
          const workdaysLimit = ws.workdays_per_month || totalSimDays || 30;
          if(dayIdx >= workdaysLimit) return false;
          return (dayMin>=start && dayMin<end);
        }catch(e){ return true; }
      }
      const isWork = inShift(simNow);
      const dayIndex = Math.max(0, Math.floor(simNow / minutesPerDay));
      const totalDays = horizonMinutes>0 ? Math.max(1, Math.ceil(horizonMinutes / minutesPerDay)) : null;
      const dayClamp = totalDays ? Math.min(dayIndex + 1, totalDays) : (dayIndex + 1);
      const dayLabel = totalDays ? `${dayClamp}일차 / ${totalDays}일` : `${dayClamp}일차`;
      const weekIndex = Math.floor(dayIndex / 7) + 1;
      const dayInCycle = cycleDays > 0 ? (dayIndex % cycleDays) : dayIndex;
      const workActive = cycleDays <= 0 ? isWork : (dayInCycle < workDays && isWork);
      const phaseLabel = workActive ? '작업 시간' : '휴무 시간';
      const weekdayLabel = weekdays[dayIndex % weekdays.length];
      const workBg = theme.canvasSurface;
      const idleBg = theme.nightBg || theme.canvasSurfaceAlt;
      ctx.fillStyle = isWork ? workBg : idleBg;
      ctx.fillRect(0,0,W,H);

      // legend
      ctx.fillStyle = '#f97316';
      ctx.beginPath(); ctx.arc(60,32,6,0,Math.PI*2); ctx.fill();
      ctx.fillStyle = theme.canvasText; ctx.font = fontPx(12);
      ctx.fillText('플랫폼', 72,36);
      ctx.fillStyle = '#22c55e';
      ctx.beginPath(); ctx.arc(132,32,6,0,Math.PI*2); ctx.fill();
      ctx.fillStyle = theme.canvasText;
      ctx.fillText('파트', 144,36);
      ctx.fillStyle = '#fde047';
      ctx.beginPath(); ctx.arc(200,32,6,0,Math.PI*2); ctx.fill();
      ctx.fillStyle = theme.canvasText;
      ctx.fillText('AMR', 212,36);
      ctx.fillStyle = '#3b82f6';
      ctx.beginPath(); ctx.arc(268,32,6,0,Math.PI*2); ctx.fill();
      ctx.beginPath(); ctx.strokeStyle=theme.accent; ctx.lineWidth=1.5; ctx.arc(268,32,6,0,Math.PI*2); ctx.stroke();
      ctx.lineWidth = 1;
      ctx.fillStyle = theme.canvasText;
      ctx.fillText('운송 인력', 280,36);
      ctx.font = fontPx(15);
      ctx.fillStyle = theme.muted;
      ctx.textAlign = 'left';
      ctx.fillText(`${dayLabel} · ${weekIndex}주차 ${weekdayLabel} · ${phaseLabel}`, labelPos.x, labelPos.y);
      ctx.font = fontPx(12);
      ctx.textAlign = 'start';

      // 1) 현재 시점 상태 계산(이동/정지)
      const atStation = {}; // station -> array of tokens to stack
      const states = tokens.map(t=>{
        let seg=null; for(let i=0;i<t.segs.length;i++){ const s=t.segs[i]; if(simNow>=s.t0 && simNow<=s.t1){ seg=s; break; } }
        if(!seg) return null;
        const st = seg.stage; const baseStage = normalizeStage(st);
        const moving = !!moveMap[st];
       if(!moving){
         // 정지 상태(스테이션 위에서 처리/대기)
         // 스테이션 키
         const key = (baseStage==='install'||baseStage==='print'||baseStage==='remove'||baseStage==='unload')? 'printers' : baseStage;
          if(!t.isMover){
            (atStation[key] = atStation[key] || []).push(t);
          }
         return {t, moving:false, station:key, stage:st};
       }else{
          // 이동 상태 → 위치 보간 계산
          const [a,b]=moveMap[st]; const p0=pos[a], p1=pos[b];
          const tt = Math.max(0, Math.min(1,(simNow-seg.t0)/Math.max(1e-6,(seg.t1-seg.t0))));
          const [x,y] = lerp(p0,p1,tt);
          return {t, moving:true, stage:st, x, y};
        }
      }).filter(Boolean);
      const busySet = new Set();
      states.forEach(s=>{
        if(!s || s.moving) return;
        if(!rainbowStations.has(s.station)) return;
        const stageName = String(s.stage||'');
        if(stageName.endsWith('_queue')) return;
        if(s.t.isPart || s.t.isAmr || s.t.isMover) return;
        busySet.add(s.station);
      });
      busyStations = busySet;

      function drawRainbowFrame(p, w, h){
        const margin = 4;
        const radius = Math.max(w, h) * 0.55 + margin * 2;
        const dirX = Math.cos(rainbowPhase);
        const dirY = Math.sin(rainbowPhase);
        const gx0 = p[0] + dirX * radius;
        const gy0 = p[1] + dirY * radius;
        const gx1 = p[0] - dirX * radius;
        const gy1 = p[1] - dirY * radius;
        const gradient = ctx.createLinearGradient(gx0, gy0, gx1, gy1);
        rainbowColors.forEach((color, idx)=>{
          gradient.addColorStop(idx/(rainbowColors.length-1), color);
        });
        ctx.save();
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 4;
        ctx.strokeRect(p[0]-w/2-margin, p[1]-h/2-margin, w+margin*2, h+margin*2);
        ctx.restore();
      }
      function drawBox(key, label){
        const dim = stageDim[key] || {w:90,h:26};
        const p = pos[key];
        const w = dim.w;
        const h = dim.h;
        if(key && busyStations.has(key)){
          drawRainbowFrame(p,w,h);
        }
        const stationFill = isWork ? theme.canvasStation : (theme.canvasStationNight || theme.canvasStation);
        ctx.fillStyle = stationFill;
        ctx.fillRect(p[0]-w/2,p[1]-h/2,w,h);
        ctx.fillStyle = theme.canvasText;
        ctx.font = fontPx(12, '600');
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(label, p[0], p[1]+h/2+6);
        ctx.textAlign = 'start';
        ctx.textBaseline = 'alphabetic';
      }
      drawBox('new_plat','신규플랫폼');
      drawBox('printers','프린터');
      drawBox('platform_wash','플랫폼 세척');
      drawBox('wash1','세척1');
      drawBox('wash2','세척2');
      drawBox('dry','건조');
      drawBox('uv','UV');
      drawBox('stacker','스택커');
      drawBox('support_plat','플랫폼 서포트');
      drawBox('support_part','파트 서포트');
      drawBox('finish','사상');
      drawBox('paint','페인팅');
      drawBox('storage','보관');

      const movingStages = new Set(Object.keys(moveMap));
      const carrierOverlay = new Map();
      states.forEach(s=>{
        if(!s.moving) return;
        if(!movingStages.has(s.stage)) return;
        const base = s.t.baseId || s.t.id;
        if(!carrierOverlay.has(s.stage)) carrierOverlay.set(s.stage, new Map());
        const stageMap = carrierOverlay.get(s.stage);
        const rec = stageMap.get(base) || {carriers:[], cargo:null};
        if(s.t.isCarrier){
          rec.carriers.push(s);
        }else if(!s.t.isPart){
          rec.cargo = s;
        }
        stageMap.set(base, rec);
      });

      const drawnIds = new Set();
      carrierOverlay.forEach(stageMap=>{
        stageMap.forEach(rec=>{
          if(rec.carriers.length===0 && !rec.cargo) return;
          const anchor = rec.cargo || rec.carriers[0];
          if(!anchor) return;
          const x = anchor.x, y = anchor.y;
          const hasCargo = !!rec.cargo;
          rec.carriers.forEach(carrierState=>{
            if(carrierState.t.isAmr){
              ctx.save(); ctx.globalAlpha=0.88; ctx.beginPath(); ctx.fillStyle='#fde047'; ctx.arc(x,y,9,0,Math.PI*2); ctx.fill(); ctx.restore();
              ctx.beginPath(); ctx.strokeStyle='#facc15'; ctx.lineWidth=2; ctx.arc(x,y,9,0,Math.PI*2); ctx.stroke();
            }else if(carrierState.t.isMover){
              if(hasCargo){
                ctx.beginPath(); ctx.strokeStyle='#60a5fa'; ctx.lineWidth=3; ctx.arc(x,y,8,0,Math.PI*2); ctx.stroke();
              }else{
                ctx.save(); ctx.globalAlpha=0.82; ctx.beginPath(); ctx.fillStyle='#3b82f6'; ctx.arc(x,y,7,0,Math.PI*2); ctx.fill(); ctx.restore();
                ctx.beginPath(); ctx.strokeStyle='#60a5fa'; ctx.lineWidth=2; ctx.arc(x,y,7,0,Math.PI*2); ctx.stroke();
              }
            }else{
              ctx.save(); ctx.globalAlpha=0.82; ctx.beginPath(); ctx.fillStyle=carrierState.t.color; ctx.arc(x,y,6,0,Math.PI*2); ctx.fill(); ctx.restore();
            }
            drawnIds.add(carrierState.t.id);
          });
          ctx.lineWidth = 1;
          if(rec.cargo && !drawnIds.has(rec.cargo.t.id)){
            const cargoRadius = rec.carriers.some(c=>c.t.isMover) ? 5.5 : 5;
            ctx.save(); ctx.globalAlpha=0.82; ctx.beginPath(); ctx.fillStyle=rec.cargo.t.color; ctx.arc(x,y,cargoRadius,0,Math.PI*2); ctx.fill(); ctx.restore();
            drawnIds.add(rec.cargo.t.id);
          }else if(!rec.cargo){
            const hasAmr = rec.carriers.some(c=>c.t.isAmr);
            const hasMover = rec.carriers.some(c=>c.t.isMover);
            if(hasAmr){
              ctx.save(); ctx.globalAlpha=0.8; ctx.beginPath(); ctx.fillStyle='#fde68a'; ctx.arc(x,y,3.5,0,Math.PI*2); ctx.fill(); ctx.restore();
            }else if(hasMover){
              ctx.save(); ctx.globalAlpha=0.8; ctx.beginPath(); ctx.fillStyle='#bfdbfe'; ctx.arc(x,y,3,0,Math.PI*2); ctx.fill(); ctx.restore();
            }
          }
        });
      });
      ctx.strokeStyle = theme.canvasText;

      // 2) 이동 토큰 그리기
      const trailSpan = 5;
      const trailAlpha = 0.3;
      const trailRadiusFactor = 1.4;
      const trailMap = new Map();
      tokens.forEach(t=>{ trailMap.set(t.id, {color:t.color, radius:tokenRadius(t)}); });
      tokens.forEach(t=>{
        const meta = trailMap.get(t.id);
        const segments = t.segs.slice(-trailSpan);
        segments.forEach(seg=>{
          if(seg.t1 <= simNow) return;
          const total = seg.t1 - seg.t0;
          if(total <= 0) return;
          const progress = (simNow - seg.t0) / total;
          if(progress < 0 || progress > 1) return;
          const move = moveMap[seg.stage];
          if(!move) return;
          const [stageFrom, stageTo] = move;
          const p0 = pos[stageFrom];
          const p1 = pos[stageTo];
          if(!p0 || !p1) return;
          const x = p0[0] + (p1[0]-p0[0])*progress;
          const y = p0[1] + (p1[1]-p0[1])*progress;
          ctx.save();
          ctx.globalAlpha = trailAlpha * (1 - Math.abs(0.5 - progress));
          ctx.beginPath(); ctx.fillStyle=meta.color; ctx.arc(x,y,meta.radius*trailRadiusFactor,0,Math.PI*2); ctx.fill();
          ctx.restore();
        });
      });

      states.filter(s=>s && s.moving).forEach(s=>{
        if(drawnIds.has(s.t.id)) return;
        const radius = tokenRadius(s.t);
        ctx.save();
        ctx.globalAlpha = 0.9;
        ctx.beginPath(); ctx.fillStyle=s.t.color; ctx.arc(s.x,s.y,radius,0,Math.PI*2); ctx.fill();
        ctx.restore();
      });

      // 3) 스테이션 정지 토큰을 장비 대수만큼 분할하여 위로 쌓기
      const workerCount = Number((P?.manual_ops?.workers) || 1) || 1;
      const defaults = {
        new_plat: 1,
        stacker: 1,
        printers: (P?.print?.printer_count || P?.print?.printers?.length || 2),
        wash_m1: (P?.auto_post?.washers_m1||1),
        wash_m2: (P?.auto_post?.washers_m2||1),
        platform_wash: (P?.platform_clean?.washers||1),
        dryers: (P?.auto_post?.dryers||1),
        uv_units: (P?.auto_post?.uv_units||1),
        support_plat: workerCount,
        support_part: workerCount,
        finish: workerCount,
        paint: workerCount
      };
      const counts = Object.assign({}, defaults, window.__ANIM_CFG__ || {});
      const stackGap = 8;
      const maxVisibleBase = 220;
      const manualKeys = new Set(['support_plat','support_part','finish','paint']);
      function drawStackFor(key, list, machineCount, anchor){
        if(!list || !list.length) return;
        const machineTotal = Math.max(1, machineCount);
        const enriched = list.map(t=>{
          const seg = t.segs.find(s=> (s.t0<=simNow && simNow<=s.t1));
          return {t, arr:(seg? seg.t0:0)};
        }).sort((a,b)=>a.arr-b.arr);
        let lanePositions = [];
        if(manualKeys.has(key)){
          const rows = machineTotal > 5 ? 2 : 1;
          const spacingX = 28;
          const spacingY = 26;
          const perRow = Math.ceil(machineTotal / rows);
          let idx = 0;
          for(let row=0; row<rows; row++){
            const remaining = machineTotal - row*perRow;
            const itemsThisRow = Math.min(perRow, Math.max(0, remaining));
            if(itemsThisRow<=0) break;
            const startX = anchor[0] - ((itemsThisRow-1) * spacingX)/2;
            const rowY = anchor[1] + (row - (rows-1)/2) * spacingY;
            for(let colIdx=0; colIdx<itemsThisRow; colIdx++){
              lanePositions[idx++] = {x:startX + colIdx*spacingX, y:rowY};
            }
          }
        }else{
          const spacing = 32;
          const startY = anchor[1] - (machineTotal-1)*spacing/2;
          for(let i=0;i<machineTotal;i++){
            lanePositions.push({x:anchor[0], y:startY + i*spacing});
          }
        }
        if(lanePositions.length === 0){
          lanePositions = [{x:anchor[0], y:anchor[1]}];
        }
        const laneCount = lanePositions.length;
        const lanes = Array.from({length:laneCount}, ()=>[]);
        enriched.forEach((e)=>{
          let minIdx = 0; for(let i=1;i<laneCount;i++){ if(lanes[i].length < lanes[minIdx].length) minIdx=i; }
          lanes[minIdx].push(e.t);
        });
        const machineFill = isWork ? theme.canvasStation : (theme.canvasStationNight || theme.canvasStation);
        for(let i=0;i<laneCount;i++){
          const center = lanePositions[i];
          const isManual = manualKeys.has(key);
          const dim = stageDim[key] || {w:44,h:20};
          const boxW = isManual ? Math.max(34, dim.w * 0.4) : Math.max(36, dim.w * 0.5);
          const boxH = isManual ? Math.max(16, dim.h * 0.5) : Math.max(18, dim.h * 0.7);
        ctx.fillStyle = machineFill;
        ctx.fillRect(center.x - boxW/2, center.y - boxH/2, boxW, boxH);
        ctx.strokeStyle = '#000';
        ctx.lineWidth = isManual ? 1 : 0.8;
        ctx.strokeRect(center.x - boxW/2, center.y - boxH/2, boxW, boxH);
        const isStorage = key === 'storage';
        const show = isStorage ? lanes[i] : lanes[i].slice(-maxVisibleBase);
        const perColumn = isStorage ? 50 : 20;
        const columnSpacing = isStorage ? 9 : (isManual ? 10 : 12);
        const columns = Math.max(1, Math.ceil(show.length / perColumn));
        const baseX = isStorage
          ? (pos.storage[0] - (stageDim.storage?.w || 90)/2 - 12)
          : center.x - (columns - 1) * columnSpacing * 0.5;
        const localStackGap = isStorage ? 6 : stackGap;
        const baseY = center.y - boxH/2 - (isStorage ? 10 : 8);
        show.forEach((t,idx)=>{
          const column = Math.floor(idx / perColumn);
          const row = idx % perColumn;
          const x = baseX + column*columnSpacing;
          const y = baseY - row*localStackGap;
          const radius = tokenRadius(t);
            ctx.save();
            ctx.globalAlpha = isStorage ? 0.75 : 0.82;
            ctx.beginPath(); ctx.fillStyle=t.color; ctx.arc(x,y,radius,0,Math.PI*2); ctx.fill();
            ctx.restore();
          });
          ctx.save();
          ctx.fillStyle=theme.canvasText;
          ctx.font=fontPx(10, '600');
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(`${lanes[i].length}`, center.x, center.y);
          ctx.restore();
        }
      }
      // 각 스테이션 처리
      drawStackFor('new_plat', atStation['new_plat']||[], counts.new_plat||1, pos.new_plat);
      drawStackFor('stacker',  atStation['stacker']||[],  counts.stacker||1,  pos.stacker);
      drawStackFor('printers', atStation['printers']||[], counts.printers, pos.printers);
      drawStackFor('wash1',    atStation['wash1']||[],    counts.wash_m1, pos.wash1);
      drawStackFor('wash2',    atStation['wash2']||[],    counts.wash_m2, pos.wash2);
      drawStackFor('dry',      atStation['dry']||[],      counts.dryers,  pos.dry);
      drawStackFor('uv',       atStation['uv']||[],       counts.uv_units,pos.uv);
      drawStackFor('support_plat',  atStation['support_plat']||[],  counts.support_plat || workerCount, pos.support_plat);
      drawStackFor('support_part',  atStation['support_part']||[],  counts.support_part || workerCount, pos.support_part);
      drawStackFor('finish',   atStation['finish']||[],   counts.finish || workerCount,  pos.finish);
      drawStackFor('paint',    atStation['paint']||[],    counts.paint || workerCount,   pos.paint);
      // storage (completed parts)
      drawStackFor('storage',  atStation['storage']||[],  1,              pos.storage);

      // Overlays: Support platform backlog and per-worker approx assignment, Storage count
      // platform-level backlog
      const stackerWaiting = (atStation['stacker']||[]).length;
      ctx.fillStyle = theme.text; ctx.font = fontPx(11, '600');
      ctx.fillText(`큐:${stackerWaiting}`, pos.stacker[0] + stageDim.stacker.w/2 + 10, pos.stacker[1]-6);

      const suppPlatList = atStation['support_plat']||[];
      const suppPlatSet = new Set();
      suppPlatList.forEach(t=>{ const b=t.id.includes('-P')? t.id.split('-P')[0] : t.id; suppPlatSet.add(b); });
      const suppPlatforms = suppPlatSet.size;
      ctx.fillStyle = theme.text; ctx.font = fontPx(11, '600');
      ctx.fillText(`PL:${suppPlatforms}`, pos.support_plat[0] + stageDim.support_plat.w/2 + 10, pos.support_plat[1]-6);
      // part-level approx per worker based on platform backlog and capacity
      const capPerPlat = Number((P.print && P.print.max_parts_per_platform) || 0) || 0;
      const approxParts = suppPlatforms * capPerPlat;
      const perWorker = Math.ceil(approxParts / Math.max(1, workerCount));
      ctx.fillStyle = theme.text; ctx.font = fontPx(11, '600');
      ctx.fillText(`~perW:${perWorker}`, pos.support_part[0] + stageDim.support_part.w/2 + 10, pos.support_part[1]-6);

      const stored = (atStation['storage']||[]).length;
      const targetStored = opts.totalParts ?? null;
      ctx.save();
      ctx.fillStyle = theme.text; ctx.font = fontPx(12, '600');
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      const label = targetStored != null && targetStored !== undefined ? `${stored}/${targetStored}` : `${stored}`;
      ctx.fillText(label, pos.storage[0], pos.storage[1] + stageDim.storage.h/2 + 6);
      ctx.restore();

      // loop playback when end reached
      if(simNow>simT1) simNow = simT0;
      requestAnimationFrame(draw);
    }
    draw();
  }
  function cycleBreakdown(P, mode){
    const parts = Number(P.print.max_parts_per_platform||0);
    // preproc
    const pre = Number(P.preproc.healing_time_per_platform_min||0) + Number(P.preproc.placement_time_per_platform_min||0) + Number(P.preproc.support_time_per_platform_min||0);
    const print = Number(P.print.t_print_per_platform_min||0);
    let post = 0;
    if(mode==='automated'){
      const ap=P.auto_post; const amr=(d)=> (Number(d||0)/Number(ap.amr_speed_m_per_s||1))/60 + Number(ap.amr_load_min||0) + Number(ap.amr_unload_min||0);
      const clean = amr(ap.dist_m.support_to_platform_wash) + Number((P.platform_clean?.wash_time_min)||0) + amr(ap.dist_m.platform_wash_to_new);
      post = amr(ap.dist_m.printer_to_wash) + Number(ap.t_wash1_min||0) + amr(ap.dist_m.wash_to_dry) + Number(ap.t_wash2_min||0) + Number(ap.t_dry_min||0) + amr(ap.dist_m.dry_to_uv) + Number(ap.t_uv_min||0) + amr(ap.dist_m.uv_to_stacker) + clean;
    }else{
      const mp=P.manual_post; const ap=P.auto_post; const mm=P.manual_move||{};
      const speed = Math.max(Number(mm.speed_m_per_s||2.0), 1e-6);
      const distMap = mm.dist_m||{};
      const fallbackKey = r => ({wash1_to_wash2:'wash_to_dry'}[r]||r);
      const travel = (route, override)=>{
        if(override !== undefined && override !== null && Number(override) > 0){
          return Number(override);
        }
        const distRaw = distMap[route] ?? (ap.dist_m?.[fallbackKey(route)] ?? 0);
        const dist = Number(distRaw||0);
        if(dist<=0) return 0;
        return (dist / speed) / 60;
      };
      const clean = travel('support_to_platform_wash', mp.to_platform_wash_travel_min)
        + Number((P.platform_clean?.wash_time_min)||0)
        + travel('platform_wash_to_new', mp.to_newplatform_travel_min);
      post = Number(mp.human_unload_min||0)
        + travel('printer_to_wash', mp.to_washer_travel_min)
        + Number(ap.t_wash1_min||0)
        + travel('wash1_to_wash2', mp.to_wash2_travel_min)
        + Number(ap.t_wash2_min||0)
        + travel('wash_to_dry', mp.to_dryer_travel_min)
        + Number(ap.t_dry_min||0)
        + travel('uv_to_stacker', mp.to_stacker_travel_min)
        + Number(ap.t_uv_min||0)
        + clean;
    }
    const ops = Number(P.manual_ops.support_time_per_platform_min||0)
      + Number(P.manual_ops.move_platform_to_support_min||0)
      + Number(P.manual_ops.support_time_per_part_min||0)*parts
      + (Number(P.manual_ops.move_support_to_finish_min||0) * parts)
      + Number(P.manual_ops.finish_time_per_part_min||0)*parts
      + (Number(P.manual_ops.move_finish_to_paint_min||0) * parts)
      + Number(P.manual_ops.paint_time_per_part_min||0)*parts
      + (Number(P.manual_ops.move_paint_to_storage_min||0) * parts);
    const total = pre+print+post+ops;
    return {pre, print, post, ops, total};
  }
  function cycleSvg(b){
    const W=860,H=120,unit= (W-180)/Math.max(1, Number(b.total||0)); let x=80;
    function seg(w,color,label){
      const ww=Math.max(4,w*unit);
      const g=`<g><rect x="${x}" y="36" width="${ww}" height="28" fill="${color}" rx="8"/>
      <text x="${x+8}" y="56" font-size="14" fill="var(--cycle-text)" font-weight="600">${label}</text></g>`;
      x+=ww+10;
      return g;
    }
    const svg = seg(b.pre,'var(--cycle-pre)',`전처리 ${b.pre.toFixed(0)}m`) + seg(b.print,'var(--cycle-print)',`프린팅 ${b.print.toFixed(0)}m`) + seg(b.post,'var(--cycle-post)',`후처리 ${b.post.toFixed(0)}m`) + seg(b.ops,'var(--cycle-ops)',`수작업 ${b.ops.toFixed(0)}m`);
    const ttl = (Number(b.total||0)/60).toFixed(2);
    return `<div style="overflow-x:auto"><svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}"><text x="0" y="24" font-size="14" fill="var(--muted)">총 ${ttl} 시간/빌드플랫폼</text>${svg}</svg></div>`;
  }
  const ba = cycleBreakdown(P,'automated');
  const bm = cycleBreakdown(P,'manual');
  if(!m){
    document.getElementById('out').innerHTML = `
      <div class="card"><h3>결과(자동화)</h3>
        ${tbl(a)}
        ${utilTbl(utilA)}
        ${costTbl(a.cost)}
        <h4>자원별 대기시간</h4>
        ${waitTbl(a.wait_stats)}
        <h4>공정 레이아웃</h4>
        ${layoutA}
        <h4>한 사이클(빌드플랫폼 1개) 시간 분해</h4>
        ${cycleSvg(ba)}
        <h4>2D Animation</h4>
        <div id="viz2dA" style="height:520px"></div>
      </div>`;
    window.__ANIM_CFG__ = {
      new_plat: 1,
      stacker: 1,
      printers: (P.print?.printer_count || P.print?.printers?.length || 2),
      wash_m1:(P.auto_post?.washers_m1||1),
      wash_m2:(P.auto_post?.washers_m2||1),
      platform_wash:(P.platform_clean?.washers||1),
      dryers:(P.auto_post?.dryers||1),
      uv_units:(P.auto_post?.uv_units||1),
      support_plat:(P.manual_ops?.workers||2),
      support_part:(P.manual_ops?.workers||2),
      finish:(P.manual_ops?.workers||2),
      paint:(P.manual_ops?.workers||2)
    };
    renderFactory2D('viz2dA', a.trace, { totalParts: a.completed_parts, automated: true, horizonMinutes: a.horizon_minutes });
  } else {
    document.getElementById('out').innerHTML = `
      <div class="card"><h3>결과 비교: 자동화 vs 비자동화</h3>
        <div class="grid">
          <div>
            <h4>자동화</h4>
            ${tbl(a)}
            ${utilTbl(utilA)}
            ${costTbl(a.cost)}
            <h4>자원별 대기시간</h4>
            ${waitTbl(a.wait_stats)}
            <h4>공정 레이아웃</h4>
            ${layoutA}
            <h4>한 사이클(빌드플랫폼 1개) 시간 분해</h4>
            ${cycleSvg(ba)}
          </div>
          <div>
            <h4>비자동화</h4>
            ${tbl(m)}
            ${utilTbl(utilM)}
            ${costTbl(m.cost)}
            <h4>자원별 대기시간</h4>
            ${waitTbl(m.wait_stats)}
            <h4>공정 레이아웃</h4>
            ${layoutM}
            <h4>한 사이클(빌드플랫폼 1개) 시간 분해</h4>
            ${cycleSvg(bm)}
          </div>
        </div>
        <h4 style="margin-top:12px">2D Animation</h4>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
          <label><input type="radio" name="anim_mode" value="automated" checked> 자동화</label>
          <label><input type="radio" name="anim_mode" value="manual"> 비자동화</label>
        </div>
        <div id="viz2d" style="height:520px; overflow:hidden; border-radius:14px;"></div>
        </div>
      ${simpyExtraHtml(data.results.simpy)}
      </div>`;
    window.__ANIM_CFG__ = {
      new_plat: 1,
      stacker: 1,
      printers: (P.print?.printer_count || P.print?.printers?.length || 2),
      wash_m1:(P.auto_post?.washers_m1||1),
      wash_m2:(P.auto_post?.washers_m2||1),
      platform_wash:(P.platform_clean?.washers||1),
      dryers:(P.auto_post?.dryers||1),
      uv_units:(P.auto_post?.uv_units||1),
      support_plat:(P.manual_ops?.workers||2),
      support_part:(P.manual_ops?.workers||2),
      finish:(P.manual_ops?.workers||2),
      paint:(P.manual_ops?.workers||2)
    };
    // default to automated animation
    renderFactory2D('viz2d', a.trace, { totalParts: a.completed_parts, automated: true, horizonMinutes: a.horizon_minutes });
    // toggle between automated/manual traces
    document.querySelectorAll('input[name="anim_mode"]').forEach(el=>{
      el.addEventListener('change', (e)=>{
        const mode = e.target.value;
        const trace = mode==='manual'? m.trace : a.trace;
        // rerender in same container
        const info = mode==='manual'? m : a;
        renderFactory2D('viz2d', trace, { totalParts: info.completed_parts, automated: (mode==='automated'), horizonMinutes: info.horizon_minutes });
      });
    });
  }
}
</script>
</div>
</body></html>
    """
    return HTMLResponse(html)

@app.post("/simulate")
def simulate(req: SimRequest):
    try:
        # automated scenario
        P_auto = deepcopy(DEFAULT)
        if req.params:
            deep_merge(P_auto, req.params)
        P_auto["flow_mode"] = "automated"
        normalize_printer_count(P_auto)
        apply_global_printer_defaults(P_auto)
        out_a = run_sim(P_auto, seed=req.seed)
        cost_a = compute_costs(P_auto, KPI(**{  # reconstruct KPI-like fields from results
            'completed_parts': out_a['completed_parts'],
            'scrapped_parts': out_a['scrapped_parts'],
            'completed_platforms': out_a.get('completed_platforms',0),
            'printer_busy_min': 0.0,
            'wash1_busy_min': 0.0,
            'wash2_busy_min': 0.0,
            'dry_busy_min': 0.0,
            'uv_busy_min': 0.0,
            'amr_travel_min': 0.0,
            'resin_used_kg': out_a['resin_used_kg'],
            'resin_cost_krw': out_a['resin_cost_krw'],
            'preproc_busy_min': 0.0,
            'manual_busy_min': 0.0,
            'sum_platform_lead_time_min': 0.0,
            'finished_platforms': 0,
            'max_stacker_wip': 0
        }), out_a['horizon_minutes'], out_a.get('utilization', {}))
        out_a["cost"] = cost_a

        # manual scenario
        P_man = deepcopy(DEFAULT)
        if req.params:
            deep_merge(P_man, req.params)
        P_man["flow_mode"] = "manual"
        normalize_printer_count(P_man)
        apply_global_printer_defaults(P_man)
        out_m = run_sim(P_man, seed=req.seed)
        cost_m = compute_costs(P_man, KPI(**{
            'completed_parts': out_m['completed_parts'],
            'scrapped_parts': out_m['scrapped_parts'],
            'completed_platforms': out_m.get('completed_platforms',0),
            'printer_busy_min': 0.0,
            'wash1_busy_min': 0.0,
            'wash2_busy_min': 0.0,
            'dry_busy_min': 0.0,
            'uv_busy_min': 0.0,
            'amr_travel_min': 0.0,
            'resin_used_kg': out_m['resin_used_kg'],
            'resin_cost_krw': out_m['resin_cost_krw'],
            'preproc_busy_min': 0.0,
            'manual_busy_min': 0.0,
            'sum_platform_lead_time_min': 0.0,
            'finished_platforms': 0,
            'max_stacker_wip': 0
        }), out_m['horizon_minutes'], out_m.get('utilization', {}))
        out_m["cost"] = cost_m


        # ---------------------------------------------------------
        # (추가) SimPy 코드베이스(main_SimPy.py)에서 계산되는 확장 KPI/파라미터를
        # 기존 웹 뼈대(des_factory_web.py)의 결과(JSON)에 "simpy"로 덧붙입니다.
        # - 기존 자동화/비자동화 DES 결과(out_a/out_m) 및 2D 애니메이션 trace는 그대로 유지
        # - 실패 시에도 웹은 정상 동작하도록 try/except로 감쌉니다.
        # ---------------------------------------------------------
        simpy_result = None
        try:
            from config_SimPy import apply_web_overrides_to_cfg
            from main_SimPy import run_full_simulation

            # des_factory_web 파라미터(DEFAULT 스키마) -> config_SimPy.apply_web_overrides_to_cfg 스키마로 매핑
            P_base = deepcopy(DEFAULT)
            if req.params:
                deep_merge(P_base, req.params)

            # demand/material/printing/preprocess/... 형태로 변환(가능한 항목만)
            web_cfg = {
                "demand": {
                    # platform interarrival(분) → order_cycle_min(분)로 근사
                    "order_cycle_min": (
                        (P_base.get("demand", {}).get("platform_interarrival_min_min", 30)
                         + P_base.get("demand", {}).get("platform_interarrival_max_min", 30)) / 2.0
                    ),
                    # 환자/아이템 개수는 이 UI에 없으므로 None으로 둠
                    "patients_min": None,
                    "patients_max": None,
                    "items_min": None,
                    "items_max": None,
                },
                "material": {
                    "part_weight_g": (P_base.get("material", {}).get("avg_part_mass_kg", 0.0) or 0.0) * 1000.0,
                    "resin_cost_per_kg": P_base.get("material", {}).get("resin_price_per_kg"),
                    "pallet_size_limit": P_base.get("material", {}).get("max_parts_per_platform"),
                    "initial_platforms": P_base.get("demand", {}).get("initial_platforms"),
                },
                "printing": {
                    "printer_count": P_base.get("print", {}).get("printer_count"),
                    "print_time_min": P_base.get("print", {}).get("t_print_per_platform_min"),
                    "defect_rate": P_base.get("print", {}).get("defect_rate"),
                    # (고장/정비 파라미터는 UI에 없으므로 None)
                    "breakdown_enabled": None,
                    "mtbf_min": None,
                    "mttr_min": None,
                    "maintenance_cycle_h": None,
                    "maintenance_duration_min": None,
                },
                "preprocess": {
                    "healing_time": P_base.get("preproc", {}).get("healing_time_per_platform_min"),
                    "placement_time": P_base.get("preproc", {}).get("placement_time_per_platform_min"),
                    "support_time": P_base.get("preproc", {}).get("support_time_per_platform_min"),
                },
                "platform_clean": {
                    "washer_count": P_base.get("platform_clean", {}).get("washers"),
                    "platform_clean_time": P_base.get("platform_clean", {}).get("wash_time_min"),
                },
                "auto_post_common": {
                    "wash1_count": P_base.get("auto_post", {}).get("washers_m1"),
                    "wash1_time": P_base.get("auto_post", {}).get("t_wash1_min"),
                    "wash2_count": P_base.get("auto_post", {}).get("washers_m2"),
                    "wash2_time": P_base.get("auto_post", {}).get("t_wash2_min"),
                    "dry_count": P_base.get("auto_post", {}).get("dryers"),
                    "dry_time": P_base.get("auto_post", {}).get("t_dry_min"),
                    "uv_count": P_base.get("auto_post", {}).get("uv_units"),
                    "uv_time": P_base.get("auto_post", {}).get("t_uv_min"),
                    "defect_wash1": P_base.get("auto_post", {}).get("defect_rate_wash"),
                    "defect_dry": P_base.get("auto_post", {}).get("defect_rate_dry"),
                    "defect_uv": P_base.get("auto_post", {}).get("defect_rate_uv"),
                },
                "auto_post_amr": {
                    "amr_count": P_base.get("auto_post", {}).get("amr_count"),
                    "amr_speed": P_base.get("auto_post", {}).get("amr_speed_m_per_s"),
                    "load_time": P_base.get("auto_post", {}).get("amr_load_min"),
                    "unload_time": P_base.get("auto_post", {}).get("amr_unload_min"),
                    # 거리 키 매핑(있으면)
                    "dist_printer_w1": (P_base.get("auto_post", {}).get("dist_m", {}) or {}).get("printer_to_wash"),
                    "dist_w1_w2": (P_base.get("auto_post", {}).get("dist_m", {}) or {}).get("printer_to_wash") or (P_base.get("auto_post", {}).get("dist_m", {}) or {}).get("wash_to_dry"),
                    "dist_w2_dry": (P_base.get("auto_post", {}).get("dist_m", {}) or {}).get("wash_to_dry"),
                    "dist_dry_uv": (P_base.get("auto_post", {}).get("dist_m", {}) or {}).get("dry_to_uv"),
                },
                "manual_transport": {
                    "mover_count": P_base.get("manual_move", {}).get("workers"),
                    "mover_speed": P_base.get("manual_move", {}).get("speed_m_per_s"),
                    "travel_printer_wash": (P_base.get("manual_move", {}).get("dist_m", {}) or {}).get("printer_to_wash"),
                    "travel_wash1_wash2": (P_base.get("manual_move", {}).get("dist_m", {}) or {}).get("wash1_to_wash2"),
                    "travel_wash2_dry": (P_base.get("manual_move", {}).get("dist_m", {}) or {}).get("wash_to_dry"),
                    "travel_dry_uv": (P_base.get("manual_move", {}).get("dist_m", {}) or {}).get("dry_to_uv"),
                    "shift_start": (P_base.get("manual_move", {}).get("work_shift", {}) or {}).get("start_hhmm"),
                    "shift_end": (P_base.get("manual_move", {}).get("work_shift", {}) or {}).get("end_hhmm"),
                    "workdays_per_week": (P_base.get("manual_move", {}).get("work_shift", {}) or {}).get("workdays_per_cycle"),
                },
                "manual_post": {
                    "human_unload_min": P_base.get("manual_post", {}).get("human_unload_min"),
                    "wash1_time": P_base.get("manual_post", {}).get("t_wash1_min"),
                    "wash2_time": P_base.get("manual_post", {}).get("t_wash2_min"),
                    "dry_time": P_base.get("manual_post", {}).get("t_dry_min"),
                    "uv_time": P_base.get("manual_post", {}).get("t_uv_min"),
                    "shift_start": (P_base.get("manual_ops", {}).get("work_shift", {}) or {}).get("start_hhmm"),
                    "shift_end": (P_base.get("manual_ops", {}).get("work_shift", {}) or {}).get("end_hhmm"),
                    "workdays_per_week": (P_base.get("manual_ops", {}).get("work_shift", {}) or {}).get("workdays_per_cycle"),
                    "workers": P_base.get("manual_ops", {}).get("workers"),
                },
                "stacker": {
                    "enabled": P_base.get("stacker_guard", {}).get("enabled"),
                    "capacity": P_base.get("stacker_guard", {}).get("max_platforms"),
                },
                "cost": {
                    "wage_per_hour": (P_base.get("cost", {}) or {}).get("wage_per_hour_krw"),
                    "overhead_per_month": (P_base.get("cost", {}) or {}).get("overhead_krw_per_month"),
                    "depreciation_years": (P_base.get("cost", {}) or {}).get("depreciation_years"),
                    "equipment": (P_base.get("cost", {}) or {}).get("equipment", {}),
                },
            }

            apply_web_overrides_to_cfg(web_cfg)

            # des_factory_web의 horizon(분)에 맞춰 SimPy도 동일 horizon으로 실행
            sim_duration = int(out_a.get("horizon_minutes") or 0) or 60
            simpy_result = run_full_simulation(sim_duration=sim_duration, show_gantt=False)

        except Exception:
            simpy_result = None
        return JSONResponse({"ok": True, "results": {"automated": out_a, "manual": out_m, "simpy": simpy_result}})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
