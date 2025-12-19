import simpy
import random
from typing import Any, Dict
from specialized_Processor import Printer, WashM1, WashM2, Dryer, UVStation, Worker, PlatformWasher, AMR
from flow_utils import sample_stage_defects, build_stacker_payload, manual_travel_time_min
from platform_manager import PlatformManager
from KPI import KPI
from base_Job import Job


# Common utility
def _allocate_slot(slot_usage: Dict[int, list], capacity: int, start: float, end: float) -> int:
    """
    slot_usage: {slot_idx: [(s,e), ...]}
    capacity  : number of units (e.g. 3 printers -> capacity=3)
    start,end : start/end time for this interval

    Returns:
        The allocated slot index (1..capacity).
    """
    for slot in range(1, max(1, capacity) + 1):
        intervals = slot_usage.get(slot, [])
        conflict = False
        for s, e in intervals:
            # If [s,e] intersects [start,end], this slot is not free.
            if not (end <= s or start >= e):
                conflict = True
                break
        if not conflict:
            slot_usage.setdefault(slot, []).append((start, end))
            return slot

    # If all slots are overlapping, just stack it into slot 1.
    # In theory, since SimPy resource capacity is not exceeded, this should rarely happen.
    slot_usage.setdefault(1, []).append((start, end))
    return 1



# 1. Preprocess
class Preprocess:
    def __init__(self, env: simpy.Environment, cfg: Dict[str, Any],
                 platM: PlatformManager, kpi: KPI, logger=None):
        self.env = env
        self.cfg = cfg["preproc"]
        self.kpi = kpi
        self.logger = logger
        self.platM = platM
        self.server = simpy.Resource(env, capacity=self.cfg["servers"])

    def run(self, job: Job):
        """Assign platform + basic prep timing."""
        with self.server.request() as req:
            yield req

            # Get an available platform from PlatformManager
            token = yield self.platM.get_clean_platform()

            # Map Job to Platform
            platform_id = self.platM.assign(job.id_job, token)
            job.platform_id = platform_id
            job.start_time = self.env.now
            self.kpi.started_platforms += 1
            self.kpi.n_preproc_jobs += 1

            #Preprocess timing
            t = (
                self.cfg["healing_time_per_platform_min"]
                + self.cfg["placement_time_per_platform_min"]
                + self.cfg["support_generation_time_min"]
            )
            self.kpi.preproc_busy_min += t

            if self.logger:
                self.logger.log_event(
                    "PREPROC",
                    f"[START] Job {job.id_job} | Platform {platform_id} | "
                    f"preprocessing will take {t} min"
                )

            yield self.env.timeout(t)

            if self.logger:
                self.logger.log_event(
                    "PREPROC",
                    (
                        f"[END] Job {job.id_job} ← Platform {platform_id} | "
                        f"preprocessing_total_time: {t} min "
                        # f"(healing={self.cfg['healing_time_per_platform_min']} min, "
                        # f"placement={self.cfg['placement_time_per_platform_min']} min, "
                        # f"support_gen={self.cfg['support_generation_time_min']} min)"
                    ),
                )

            return token



# 2. Print
class Print:
    def __init__(self, env, cfg, kpi: KPI, logger=None):
        self.env = env
        self.kpi = kpi
        self.logger = logger

        self.proc = Printer(env, cfg["printer_count"], cfg["t_print_per_platform_min"])
        self._slot_idx = 0

        # Slot usage records for Gantt (Printer_1, Printer_2, ...)
        self._printer_slots: Dict[int, list] = {}

        self.defect_rate = float(cfg.get("defect_rate"))

        # Breakdown / Maintenance config
        bd_cfg = cfg.get("breakdown")
        mt_cfg = cfg.get("maintenance")

        self.breakdown_enabled = bool(bd_cfg.get("enabled"))
        self.mtbf_min = float(bd_cfg.get("mtbf_min"))
        self.mttr_min = float(bd_cfg.get("mttr_min"))

        self.maint_enabled = bool(mt_cfg.get("enabled"))
        self.maint_start_hhmm = mt_cfg.get("start_hhmm")
        self.maint_duration_min = float(mt_cfg.get("duration_min"))
        self.maint_cycle_days = int(mt_cfg.get("cycle_days"))

        # State flags
        self._is_breakdown = False
        self._is_maintenance = False

        # Background processes
        if self.breakdown_enabled and self.mtbf_min > 0 and self.mttr_min > 0:
            self.env.process(self._random_breakdown_process())

        if self.maint_enabled and self.maint_duration_min > 0:
            self.env.process(self._maintenance_process())

    def _is_down(self) -> bool:
        """Return True if printer bank is down due to breakdown or maintenance."""
        return self._is_breakdown or self._is_maintenance

    def _random_breakdown_process(self):
        """
        Simple random breakdown model for the whole printer bank:
        - Time to failure ~ Exp(MTBF) [minutes]
        - Once breakdown occurs, downtime lasts MTTR minutes
        - While down, no new print jobs can start (jobs wait in queue).
          (In-progress jobs are assumed to complete normally for simplicity.)
        """
        while True:
            # Wait until next breakdown
            if self.mtbf_min <= 0:
                return
            ttf = random.expovariate(1.0 / self.mtbf_min) # time to failure
            yield self.env.timeout(ttf)

            # Breakdown start
            self._is_breakdown = True

            ## Debugging
            """
            if self.logger:
                self.logger.log_event(
                    "PRINT",
                    f"BREAKDOWN start (MTTR={self.mttr_min:.1f} min)")
            """

            # Repair time
            yield self.env.timeout(self.mttr_min)

            # Breakdown end
            self._is_breakdown = False
            
            ## Debugging
            """
            if self.logger:
                self.logger.log_event(
                    "PRINT",
                    "BREAKDOWN end (printer back to service)")
            """

    def _hhmm_to_min(self, hhmm: str) -> int:
        """Convert 'HH:MM' string to minutes since day start."""
        try:
            h, m = hhmm.split(":")
            return int(h) * 60 + int(m)
        except Exception:
            return 0

    def _maintenance_process(self):
        """
        Periodic maintenance model:
        - At start_hhmm on every cycle_days, printer bank goes down
        - Stays down for duration_min minutes
        """
        day_min = 24 * 60
        start_offset = self._hhmm_to_min(self.maint_start_hhmm)
        cycle_days = max(1, self.maint_cycle_days)

        while True:
            now = self.env.now
            # Compute the next maintenance start time relative to now
            current_cycle = int(now // (cycle_days * day_min))
            next_start = current_cycle * cycle_days * day_min + start_offset

            if next_start <= now:
                # If already passed, move to next cycle
                next_start += cycle_days * day_min

            wait = max(0.0, next_start - now)
            if wait > 0:
                yield self.env.timeout(wait)

            # Maintenance start
            self._is_maintenance = True

            ## Debugging
            """
            if self.logger:
                self.logger.log_event(
                    "PRINT",
                    f"MAINTENANCE start at t={self.env.now:.1f} "
                    f"(duration={self.maint_duration_min:.1f} min)"
                )
            """

            # Maintenance duration
            yield self.env.timeout(self.maint_duration_min)

            # Maintenance end
            self._is_maintenance = False

            ## Debugging
            """
            if self.logger:
                self.logger.log_event( 
                            "PRINT",
                            f"MAINTENANCE end at t={self.env.now:.1f}"
                            )
            """

    def _record_trace(self, t0: float, t1: float, job: Job):
        """Record printer usage interval into logger.trace_events."""
        if not self.logger or not hasattr(self.logger, "trace_events"):
            return

        cap = getattr(self.proc.resource, "capacity", 1)
        slot_idx = _allocate_slot(self._printer_slots, cap, t0, t1)
        res_name = f"Printer_{slot_idx}"

        self.logger.trace_events.append(
            {
                "id": f"Job {job.id_job}",
                "job_id": job.id_job,
                "platform_id": getattr(job, "platform_id", None),
                "stage": "Printer",
                "Resource": res_name,
                "t0": float(t0),
                "t1": float(t1),
            }
        )

    def run(self, job: Job):
        # If printer is currently down, wait until it becomes available

        ## Debugging
        platform_id = getattr(job, "platform_id", f"JOB-{job.id_job}")
        n_parts = len(job.list_items)

        """
        if self.logger and self._is_down():
            self.logger.log_event(
                "PRINT",
                f"Job {job.id_job} waiting: printer DOWN (breakdown/maintenance)"
            )
        """
        while self._is_down():
            # Poll every 10 minutes
            yield self.env.timeout(10.0)

        # Request printer resource
        with self.proc.resource.request() as req:
            yield req

            # If printer goes down after acquiring the resource, wait again
            while self._is_down():
                if self.logger:
                    self.logger.log_event(
                        "PRINT",
                        f"Job {job.id_job} got printer but waiting until DOWN state ends"
                    )
                yield self.env.timeout(10.0)

            # printing
            t = self.proc.proc_time
            start = self.env.now
            self.kpi.printer_busy_min += t
            self.kpi.n_print_jobs += 1

            ## Debugging
            if self.logger:
                self.logger.log_event(
                    "PRINT",
                    f"[START] Job {job.id_job} | Platform {getattr(job, 'platform_id', None)} | "
                    f"print_time={t:.1f} min"
                )
        
            yield self.env.timeout(t)
            end = self.env.now

            if self.logger:
                self.logger.log_event(
                    "PRINT",
                    f"[END] Job {job.id_job} | Platform {getattr(job, 'platform_id', None)} | "
                    f"print_finished (elapsed={end-start:.1f} min)"
                )

            scrapped = 0

            if n_parts > 0 and self.defect_rate > 0.0:
                # Factory에서처럼 Bernoulli(플랫폼 전체)로 판단
                if random.random() < self.defect_rate:
                    # 이 플랫폼에 올라간 모든 부품 스크랩
                    scrapped = n_parts
                    self.kpi.scrapped_parts += scrapped

                    job.is_scrapped_job = True
                    job.scrap_stage = "print"

                    # Job에 상태 플래그/정보 기록
                    job.is_scrapped_in_print = True      # 플랫폼 전체 불량
                    job.print_n_parts = n_parts          # 프린트에 올라갔던 부품 수
                    job.print_good_parts = 0             # 프린트 이후 양품 0
                    job.scrapped_in_print = scrapped     # 프린트에서 스크랩된 개수

                    if hasattr(job, "list_items") and job.list_items is not None:
                        job.list_items = []

                    if self.logger:
                        self.logger.log_event(
                            "PRINT",
                            (
                                f"[SCRAP] Job {job.id_job} | Platform {platform_id} | "
                                f"defect_rate={self.defect_rate:.4f}, "
                                f"n_parts={n_parts}, scrapped_in_print={scrapped}"
                            ),
                        )
                else:
                    # 불량 안 난 정상 플랫폼
                    job.is_scrapped_in_print = False
                    job.print_n_parts = n_parts
                    job.print_good_parts = n_parts
                    job.scrapped_in_print = 0
            else:
                # 부품 수가 0이거나 defect_rate=0인 경우 기본값만 세팅
                job.is_scrapped_in_print = False
                job.print_n_parts = n_parts
                job.print_good_parts = n_parts
                job.scrapped_in_print = 0

            # Gantt trace
            if self.logger and hasattr(self.logger, "trace_events"):
                cap = getattr(self.proc.resource, "capacity", 1)
                self._slot_idx = (self._slot_idx % cap) + 1
                res_name = f"Printer_{self._slot_idx}"

                rec = {
                    "id": f"Job {job.id_job}",
                    "job_id": job.id_job,
                    "platform_id": getattr(job, "platform_id", None),
                    "stage": "Printer",
                    "Resource": res_name,
                    "t0": start,
                    "t1": end,
                }
                self.logger.trace_events.append(rec)

                # print("[TRACE DEBUG] Print.run appended:", rec)
            else:
                """
                print("[TRACE DEBUG] Print.run: logger or trace_events missing",
                      self.logger, hasattr(self.logger, "trace_events"))
                """


# 3. PostProcessLine (WashM1, WashM2, Dryer, UV, AMRPool OR Human Movers)
class PostProcessLine:
    """
    Post-processing line (des_factory_web style):

    - flow_mode == "automated": AMR moves between stations (24/7)
    - flow_mode == "manual":    Human movers carry platforms between stations (SHIFT-GATED for 이동/언로드)
    - Machines (Wash1/Wash2/Dry/UV) run continuously once started (no shift gating on processing)

    - Defect policy: stage-level Bernoulli => FULL scrap & STOP immediately
      (fail gates at Wash1 / Dry / UV only; Wash2 has no defect gate)
    - On success: push payload to stacker so ManualOps can process support/finish/paint
    """

    def __init__(self, env, cfg, kpi: "KPI", logger=None, amr_pool: "AMR" = None, stacker=None, shift_move=None):

        self.env = env
        self.kpi = kpi
        self.logger = logger
        self.stacker = stacker

        # flow_mode 선택
        # - cfg["flow_mode"] in {"automated","manual"}
        self.flow_mode = str(cfg.get("flow_mode", "automated")).lower().strip()
        if self.flow_mode not in ("automated", "manual"):
            self.flow_mode = "automated"

        # auto_post = 공통 설비/AMR 파라미터 (automated 모드)
        self.cfg = cfg["auto_post"]

        # manual_move / manual_post = manual 모드 이동/언로드 시간 파라미터
        self.manual_move_cfg = cfg.get("manual_move", {})
        self.manual_post_cfg = cfg.get("manual_post", {})

        # Machines (same)
        self.m1 = WashM1(env, self.cfg["washers_m1"], self.cfg["t_wash1_min"])
        self.m2 = WashM2(env, self.cfg["washers_m2"], self.cfg["t_wash2_min"])
        self.dry = Dryer(env, self.cfg["dryers"], self.cfg["t_dry_min"])
        self.uv = UVStation(env, self.cfg["uv_units"], self.cfg["t_uv_min"])

        # Slot usage for Gantt
        self._slots_m1: Dict[int, list] = {}
        self._slots_m2: Dict[int, list] = {}
        self._slots_dry: Dict[int, list] = {}
        self._slots_uv: Dict[int, list] = {}
        self._slots_amr: Dict[int, list] = {}
        self._slots_mover: Dict[int, list] = {}  # NEW (manual mover)

        # AMR
        self.amr_pool = amr_pool

        # AMR parameters
        self.dist = self.cfg["dist_m"]
        self.speed_m_per_s = self.cfg["amr_speed_m_per_s"]
        self.load_min = self.cfg["amr_load_min"]
        self.unload_min = self.cfg["amr_unload_min"]

        # Defect rate per stage (des_factory_web style full-scrap gates)
        self.def_wash = self.cfg.get("defect_rate_wash")
        self.def_dry = self.cfg.get("defect_rate_dry")
        self.def_uv = self.cfg.get("defect_rate_uv")

        # Stacker guard config
        guard_cfg = cfg.get("stacker_guard", {})
        self.stacker_guard_enabled = bool(guard_cfg.get("enabled", False))
        self.stacker_guard_limit = int(guard_cfg.get("max_platforms", 0) or 0)

        # manual mover resource + shift
        self.shift_move = shift_move

        # manual movers (capacity = manual_move["workers"])
        movers_cap = int(self.manual_move_cfg.get("workers", 0) or 0)
        self.movers = Worker(env, movers_cap if movers_cap > 0 else 1, 0)  # fallback 1

        self.manual_speed_m_per_s = float(self.manual_move_cfg.get("speed_m_per_s", 0.1) or 0.1)
        self.manual_dist = self.manual_move_cfg.get("dist_m", {})

        # manual_post 언로드/이동 override(분)
        self.human_unload_min = float(self.manual_post_cfg.get("human_unload_min", 0.0) or 0.0)
        self.to_washer_travel_min = self.manual_post_cfg.get("to_washer_travel_min")
        self.to_wash2_travel_min = self.manual_post_cfg.get("to_wash2_travel_min")
        self.to_dryer_travel_min = self.manual_post_cfg.get("to_dryer_travel_min")
        self.to_stacker_travel_min = self.manual_post_cfg.get("to_stacker_travel_min")

    # Trace helpers
    def _record_trace_stage(self, stage_name: str, t0: float, t1: float, job: "Job"):
        if not self.logger or not hasattr(self.logger, "trace_events"):
            return

        if stage_name == "WashM1":
            slots = self._slots_m1
            cap = getattr(self.m1.resource, "capacity", 1)
        elif stage_name == "WashM2":
            slots = self._slots_m2
            cap = getattr(self.m2.resource, "capacity", 1)
        elif stage_name == "Dryer":
            slots = self._slots_dry
            cap = getattr(self.dry.resource, "capacity", 1)
        elif stage_name == "UV":
            slots = self._slots_uv
            cap = getattr(self.uv.resource, "capacity", 1)
        else:
            return

        slot_idx = _allocate_slot(slots, cap, t0, t1)
        res_name = f"{stage_name}_{slot_idx}"

        self.logger.trace_events.append(
            {
                "id": f"Job {job.id_job}",
                "job_id": job.id_job,
                "platform_id": getattr(job, "platform_id", None),
                "stage": stage_name,
                "Resource": res_name,
                "t0": float(t0),
                "t1": float(t1),
            }
        )

    def _record_trace_amr(self, t0: float, t1: float, job: "Job"):
        if not self.logger or not hasattr(self.logger, "trace_events"):
            return
        if self.amr_pool is None:
            return

        cap = getattr(self.amr_pool.resource, "capacity", 1)
        slot_idx = _allocate_slot(self._slots_amr, cap, t0, t1)
        res_name = f"AMRPool_{slot_idx}"

        self.logger.trace_events.append(
            {
                "id": f"Job {job.id_job}",
                "job_id": job.id_job,
                "platform_id": getattr(job, "platform_id", None),
                "stage": "AMRPool",
                "Resource": res_name,
                "t0": float(t0),
                "t1": float(t1),
            }
        )

    def _record_trace_mover(self, t0: float, t1: float, job: "Job"):
        if not self.logger or not hasattr(self.logger, "trace_events"):
            return

        cap = getattr(self.movers.resource, "capacity", 1)
        slot_idx = _allocate_slot(self._slots_mover, cap, t0, t1)
        res_name = f"ManualMover_{slot_idx}"

        self.logger.trace_events.append(
            {
                "id": f"Job {job.id_job}",
                "job_id": job.id_job,
                "platform_id": getattr(job, "platform_id", None),
                "stage": "ManualMove",
                "Resource": res_name,
                "t0": float(t0),
                "t1": float(t1),
            }
        )

    # Guard
    def _stacker_guard_wait(self):
        while True:
            if (not self.stacker_guard_enabled) or (self.stacker_guard_limit <= 0):
                return
            if self.stacker is None:
                return
            if len(self.stacker.items) < self.stacker_guard_limit:
                return
            yield self.env.timeout(1.0)

    # NEW: shift gating for manual movers
    def _wait_shift_move(self):
        if self.shift_move is None:
            return
        if not self.shift_move.active(int(self.env.now)):
            yield from self.shift_move.wait_if_inactive()

    def _manual_travel_min(self, key: str, override_min: float = None) -> float:
        # des_factory_web: manual_post에 to_*_travel_min 있으면 그걸 우선
        if override_min is not None:
            try:
                ov = float(override_min)
                if ov > 0:
                    return ov
            except Exception:
                pass

        d = float(self.manual_dist.get(key, 0.0) or 0.0)
        if self.manual_speed_m_per_s <= 0:
            return 0.0
        return (d / self.manual_speed_m_per_s) / 60.0

    def _manual_move(self, route_key: str, job: "Job" = None, *, include_unload=False):
        """
        manual mover travel between stations (SHIFT-GATED for movement)
        - seize mover
        - if include_unload: spend human_unload_min (shift gated)
        - travel time (shift gated)
        """
        yield from self._wait_shift_move()
        

        if self.logger:
            self.logger.log_event(
                "VALIDATION",
                (
                    f"[MOVE-MANUAL] Job {getattr(job,'id_job',None)} | Platform {getattr(job,'platform_id',None)} | "
                    f"route={route_key} "
                )
            )

        unload_used = 0.0
        travel_min = 0.0
        t0 = t1 = self.env.now

        with self.movers.resource.request() as req:
            yield req
            yield from self._wait_shift_move()

            t0 = self.env.now

            # unload (optional, only for printer -> wash1 in manual flow)
            if include_unload and self.human_unload_min > 0:
                unload_used = float(self.human_unload_min)
                yield self.env.timeout(unload_used)


            # travel (shift-gated, but we model it as continuous within active shift segments
            # here via wait gating + timeout; des_factory_web uses a more granular work-with-shift,
            # but effect is "no movement outside shift")
            # route_key naming: printer_to_wash1 / wash1_to_wash2 / wash_to_dry / uv_to_stacker
            override = None
            if route_key == "printer_to_wash1":
                override = self.to_washer_travel_min
            elif route_key == "wash1_to_wash2":
                override = self.to_wash2_travel_min
            elif route_key == "wash_to_dry":
                override = self.to_dryer_travel_min
            elif route_key in ("uv_to_stacker", "to_stacker"):
                override = self.to_stacker_travel_min

            travel_min = self._manual_travel_min(route_key, override_min=override)
            remaining = travel_min

            while remaining > 0:
                # 1) 근무시간 아니면 다음 근무까지 대기
                if not self.shift_move.active(int(self.env.now)):
                    yield from self.shift_move.wait_if_inactive()
                    continue

                # 2) 이번 근무에서 남은 시간
                workable = self.shift_move.remaining_work_minutes(int(self.env.now))
                if workable <= 0:
                    yield from self.shift_move.wait_if_inactive()
                    continue

                # 3) 이번에 실제 이동할 시간
                move_now = min(remaining, workable)

                # 4) 이동 수행
                yield self.env.timeout(move_now)
                remaining -= move_now


            t1 = self.env.now

        if job is not None:
            self._record_trace_mover(t0, t1, job)

        if self.logger:
            self.logger.log_event(
                "VALIDATION",
                (   
                    f"[MOVE-MANUAL][DONE] Job {getattr(job,'id_job',None)} | Platform {getattr(job,'platform_id',None)} | "
                    f"route={route_key} | travel_min={travel_min:.2f} | "
                    f"t0={t0:.2f} -> t1={t1:.2f} (elapsed={(t1-t0):.2f})"
                )
            )

 
    # AMR move (unchanged)
    def _amr_move(self, from_to_key: str, job: "Job" = None, dest_resource=None):
        if self.amr_pool is None:
            if self.logger:
                self.logger.log_event(
                    "VALIDATION",
                    f"[MOVE-AMR][SKIP] Job {getattr(job,'id_job',None)} | route={from_to_key} | amr_pool=None"
                )
            yield self.env.timeout(0)
            return None

        dist_m = self.dist[from_to_key]
        drive_min = (dist_m / self.speed_m_per_s) / 60.0
        travel_min = float(self.load_min) + float(drive_min)
        unload_min = float(self.unload_min)

        if self.logger:
            self.logger.log_event(
                "VALIDATION",
                (
                    f"[MOVE-AMR] Job {getattr(job,'id_job',None)} | Platform {getattr(job,'platform_id',None)} | "
                    f"route={from_to_key} | dist_m={dist_m} | speed_mps={self.speed_m_per_s} | "
                    f"load_min={float(self.load_min):.2f} | drive_min={drive_min:.2f} | unload_min={unload_min:.2f} | "
                    f"travel_min={travel_min:.2f} | amr_cap={getattr(self.amr_pool.resource,'capacity',1)} | "
                )
            )

        self.kpi.amr_travel_min += (travel_min + unload_min)

        dest_req = None
        with self.amr_pool.resource.request() as amr_req:
            yield amr_req
            t0 = self.env.now

            yield self.env.timeout(travel_min)

            if dest_resource is not None:
                dest_req = dest_resource.request()
                yield dest_req

            if unload_min > 0:
                yield self.env.timeout(unload_min)

            t1 = self.env.now

        if job is not None:
            self._record_trace_amr(t0, t1, job)

        if self.logger:
            self.logger.log_event(
                "VALIDATION",
                (
                    f"[MOVE-AMR][DONE] Job {getattr(job,'id_job',None)} | Platform {getattr(job,'platform_id',None)} | "
                    f"route={from_to_key} | t0={t0:.2f} -> t1={t1:.2f} (elapsed={(t1-t0):.2f}) | "
                    f"dest_seized={'Y' if dest_req is not None else 'N'}"
                )
            )

        return dest_req

    # Generic stage runner
    def _run_stage(self, *, route_key: str, stage_name: str, stage_resource, proc_time_min: float,
                   kpi_busy_field: str, job: "Job", include_unload=False):
        """
        des_factory_web style:
        - 이동 방식은 flow_mode에 따라 AMR 또는 manual mover
        - 설비 가공(세척/건조/UV)은 시작하면 24/7 (shift gating 없음)
        """

        # 1) 이동 + 도착지 리소스 점유
        req = None
        if self.flow_mode == "manual":
            if self.logger:
                self.logger.log_event(
                    "VALIDATION",
                    f"[ROUTE] mode=manual | stage={stage_name} | route_key={route_key} | job={job.id_job}"
                )
            yield from self._manual_move(route_key, job, include_unload=include_unload)
            req = stage_resource.request()
            yield req
        else:
            if self.logger:
                self.logger.log_event(
                    "VALIDATION",
                    f"[ROUTE] mode=automated | stage={stage_name} | route_key={route_key} | job={job.id_job}"
                )

            if self.amr_pool is None:
                req = stage_resource.request()
                yield req
            else:
                req = yield self.env.process(self._amr_move(route_key, job, dest_resource=stage_resource))
                if req is None:
                    req = stage_resource.request()
                    yield req

        # 2) 공정 시작 로그
        if self.logger:
            self.logger.log_event(
                "PostProcessLine",
                f"[START] Job {job.id_job} | Platform {getattr(job, 'platform_id', None)} | "
                f"{stage_name}_start (parts={len(getattr(job, 'list_items', []) or [])})"
            )

        # 3) 설비 가공(shift gating 없음)
        setattr(self.kpi, kpi_busy_field, getattr(self.kpi, kpi_busy_field) + float(proc_time_min))
        t0 = self.env.now
        yield self.env.timeout(proc_time_min)
        t1 = self.env.now

        if self.logger:
            self.logger.log_event(
                "PostProcessLine",
                f"[COMPLETED] Job {job.id_job} | Platform {getattr(job, 'platform_id', None)} | "
                f"{stage_name}_completed (parts={len(getattr(job, 'list_items', []) or [])})"
            )

        # 4) release
        if req is not None:
            stage_resource.release(req)

        # 5) trace
        self._record_trace_stage(stage_name, t0, t1, job)

        return (t0, t1)

    # Defect / scrap (full scrap & stop)
    def _fail(self, rate: float) -> bool:
        try:
            r = float(rate)
        except Exception:
            r = 0.0
        return (r > 0.0) and (random.random() < r)

    def _scrap_all_and_stop(self, job: "Job", *, stage: str, n_good: int, scrapped_total: int):
        scr = int(max(0, n_good))
        if scr > 0:
            self.kpi.scrapped_parts += scr
            scrapped_total += scr

        job.is_scrapped_job = True
        job.scrap_stage = stage
        job.scrapped_in_auto = scrapped_total
        job.auto_post_good_parts = 0
        job.auto_post_scrapped_parts = scrapped_total

        if hasattr(job, "list_items") and job.list_items is not None:
            job.list_items = []

        if self.logger:
            self.logger.log_event(
                "PostProcessLine",
                f"[SCRAP-FULL] Job {job.id_job} | Platform {getattr(job,'platform_id',None)} | "
                f"stage={stage} failed, scrapped_all_remaining={scr}, scrapped_total={scrapped_total}"
            )

        return 0, scrapped_total, True

    # Stage methods
    def _wash1(self, job: "Job", n_good: int, scrapped_total: int):
        t = self.m1.proc_time
        # manual 모드에서는 printer->wash1 이동 시 human_unload 포함(des_factory_web의 unload 반영)
        yield from self._run_stage(
            route_key="printer_to_wash1",
            stage_name="WashM1",
            stage_resource=self.m1.resource,
            proc_time_min=t,
            kpi_busy_field="wash1_busy_min",
            job=job,
            include_unload=(self.flow_mode == "manual"),
        )

        if self._fail(self.def_wash):
            return self._scrap_all_and_stop(job, stage="wash1", n_good=n_good, scrapped_total=scrapped_total)

        return n_good, scrapped_total, False

    def _wash2(self, job: "Job", n_good: int, scrapped_total: int):
        t = self.m2.proc_time
        yield from self._run_stage(
            route_key="wash1_to_wash2",
            stage_name="WashM2",
            stage_resource=self.m2.resource,
            proc_time_min=t,
            kpi_busy_field="wash2_busy_min",
            job=job,
        )
        # des_factory_web 정책: wash2에서 불량 게이트 없음
        return n_good, scrapped_total

    def _dry(self, job: "Job", n_good: int, scrapped_total: int):
        t = self.dry.proc_time
        yield from self._run_stage(
            route_key="wash_to_dry",
            stage_name="Dryer",
            stage_resource=self.dry.resource,
            proc_time_min=t,
            kpi_busy_field="dry_busy_min",
            job=job,
        )

        if self._fail(self.def_dry):
            return self._scrap_all_and_stop(job, stage="dry", n_good=n_good, scrapped_total=scrapped_total)

        return n_good, scrapped_total, False

    def _uv(self, job: "Job", n_good: int, scrapped_total: int):
        t = self.uv.proc_time
        yield from self._run_stage(
            route_key="dry_to_uv",
            stage_name="UV",
            stage_resource=self.uv.resource,
            proc_time_min=t,
            kpi_busy_field="uv_busy_min",
            job=job,
        )

        if self._fail(self.def_uv):
            return self._scrap_all_and_stop(job, stage="uv", n_good=n_good, scrapped_total=scrapped_total)

        return n_good, scrapped_total, False

    # Stacker push (flow_mode에 따라 이동 방식 선택)
    def _build_payload(self, job: "Job", n_good: int, scrapped_total: int):
        payload = build_stacker_payload(job, platform_id=job.platform_id, env_now=self.env.now)
        payload["good_parts_after_auto"] = n_good
        payload["scrapped_in_auto"] = scrapped_total
        payload["is_scrapped_job"] = (n_good == 0)
        return payload

    def _push_to_stacker(self, job: "Job", payload: Dict[str, Any]):
        if self.stacker is None:
            return

        # manual: shift-gated move -> guard -> put
        if self.flow_mode == "manual":
            yield from self._stacker_guard_wait()
            yield self.stacker.put(payload)
            return

        # automated: AMR move + guard + put + unload
        if self.amr_pool is None:
            yield from self._stacker_guard_wait()
            yield self.stacker.put(payload)
            return

        dist_m = self.dist["uv_to_stacker"]
        drive_min = (dist_m / self.speed_m_per_s) / 60.0
        travel_min = float(self.load_min) + float(drive_min)
        unload_min = float(self.unload_min)

        self.kpi.amr_travel_min += (travel_min + unload_min)

        with self.amr_pool.resource.request() as amr_req:
            yield amr_req
            t0 = self.env.now

            yield self.env.timeout(travel_min)
            yield from self._stacker_guard_wait()
            yield self.stacker.put(payload)

            if unload_min > 0:
                yield self.env.timeout(unload_min)

            t1 = self.env.now

        self._record_trace_amr(t0, t1, job)

    # Main run
    def run(self, job: "Job"):
        n_parts = len(getattr(job, "list_items", []) or [])
        n_good = n_parts
        scrapped_total = 0

        if getattr(job, "is_scrapped_job", False):
            if self.logger:
                self.logger.log_event(
                    "PostProcessLine",
                    f"[SKIP] Job {job.id_job} already scrapped at {getattr(job, 'scrap_stage', 'unknown')}",
                )
            return

        # Wash1 (fail gate)
        n_good, scrapped_total, stop = (yield from self._wash1(job, n_good, scrapped_total))
        if stop:
            if self.logger:
                self.logger.log_event(
                    "VALIDATION",
                    f"[SCRAP->CLEAN] Job {job.id_job} | Platform {getattr(job,'platform_id',None)} | "
                    f"Wash1 failed → STOP post line and proceed to PlatformClean."
                )
            return

        # Wash2 (no fail gate)
        n_good, scrapped_total = (yield from self._wash2(job, n_good, scrapped_total))

        # Dry (fail gate)
        n_good, scrapped_total, stop = (yield from self._dry(job, n_good, scrapped_total))
        if stop:
            if self.logger:
                self.logger.log_event(
                    "VALIDATION",
                    f"[SCRAP->CLEAN] Job {job.id_job} | Platform {getattr(job,'platform_id',None)} | "
                    f"Dry failed → STOP post line and proceed to PlatformClean."
                )
            return

        # UV (fail gate)
        n_good, scrapped_total, stop = (yield from self._uv(job, n_good, scrapped_total))
        if stop:
            if self.logger:
                self.logger.log_event(
                    "VALIDATION",
                    f"[SCRAP->CLEAN] Job {job.id_job} | Platform {getattr(job,'platform_id',None)} | "
                    f"UV failed → STOP post line and proceed to PlatformClean."
                )
            return

        # success -> stacker
        if self.stacker is not None:
            payload = self._build_payload(job, n_good, scrapped_total)
            yield from self._push_to_stacker(job, payload)

            wip_now = len(self.stacker.items)
            if wip_now > self.kpi.max_stacker_wip:
                self.kpi.max_stacker_wip = wip_now

        if self.logger:
            self.logger.log_event(
                "PostProcessLine",
                f"[END] Job {job.id_job} | Platform {getattr(job, 'platform_id', None)} | "
                f"post_line_finished (mode={self.flow_mode}, good={n_good}, scrap={scrapped_total})"
            )


# 4. ManualPost (Worker)  -> 사실상 manual_ops(서포트/피니시/페인트)
class ManualPost:
    def __init__(self, env, cfg, kpi: "KPI", logger=None, stacker=None, shift=None):
        self.env = env
        self.cfg = cfg["manual_ops"]
        self.kpi = kpi
        self.logger = logger
        self.stacker = stacker
        self.shift = shift

        self.workers = Worker(env, int(self.cfg.get("workers", 1) or 1), 0)
        self._worker_slots: Dict[int, list] = {}

        self.move_times = {
            "plat_to_support": self.cfg["move_platform_to_support_min"],
            "support_to_finish": self.cfg["move_support_to_finish_min"],
            "finish_to_paint": self.cfg["move_finish_to_paint_min"],
            "paint_to_storage": self.cfg["move_paint_to_storage_min"],
        }

        self.dist = self.cfg.get("dist_m", {})
        self.speed_m_per_s = float(self.cfg.get("speed_m_per_s", 1.0))

    def _record_trace(self, t0: float, t1: float, job: "Job", platform_id: str):
        if not self.logger or not hasattr(self.logger, "trace_events"):
            return

        cap = getattr(self.workers.resource, "capacity", 1)
        slot_idx = _allocate_slot(self._worker_slots, cap, t0, t1)
        res_name = f"Worker_{slot_idx}"

        self.logger.trace_events.append(
            {
                "id": f"Job {job.id_job}",
                "job_id": job.id_job,
                "platform_id": platform_id,
                "stage": "Worker",
                "Resource": res_name,
                "t0": float(t0),
                "t1": float(t1),
            }
        )

    def run(self, job: "Job"):
        # shift 적용
        if self.shift and not self.shift.active(int(self.env.now)):
            yield from self.shift.wait_if_inactive()

        # stacker에서 payload pull
        platform_id = getattr(job, "platform_id", f"JOB-{job.id_job}")
        n_parts = len(getattr(job, "list_items", []) or [])

        payload = None
        if self.stacker is not None:
            payload = yield self.stacker.get()
            platform_id = payload.get("platform_id", platform_id)

            good_after_auto = payload.get("good_parts_after_auto")
            is_scrapped = payload.get("is_scrapped_job", False)

            if good_after_auto is not None:
                n_parts = int(good_after_auto)
            else:
                n_parts = int(payload.get("n_parts", n_parts))

            if is_scrapped or n_parts <= 0:
                if self.logger:
                    self.logger.log_event(
                        "MANUAL",
                        f"[SKIP] Job {job.id_job} | Platform {platform_id} fully scrapped before manual_ops"
                    )
                return

        # FIX: 플랫폼 단위 시간 키를 config와 맞춤
        # - config에는 support_removal_time_min(=플랫폼 단위 작업)로 되어 있음
        t_platform = float(self.cfg.get("support_removal_time_min", 0.0))

        # part-level times
        t_support_parts = float(self.cfg["support_time_per_part_min"]) * n_parts
        t_finish_parts = float(self.cfg["finish_time_per_part_min"]) * n_parts
        t_paint_parts = float(self.cfg["paint_time_per_part_min"]) * n_parts

        # movement times
        if self.dist:
            t_plat_to_support = manual_travel_time_min(self.dist.get("plat_to_support", 0.0), self.speed_m_per_s)
            t_support_to_finish = manual_travel_time_min(self.dist.get("support_to_finish", 0.0), self.speed_m_per_s)
            t_finish_to_paint = manual_travel_time_min(self.dist.get("finish_to_paint", 0.0), self.speed_m_per_s)
            t_paint_to_storage = manual_travel_time_min(self.dist.get("paint_to_storage", 0.0), self.speed_m_per_s)
            t_move = t_plat_to_support + t_support_to_finish + t_finish_to_paint + t_paint_to_storage
        else:
            t_move = (
                float(self.move_times["plat_to_support"])
                + float(self.move_times["support_to_finish"])
                + float(self.move_times["finish_to_paint"])
                + float(self.move_times["paint_to_storage"])
            )

        t_total = t_platform + t_support_parts + t_finish_parts + t_paint_parts + t_move
        self.kpi.manual_busy_min += t_total

        with self.workers.resource.request() as req:
            yield req

            if self.shift and not self.shift.active(int(self.env.now)):
                yield from self.shift.wait_if_inactive()

            start = self.env.now
            yield self.env.timeout(t_total)
            end = self.env.now

        self._record_trace(start, end, job, platform_id)

        self.kpi.completed_parts += n_parts
        if hasattr(job, "start_time"):
            self.kpi.sum_platform_lead_time_min += self.env.now - job.start_time


# 5. PlatformClean (PlatformWasher)
class PlatformClean:
    def __init__(self, env, cfg, platM: PlatformManager, kpi: KPI, logger=None):
        self.env = env
        self.cfg = cfg["platform_clean"]
        self.platM = platM
        self.kpi = kpi
        self.logger = logger

        # Platform washer machine
        self.cleaner = PlatformWasher(
            env,
            self.cfg["washers"],
            self.cfg["wash_time_min"],
        )

        # Slot usage for washer Gantt
        self._washer_slots: Dict[int, list] = {}

    def _record_trace(self, t0: float, t1: float, job: Job, platform_id: str):
        """Record platform washer usage interval."""
        if not self.logger or not hasattr(self.logger, "trace_events"):
            return

        cap = getattr(self.cleaner.resource, "capacity", 1)
        slot_idx = _allocate_slot(self._washer_slots, cap, t0, t1)
        res_name = f"PlatformWasher_{slot_idx}"

        self.logger.trace_events.append(
            {
                "id": f"Job {job.id_job}",
                "job_id": job.id_job,
                "platform_id": platform_id,
                "stage": "PlatformWasher",
                "Resource": res_name,
                "t0": float(t0),
                "t1": float(t1),
            }
        )

    def run(self, job: Job, platform_token: Dict[str, Any]):
        """
        Final platform cleaning stage:
          - Wash the used build platform
          - Return the platform token back to the clean pool via PlatformManager
          - Update KPI (platform_wash_busy_min, lead time, finished_platforms)
        """
        plat_id = platform_token.get("id", "?") if platform_token else "UNKNOWN"
        """
        if self.logger:
            self.logger.log_event(
                "PLATFORM_CLEAN",
                f"[START] Job {job.id_job} | Platform {plat_id} entering platform cleaning"
            )
        """
        # Seize platform washer
        with self.cleaner.resource.request() as req:
            yield req

            t = self.cleaner.proc_time
            self.kpi.platform_wash_busy_min += t

            t0 = self.env.now
            yield self.env.timeout(t)
            t1 = self.env.now

        # Record Gantt trace
        self._record_trace(t0, t1, job, plat_id)

        # Return cleaned platform to PlatformManager pool
        if platform_token is not None:
            # PlatformManager.release_platform() returns a process, so yield it.
            yield self.platM.release_platform(platform_token)

        # KPI: lead time + finished platforms
        if hasattr(job, "start_time"):
            lead = self.env.now - job.start_time
            self.kpi.sum_platform_lead_time_min += lead

        self.kpi.finished_platforms += 1

        
        # Logging
        if self.logger:
            self.logger.log_event(
                "PLATFORM_CLEAN",
                (
                    f"[END] Job {job.id_job} | Platform {plat_id} washed and returned to pool "
                    f"(wash_time={self.cleaner.proc_time} min)"
                ),
            )
        
