import simpy
import random
from typing import Any, Dict
from specialized_Processor import (Printer, WashM1, WashM2, Dryer, UVStation, Worker, PlatformWasher, AMR)
from flow_utils import (amr_travel_time_min, sample_stage_defects, build_stacker_payload, manual_travel_time_min,)
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
                        f"(healing={self.cfg['healing_time_per_platform_min']} min, "
                        f"placement={self.cfg['placement_time_per_platform_min']} min, "
                        f"support_gen={self.cfg['support_generation_time_min']} min)"
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

       
        # Breakdown / Maintenance config
        bd_cfg = cfg.get("breakdown", {})
        mt_cfg = cfg.get("maintenance", {})

        self.breakdown_enabled = bool(bd_cfg.get("enabled", False))
        self.mtbf_min = float(bd_cfg.get("mtbf_min", 0.0) or 0.0)
        self.mttr_min = float(bd_cfg.get("mttr_min", 0.0) or 0.0)

        self.maint_enabled = bool(mt_cfg.get("enabled", False))
        self.maint_start_hhmm = mt_cfg.get("start_hhmm", "00:00")
        self.maint_duration_min = float(mt_cfg.get("duration_min", 0.0) or 0.0)
        self.maint_cycle_days = int(mt_cfg.get("cycle_days", 1) or 1)

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

    def _log(self, msg: str):
        if self.logger:
            self.logger.log_event("Printer", msg)

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
            ttf = random.expovariate(1.0 / self.mtbf_min)
            yield self.env.timeout(ttf)

            # Breakdown start
            self._is_breakdown = True
            self._log(f"BREAKDOWN start (MTTR={self.mttr_min:.1f} min)")

            # Repair time
            yield self.env.timeout(self.mttr_min)

            # Breakdown end
            self._is_breakdown = False
            self._log("BREAKDOWN end (printer back to service)")

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
            self._log(
                f"MAINTENANCE start at t={self.env.now:.1f} "
                f"(duration={self.maint_duration_min:.1f} min)"
            )

            # Maintenance duration
            yield self.env.timeout(self.maint_duration_min)

            # Maintenance end
            self._is_maintenance = False
            self._log(f"MAINTENANCE end at t={self.env.now:.1f}")

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
        if self.logger and self._is_down():
            self.logger.log_event(
                "PRINT",
                f"Job {job.id_job} waiting: printer DOWN (breakdown/maintenance)"
            )

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

            if self.logger:
                self.logger.log_event(
                    "PRINT",
                    f"[START] Job {job.id_job} | Platform {getattr(job, 'platform_id', '?')} | "
                    f"print_time={t:.1f} min"
                )

            yield self.env.timeout(t)
            end = self.env.now

            if self.logger:
                self.logger.log_event(
                    "PRINT",
                    f"[END] Job {job.id_job} | Platform {getattr(job, 'platform_id', '?')} | "
                    f"print_finished (elapsed={end-start:.1f} min)"
                )

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

                print("[TRACE DEBUG] Print.run appended:", rec)
            else:
                print("[TRACE DEBUG] Print.run: logger or trace_events missing",
                      self.logger, hasattr(self.logger, "trace_events"))


# 3. AutoPost (WashM1, WashM2, Dryer, UV, AMRPool)
class AutoPost:
    """
    Automatic post-processing stage:
      - AMR travel between stations
      - Wash1 / Wash2 / Dry / UV processing
      - Defect / scrap sampling per stage
      - Push payload to stacker (for manual post-process)
    """

    def __init__(
        self,
        env,
        cfg,
        kpi: KPI,
        logger=None,
        amr_pool: AMR = None,
        stacker=None,
        shift_amr=None,
    ):
        self.env = env
        self.cfg = cfg["auto_post"]
        self.kpi = kpi
        self.logger = logger

        # Machines
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

        # AMR / stacker / shift
        self.amr_pool = amr_pool
        self.stacker = stacker
        self.shift_amr = shift_amr

        # AMR parameters
        self.dist = self.cfg["dist_m"]
        self.speed_m_per_s = self.cfg["amr_speed_m_per_s"]
        self.load_min = self.cfg["amr_load_min"]
        self.unload_min = self.cfg["amr_unload_min"]

        # Defect rate per stage
        self.def_wash = self.cfg.get("defect_rate_wash", 0.0)
        self.def_dry = self.cfg.get("defect_rate_dry", 0.0)
        self.def_uv = self.cfg.get("defect_rate_uv", 0.0)

        # Stacker guard config
        guard_cfg = cfg.get("stacker_guard", {})
        self.stacker_guard_enabled = bool(guard_cfg.get("enabled", False))
        self.stacker_guard_limit = int(guard_cfg.get("max_platforms", 0) or 0)

    # Gantt trace helpers
    def _record_trace_stage(self, stage_name: str, t0: float, t1: float, job: Job):
        """Record intervals for WashM1 / WashM2 / Dryer / UV."""
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

    def _record_trace_amr(self, t0: float, t1: float, job: Job):
        """Record AMRPool travel intervals."""
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

    # AMR travel
    def _amr_move(self, from_to_key: str, job: Job = None):
        """
        AMR travel between stations + trace recording.
        """
        if self.amr_pool is None:
            return self.env.timeout(0)

        dist_m = self.dist[from_to_key]
        t_move = amr_travel_time_min(
            dist_m,
            self.speed_m_per_s,
            load_min=self.load_min,
            unload_min=self.unload_min,
        )

        self.kpi.amr_travel_min += t_move

        if self.logger:
            self.logger.log_event(
                "AUTO_POST",
                f"AMR MOVE [{from_to_key}] for Job {getattr(job, 'id_job', '?')} | "
                f"dist={dist_m} m, t_move={t_move:.1f} min"
            )

        with self.amr_pool.resource.request() as req:
            yield req
            t0 = self.env.now
            yield self.env.timeout(t_move)
            t1 = self.env.now

        if job is not None:
            self._record_trace_amr(t0, t1, job)

    # main run
    def run(self, job: Job):
        n_parts = len(job.list_items)
        n_good = n_parts
        scrapped_total = 0

        if self.logger:
            self.logger.log_event(
                "AUTO_POST",
                f"[START] Job {job.id_job} | Platform {getattr(job, 'platform_id', '?')} | "
                f"auto-post process begins (n_parts={n_parts})"
            )

        # Wash1 
        if self.shift_amr and not self.shift_amr.active(int(self.env.now)):
            yield from self.shift_amr.wait_if_inactive()

        # AMR: printer -> wash1
        yield self.env.process(self._amr_move("printer_to_wash1", job))

        if self.logger:
            self.logger.log_event(
                "AUTO_POST",
                f"Job {job.id_job} entering WashM1"
            )

        with self.m1.resource.request() as req:
            yield req
            t = self.m1.proc_time
            self.kpi.wash1_busy_min += t
            t0 = self.env.now
            yield self.env.timeout(t)
            t1 = self.env.now

        # WashM1 trace
        self._record_trace_stage("WashM1", t0, t1, job)

        scr = sample_stage_defects(n_good, self.def_wash)
        n_good -= scr
        scrapped_total += scr
        self.kpi.scrapped_parts += scr

        if self.logger:
            self.logger.log_event(
                "AUTO_POST",
                f"WashM1 done for Job {job.id_job} | t={t:.1f} min, "
                f"scrapped={scr}, remaining_good={n_good}"
            )

        # Wash2
        if self.shift_amr and not self.shift_amr.active(int(self.env.now)):
            yield from self.shift_amr.wait_if_inactive()

        yield self.env.process(self._amr_move("wash1_to_wash2", job))

        if self.logger:
            self.logger.log_event(
                "AUTO_POST",
                f"Job {job.id_job} entering WashM2"
            )

        with self.m2.resource.request() as req:
            yield req
            t = self.m2.proc_time
            self.kpi.wash2_busy_min += t
            t0 = self.env.now
            yield self.env.timeout(t)
            t1 = self.env.now

        # WashM2 trace
        self._record_trace_stage("WashM2", t0, t1, job)

        scr = sample_stage_defects(n_good, self.def_wash)
        n_good -= scr
        scrapped_total += scr
        self.kpi.scrapped_parts += scr

        if self.logger:
            self.logger.log_event(
                "AUTO_POST",
                f"WashM2 done for Job {job.id_job} | t={t:.1f} min, "
                f"scrapped={scr}, remaining_good={n_good}"
            )

        # Dryer
        if self.shift_amr and not self.shift_amr.active(int(self.env.now)):
            yield from self.shift_amr.wait_if_inactive()

        yield self.env.process(self._amr_move("wash_to_dry", job))

        if self.logger:
            self.logger.log_event(
                "AUTO_POST",
                f"Job {job.id_job} entering Dryer"
            )

        with self.dry.resource.request() as req:
            yield req
            t = self.dry.proc_time
            self.kpi.dry_busy_min += t
            t0 = self.env.now
            yield self.env.timeout(t)
            t1 = self.env.now

        # Dryer trace
        self._record_trace_stage("Dryer", t0, t1, job)

        scr = sample_stage_defects(n_good, self.def_dry)
        n_good -= scr
        scrapped_total += scr
        self.kpi.scrapped_parts += scr

        if self.logger:
            self.logger.log_event(
                "AUTO_POST",
                f"Dryer done for Job {job.id_job} | t={t:.1f} min, "
                f"scrapped={scr}, remaining_good={n_good}"
            )

        # UV
        if self.shift_amr and not self.shift_amr.active(int(self.env.now)):
            yield from self.shift_amr.wait_if_inactive()

        yield self.env.process(self._amr_move("dry_to_uv", job))

        if self.logger:
            self.logger.log_event(
                "AUTO_POST",
                f"Job {job.id_job} entering UV"
            )

        with self.uv.resource.request() as req:
            yield req
            t = self.uv.proc_time
            self.kpi.uv_busy_min += t
            t0 = self.env.now
            yield self.env.timeout(t)
            t1 = self.env.now

        # UV trace
        self._record_trace_stage("UV", t0, t1, job)

        scr = sample_stage_defects(n_good, self.def_uv)
        n_good -= scr
        scrapped_total += scr
        self.kpi.scrapped_parts += scr

        if self.logger:
            self.logger.log_event(
                "AUTO_POST",
                f"UV done for Job {job.id_job} | t={t:.1f} min, "
                f"scrapped={scr}, remaining_good={n_good}"
            )

        # UV -> Stacker
        if self.shift_amr and not self.shift_amr.active(int(self.env.now)):
            yield from self.shift_amr.wait_if_inactive()
        yield self.env.process(self._amr_move("uv_to_stacker", job))

        # Stacker payload
        if self.stacker is not None:
            payload = build_stacker_payload(
                job, platform_id=job.platform_id, env_now=self.env.now
            )
            payload["good_parts_after_auto"] = n_good
            payload["scrapped_in_auto"] = scrapped_total
            payload["is_scrapped_job"] = n_good == 0

            # Stacker guard (limit WIP)
            if (getattr(self, "stacker_guard_enabled", False)
                and self.stacker_guard_limit > 0):
                if self.logger:
                    self.logger.log_event(
                        "AUTO_POST",
                        f"Stacker guard active (limit={self.stacker_guard_limit}). "
                        f"Job {job.id_job} payload will wait if WIP >= limit."
                    )

            while True:
                if (not getattr(self, "stacker_guard_enabled", False)) or \
                   (self.stacker_guard_limit <= 0) or \
                   (len(self.stacker.items) < self.stacker_guard_limit):
                    break
                yield self.env.timeout(1.0)

            yield self.stacker.put(payload)

            wip_now = len(self.stacker.items)
            if wip_now > self.kpi.max_stacker_wip:
                self.kpi.max_stacker_wip = wip_now

            if self.logger:
                self.logger.log_event(
                    "AUTO_POST",
                    f"[STACKER] Job {job.id_job} payload pushed | "
                    f"good_after_auto={n_good}, scrapped_total={scrapped_total}, "
                    f"stacker_WIP={wip_now}"
                )

        if self.logger:
            self.logger.log_event(
                "AUTO_POST",
                f"[END] Job {job.id_job} | auto-post process finished "
                f"(good_after_auto={n_good}, scrapped_total={scrapped_total})"
            )


# 4. ManualPost (Worker)
class ManualPost:
    def __init__(self, env, cfg, kpi: KPI, logger=None, stacker=None, shift=None):
        self.env = env
        self.cfg = cfg["manual_ops"]
        self.kpi = kpi
        self.logger = logger

        self.stack_cfg = cfg.get("stacker_guard", {})
        self.stacker = stacker

        # Manual worker resource
        self.workers = Worker(env, self.cfg["workers"], 0)

        # Slot usage for worker Gantt
        self._worker_slots: Dict[int, list] = {}

        # Predefined move times (if distance-based modeling is not used)
        self.move_times = {
            "plat_to_support": self.cfg["move_platform_to_support_min"],
            "support_to_finish": self.cfg["move_support_to_finish_min"],
            "finish_to_paint": self.cfg["move_finish_to_paint_min"],
            "paint_to_storage": self.cfg["move_paint_to_storage_min"],
        }

        self.dist = self.cfg.get("dist_m", {})
        self.speed_m_per_s = float(self.cfg.get("speed_m_per_s", 1.0))

        # Worker shift info
        self.shift_cfg = self.cfg.get("work_shift", None)
        self.shift = shift

    def _record_trace(self, t0: float, t1: float, job: Job, platform_id: str):
        """Record manual worker processing interval."""
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

    def run(self, job: Job):
        """
        Manual post-process for one platform:
          1) Retrieve platform payload from the stacker
          2) One worker handles the entire platform
             - Platform-level tasks
             - Part-level support / finish / paint
             - Manual movement between stations
        """
        # Respect worker shift
        if self.shift and not self.shift.active(int(self.env.now)):
            yield from self.shift.wait_if_inactive()

        if self.logger:
            self.logger.log_event(
                "MANUAL",
                f"[START] Job {job.id_job} manual post-process begins"
            )

        # 1) Pull payload from stacker (if available)
        platform_id = getattr(job, "platform_id", f"JOB-{job.id_job}")
        n_parts = len(job.list_items)

        payload = None
        if self.stacker is not None:
            payload = yield self.stacker.get()
            # If payload has explicit platform id and part count, override
            platform_id = payload.get("platform_id", platform_id)
            n_parts = int(payload.get("n_parts", n_parts))

            if self.logger:
                self.logger.log_event(
                    "MANUAL",
                    f"Job {job.id_job} pulled from stacker | "
                    f"platform_id={platform_id}, n_parts={n_parts}"
                )

        # Platform-level time
        t_platform = float(self.cfg.get("support_time_per_platform_min", 0.0))

        # Part-level support / finish / paint time
        t_support_parts = float(self.cfg["support_time_per_part_min"]) * n_parts
        t_finish_parts = float(self.cfg["finish_time_per_part_min"]) * n_parts
        t_paint_parts = float(self.cfg["paint_time_per_part_min"]) * n_parts

        # Manual movement between stations
        if self.dist:
            t_plat_to_support = manual_travel_time_min(
                self.dist.get("plat_to_support", 0.0), self.speed_m_per_s
            )
            t_support_to_finish = manual_travel_time_min(
                self.dist.get("support_to_finish", 0.0), self.speed_m_per_s
            )
            t_finish_to_paint = manual_travel_time_min(
                self.dist.get("finish_to_paint", 0.0), self.speed_m_per_s
            )
            t_paint_to_storage = manual_travel_time_min(
                self.dist.get("paint_to_storage", 0.0), self.speed_m_per_s
            )

            t_move = (
                t_plat_to_support
                + t_support_to_finish
                + t_finish_to_paint
                + t_paint_to_storage
            )
        else:
            # Fallback: use fixed times from config
            t_move = (
                float(self.move_times["plat_to_support"])
                + float(self.move_times["support_to_finish"])
                + float(self.move_times["finish_to_paint"])
                + float(self.move_times["paint_to_storage"])
            )

        # Total manual processing time
        t_total = (
            t_platform + t_support_parts + t_finish_parts + t_paint_parts + t_move
        )

        # KPI: manual worker busy time
        self.kpi.manual_busy_min += t_total

        if self.logger:
            self.logger.log_event(
                "MANUAL",
                f"Job {job.id_job} | Platform {platform_id} | "
                f"manual total_time={t_total:.1f} min "
                f"(platform={t_platform:.1f}, support={t_support_parts:.1f}, "
                f"finish={t_finish_parts:.1f}, paint={t_paint_parts:.1f}, "
                f"move={t_move:.1f})"
            )

        # Acquire one worker and process the entire platform
        with self.workers.resource.request() as req:
            yield req

            # Check shift again in case we crossed a shift boundary
            if self.shift and not self.shift.active(int(self.env.now)):
                yield from self.shift.wait_if_inactive()

            start = self.env.now
            yield self.env.timeout(t_total)
            end = self.env.now

        # Record Gantt trace
        self._record_trace(start, end, job, platform_id)

        # KPI: completed parts and lead time
        self.kpi.completed_parts += n_parts
        if hasattr(job, "start_time"):
            self.kpi.sum_platform_lead_time_min += self.env.now - job.start_time

        # Logging
        if self.logger:
            self.logger.log_event(
                "MANUAL",
                f"[END] Job {job.id_job} | Platform {platform_id} | "
                f"manual post-process finished (elapsed={end-start:.1f} min)"
            )


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

        if self.logger:
            self.logger.log_event(
                "PLATFORM_CLEAN",
                f"[START] Job {job.id_job} | Platform {plat_id} entering platform cleaning"
            )

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
