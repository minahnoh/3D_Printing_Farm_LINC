import simpy
from typing import Any, Dict
from specialized_Processor import (
    Printer,
    WashM1,
    WashM2,
    Dryer,
    UVStation,
    Worker,
    PlatformWasher
)
from platform_manager import PlatformManager
from KPI import KPI
from base_Job import Job


class Preprocess:
    def __init__(self, env: simpy.Environment, cfg: Dict[str, Any], platM: PlatformManager, kpi: KPI, logger=None):
        self.env = env
        self.cfg = cfg["preproc"]
        self.kpi = kpi
        self.logger = logger
        self.platM = platM

    def run(self, job: Job):
        """Assign platform + basic prep timing."""
        
        # Get an available platform
        token = yield self.platM.get_clean_platform()
        
        # Store info on the Job object
        platform_id = self.platM.assign(job.id_job, token)
        job.platform_id = platform_id
        job.start_time = self.env.now
        self.kpi.started_platforms += 1
        self.kpi.n_preproc_jobs += 1

        # Preprocess timing logic
        t = (
            self.cfg["healing_time_per_platform_min"]
            + self.cfg["placement_time_per_platform_min"]
            + self.cfg["support_generation_time_min"]
        )
        self.kpi.preproc_busy_min += t
       
        if self.logger:
            self.logger.log_event(
                "PREPROC",
                f"[ASSIGN] Job {job.id_job} ← Platform {platform_id} | preprocessing_total_time: {t} min"
                f"| healing time = {self.cfg["placement_time_per_platform_min"]} min |"
                f"| support_generation_time= {self.cfg["support_generation_time_min"]} min |"
                
        )
        
        yield self.env.timeout(t)
        return token


class Print:
    def __init__(self, env, cfg, kpi: KPI, logger=None):
        self.env = env
        self.kpi = kpi
        self.logger = logger

        self.proc = Printer(env, cfg["printer_count"], cfg["t_print_per_platform_min"])

    def run(self, job: Job):
        with self.proc.resource.request() as req:
            yield req
            t = self.proc.proc_time
            self.kpi.printer_busy_min += t
            self.kpi.n_print_jobs += 1

            """
            if self.logger:
                self.logger.log_event("PRINT", f"Job {job.id_job} printing {t} min")
            """

            yield self.env.timeout(t)


class AutoPost:
    def __init__(self, env, cfg, kpi: KPI, logger=None):
        self.env = env
        self.cfg = cfg["auto_post"]
        self.kpi = kpi
        self.logger = logger

        self.m1 = WashM1(env, self.cfg["washers_m1"], self.cfg["t_wash1_min"])
        self.m2 = WashM2(env, self.cfg["washers_m2"], self.cfg["t_wash2_min"])
        self.dry = Dryer(env, self.cfg["dryers"], self.cfg["t_dry_min"])
        self.uv = UVStation(env, self.cfg["uv_units"], self.cfg["t_uv_min"])

    def run(self, job: Job):
        # Wash1
        with self.m1.resource.request() as req:
            yield req
            t = self.m1.proc_time
            self.kpi.wash1_busy_min += t
            if self.logger: self.logger.log_event("WASH1", f"Job {job.id_job} W1 {t} min")
            yield self.env.timeout(t)

        # Wash2
        with self.m2.resource.request() as req:
            yield req
            t = self.m2.proc_time
            self.kpi.wash2_busy_min += t
            if self.logger: self.logger.log_event("WASH2", f"Job {job.id_job} W2 {t} min")
            yield self.env.timeout(t)

        # Dry
        with self.dry.resource.request() as req:
            yield req
            t = self.dry.proc_time
            self.kpi.dry_busy_min += t
            if self.logger: self.logger.log_event("DRY", f"Job {job.id_job} Dry {t} min")
            yield self.env.timeout(t)

        # UV
        with self.uv.resource.request() as req:
            yield req
            t = self.uv.proc_time
            self.kpi.uv_busy_min += t
            if self.logger: self.logger.log_event("UV", f"Job {job.id_job} UV {t} min")
            yield self.env.timeout(t)


class ManualPost:
    def __init__(self, env, cfg, kpi: KPI, logger=None):
        self.env = env
        self.cfg = cfg["manual_ops"]
        self.kpi = kpi
        self.logger = logger

        self.workers = Worker(env, self.cfg["workers"], 0)

    def run(self, job: Job):
        with self.workers.resource.request() as req:
            yield req
            t = (
                self.cfg["support_removal_time_min"]
                + self.cfg["finish_time_per_part_min"] * len(job.list_items)
                + self.cfg["paint_time_per_part_min"] * len(job.list_items)
            )
            self.kpi.manual_busy_min += t

            """
            if self.logger:
                self.logger.log_event("MANUAL", f"Job {job.id_job} finishing {t} min")
            """

            yield self.env.timeout(t)


class PlatformClean:
    def __init__(self, env, cfg, platM: PlatformManager, kpi: KPI, logger=None):
        self.env = env
        self.cfg = cfg["platform_clean"]
        self.platM = platM
        self.kpi = kpi
        self.logger = logger

        self.cleaner = PlatformWasher(env, self.cfg["washers"], self.cfg["wash_time_min"])

    def run(self, job: Job, platform_token):
        with self.cleaner.resource.request() as req:
            yield req
            t = self.cleaner.proc_time
            self.kpi.platform_wash_busy_min += t

            """
            if self.logger:
                self.logger.log_event("PLAT_CLEAN", f"Platform {platform_token['id']} wash {t} min")
            """

            yield self.env.timeout(t)

        # platform goes back to available pool
        yield self.platM.release_platform(platform_token)

        # KPI final add
        lead = self.env.now - job.start_time
        self.kpi.sum_platform_lead_time_min += lead
        self.kpi.finished_platforms += 1

        """
        if self.logger:
            self.logger.log_event("FINISH", f"Job {job.id_job} DONE lead {lead:.1f}min")
        """
