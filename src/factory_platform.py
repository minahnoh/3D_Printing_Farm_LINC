# factory_platform.py  (FULLY FIXED VERSION)

import simpy
from collections import defaultdict
from config_SimPy import build_factory_param_dict
from config_SimPy import *
from base_Processor import (
    WasherM1, WasherM2, Dryer, UVUnit,
    PreprocServer, PrinterMachine, PlatformWasher
)
from job_platform_mapper import JobPlatformMapper
from platform_manager import *
from preprocessing_manager import *


class PlatformKPI:
    """
    KPI container for the platform-level factory.
    """
    def __init__(self):
        self.started_parts = 0
        self.started_platforms = 0
        self.completed_parts = 0
        self.finished_platforms = 0
        self.scrapped_parts = 0

        # 전처리/레진 관련 KPI
        self.n_preproc_jobs = 0        # 전처리 작업 수
        self.preproc_busy_min = 0.0 
        self.resin_used_kg = 0.0       # 실제로 사용된 레진량
        self.resin_wasted_kg = 0.0
        self.resin_cost_krw = 0.0

        self.wait_stats = defaultdict(float)

    def add_wait(self, key, delta):
        self.wait_stats[key] += float(delta)


class Factory:
    """
    Pure SimPy-style factory.
    No IsaacSim dispatcher.
    No BehaviorScript patterns.
    """

    def __init__(self, env, P=None, kpi=None, logger=None):
        self.env = env
        self.P = P or build_factory_param_dict()
        self.kpi = kpi or PlatformKPI()
        self.logger = logger

        ap = self.P["auto_post"]
        pre = self.P["preproc"]
        plat_cfg = self.P["platform_clean"]

        # Platform Manager (handles tokens, initial platforms)
        self.platform_manager = PlatformManager(env, plat_cfg,
                                                self.P.get("demand", {}),
                                                logger)

        # Mapping Job to Platform
        self.mapper = JobPlatformMapper()

        # MACHINE INITIALIZATION
        self.preproc_m = PreprocServer(
            env,
            capacity=pre["servers"],
            proc_time=self._t_preproc()
        ).resource

        self.printers = [
            PrinterMachine(env, idx+1,
                        self.P["print"]["t_print_per_platform_min"]).resource
            for idx in range(len(self.P["print"]["printers"]))
        ]


        self.washers_m1 = WasherM1(env, ap["washers_m1"],
                                   ap["t_wash1_min"]).resource

        self.washers_m2 = WasherM2(env, ap["washers_m2"],
                                   ap["t_wash2_min"]).resource

        self.dryers = Dryer(env, ap["dryers"],
                            ap["t_dry_min"]).resource

        self.uv = UVUnit(env, ap["uv_units"],
                         ap["t_uv_min"]).resource

        self.platform_wash = PlatformWasher(
            env,
            plat_cfg["washers"],
            plat_cfg["wash_time_min"]
        ).resource

        # QUEUES
        self.stacker = simpy.Store(env, capacity=9999)


        # Preprocessing Manager (handles mapping and ready queue)
        self.preprocessing = PreprocessingManager(
            env,
            self.preproc_m,
            self._t_preproc,
            self.platform_manager,
            self.mapper,
            self.kpi,
            logger,
            #self.pre_in,
            # self.ready,
        )

        # START PROCESSES
        self.env.process(self.preprocessing.run())     # Preproc worker loop
        self.env.process(self.print_pipeline())        # Full platform pipeline

        if self.logger:
            self.logger.log_event(
                "FactoryValidation",
                f"[CHECK] Printers={len(self.printers)}, "
                f"WasherM1 cap={self.washers_m1.capacity}, "
                f"WasherM2 cap={self.washers_m2.capacity}, "
                f"Dryers cap={self.dryers.capacity}, "
                f"UV cap={self.uv.capacity}, "
                # 초기 플랫폼 개수는 clean_platforms.items 길이로 확인
                f"Platform tokens={len(self.platform_manager.clean_platforms.items)}"
            )
        else:
            print("[VALIDATION] Factory Resource Init OK:")
            print(f" - Printers: {len(self.printers)}")
            print(f" - Washer M1 capacity: {self.washers_m1.capacity}")
            print(f" - Washer M2 capacity: {self.washers_m2.capacity}")
            print(f" - Dryers capacity: {self.dryers.capacity}")
            print(f" - UV Units: {self.uv.capacity}")
            print(f" - Initial platforms: {len(self.platform_manager.clean_platforms.items)}")



  
    # Preprocess time calculation
    def _t_preproc(self):
        """Compute total preprocessing time."""
        pre = self.P["preproc"]
        return (
            pre["healing_time_per_platform_min"]
            + pre["placement_time_per_platform_min"]
            + pre["support_time_per_platform_min"]
        )


    # Submit Job -> enter preprocessing queue
    def submit_job(self, job):
        """
        Manager calls this to send job into the factory pipeline.
        """
        item = {
            "job_id": job.id_job,
            "items": job.list_items,
            "created_at": self.env.now
        }
        return self.preprocessing.pre_in.put(item)


    # Main pipeline
    def print_pipeline(self):
        while True:
            item = yield self.preprocessing.ready.get()

            job_id = item["job_id"]
            platform_id = item["platform"]

            if self.logger:
                self.logger.log_event(
                    "Factory",
                    f"[PIPELINE START] Job {job_id} Platform {platform_id}"
                )

            # 1) PRINT
            req = self.printers[0].request()
            yield req
            yield self.env.timeout(self.P["print"]["t_print_per_platform_min"])

            # 2) WASH 1
            req = self.washers_m1.request()
            yield req
            yield self.env.timeout(self.P["auto_post"]["t_wash1_min"])

            # 3) WASH 2
            req = self.washers_m2.request()
            yield req
            yield self.env.timeout(self.P["auto_post"]["t_wash2_min"])

            # 4) DRY
            req = self.dryers.request()
            yield req
            yield self.env.timeout(self.P["auto_post"]["t_dry_min"])

            # 5) UV
            req = self.uv.request()
            yield req
            yield self.env.timeout(self.P["auto_post"]["t_uv_min"])

            # 6) PLATFORM WASH
            req = self.platform_wash.request()
            yield req
            yield self.env.timeout(self.P["platform_clean"]["wash_time_min"])

            # PUSH TO STACKER
            yield self.stacker.put(item)

            if self.logger:
                self.logger.log_event(
                    "Factory",
                    f"[PIPELINE END] Platform {platform_id} finished"
                )

