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
from printer import*

class PlatformKPI:
    """
    Minimal KPI container for the platform-level Factory.
    """
    def __init__(self):
        self.started_parts = 0
        self.started_platforms = 0
        self.completed_parts = 0
        self.finished_platforms = 0
        self.scrapped_parts = 0

        self.printer_busy_min = 0.0
        self.wash1_busy_min = 0.0
        self.wash2_busy_min = 0.0
        self.dry_busy_min = 0.0
        self.uv_busy_min = 0.0
        self.preproc_busy_min = 0.0
        self.platform_wash_busy_min = 0.0
        self.manual_busy_min = 0.0
        self.pre_manual_busy_min = 0.0

        self.amr_travel_min = 0.0
        self.n_amr_moves = 0

        self.max_stacker_wip = 0
        self.n_preproc_jobs = 0
        self.n_print_jobs = 0
        self.sum_platform_lead_time_min = 0.0

        self.resin_used_kg = 0.0
        self.resin_cost_krw = 0.0

        self.wait_stats = defaultdict(float)

    def add_wait(self, key, delta):
        self.wait_stats[key] += float(delta)


class Factory:

    def __init__(self, env, P=None, kpi=None, logger=None):
        self.env = env
        self.P = P or build_factory_param_dict()
        self.logger = logger
        self.kpi = kpi or PlatformKPI()

        ap = self.P["auto_post"]
        pre = self.P["preproc"]
        plat_cfg = self.P["platform_clean"]

        # Platform Manager
        self.platform_manager = PlatformManager(env, plat_cfg, self.P.get("demand", {}), logger)

        # Job ↔ Platform Mapper
        self.mapper = JobPlatformMapper()

        # Machines
        self.washers_m1 = WasherM1(env, capacity=ap["washers_m1"], wash_time=ap["t_wash1_min"]).resource
        self.washers_m2 = WasherM2(env, capacity=ap["washers_m2"], wash_time=ap["t_wash2_min"]).resource
        self.dryers = Dryer(env, capacity=ap["dryers"], dry_time=ap["t_dry_min"]).resource
        self.uv = UVUnit(env, capacity=ap["uv_units"], uv_time=ap["t_uv_min"]).resource
        self.preproc = PreprocServer(env, capacity=pre["servers"], proc_time=self.t_preproc()).resource

        self.printers = [
            PrinterMachine(env, idx+1, self.P["print"]["print_time_min"]).resource
            for idx in range(len(self.P["print"]["printers"]))
        ]

        self.platform_wash = PlatformWasher(
            env,
            capacity=plat_cfg["washers"],
            wash_time=plat_cfg["wash_time_min"]
        ).resource

        # Preprocessing Manager
        self.preprocessing = PreprocessingManager(
            env,
            self.preproc,
            self.t_preproc,
            self.platform_manager,
            self.mapper,
            self.kpi,
            logger
        )

        # Printer Dispatcher 
        self.stacker = simpy.Store(env, capacity=100_000)

        self.dispatcher = PrinterDispatcher(
            env,
            self.preprocessing.ready,
            self.stacker,
            self.P,
            self.kpi,
            logger
        )

