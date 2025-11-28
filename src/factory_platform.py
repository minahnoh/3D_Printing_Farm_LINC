import simpy
from typing import Any, Dict, Optional
from config_SimPy import CFG, FACTORY_PLATFORM_CLEAN, demand_cfg
from KPI import KPI
from base_Job import Job
from platform_manager import PlatformManager
from specialized_Process import (
    Preprocess,
    Print,
    AutoPost,
    ManualPost,
    PlatformClean,
)
from specialized_Processor import AMR
from work_shift import WorkShiftManager
from config_process import PROC_DEFAULT

class Factory:
    """
    Stage-based factory wrapper.

    This class wires together:
        - PlatformManager
        - Preprocess / Print / AutoPost / ManualPost / PlatformClean stages
        - KPI and Logger
    """

    def __init__(self,env,cfg: Dict[str, Any] = CFG, kpi: Optional[KPI] = None, logger = None) :
        self.env = env
        self.cfg = cfg
        self.logger = logger
        self.trace = []

        # Shared KPI container (platform-level)
        self.kpi = kpi if kpi is not None else KPI()

        # PlatformManager uses platform_clean & demand configuration
        self.platform_manager = PlatformManager(env, plat_cfg=FACTORY_PLATFORM_CLEAN, demand_cfg=demand_cfg, logger=self.logger)

        # entire processes parameter
        process_cfg = PROC_DEFAULT

        # Stacker: step between autopost and manualpost
        self.stacker = simpy.Store(env, capacity=10_000)

        stacker_cfg = process_cfg["stacker_guard"]
        # stacker_guard controls the upstream printing and post-processing flow so that the total downstream WIP never exceeds a safe capacity limit.
        self.stacker_guard_enabled = bool(stacker_cfg.get("enabled")) 
        self.stacker_guard_limit = int(stacker_cfg.get("max_platforms"))

        #AMR
        auto_cfg = process_cfg["auto_post"]
        self.amr_pool = AMR(env, capacity=auto_cfg.get("amr_count"), default_travel_time=0.0)

        # Manual move
        manual_move_cfg = process_cfg["manual_move"]
        self.manual_move_speed = manual_move_cfg["speed_m_per_s"]
        self.manual_move_dist = manual_move_cfg["dist_m"]

        # workter/AMR shift
        manual_shift_cfg = self.cfg["manual_ops"].get("work_shift", {})
        auto_shift_cfg   = self.cfg["auto_post"].get("work_shift_amr", manual_shift_cfg)

        self.shift_manual = WorkShiftManager(env, manual_shift_cfg)
        self.shift_amr    = WorkShiftManager(env, auto_shift_cfg)


        # Instantiate all stages
        self.preproc_stage          = Preprocess(env, cfg=self.cfg, platM=self.platform_manager, kpi=self.kpi, logger=self.logger)
        self.print_stage            = Print(env, cfg=self.cfg["print"], kpi=self.kpi, logger=self.logger)
        self.auto_post_stage        = AutoPost(env, cfg=self.cfg, kpi=self.kpi, logger=self.logger, amr_pool=self.amr_pool, stacker=self.stacker, shift_amr=self.shift_amr)
        self.manual_stage           = ManualPost(env, cfg=self.cfg, kpi=self.kpi, logger=self.logger, shift=self.shift_manual)
        self.platform_clean_stage   = PlatformClean(env, cfg=self.cfg, platM=self.platform_manager, kpi=self.kpi, logger=self.logger)

        if self.logger:
            self.logger.log_event(
                "Factory",
                "Stage-based factory initialized "
                "(Preprocess → Print → AutoPost → ManualPost → PlatformClean).",
            )

  
    # Public API
    def submit_job(self, job: Job):
        """
        Entry point used by Manager.

        This schedules a SimPy process that runs the full pipeline
        for the given Job.
        """
        if self.logger:
            self.logger.log_event(
                "Factory",
                f"Submit job {job.id_job} with {len(job.list_items)} items to pipeline",
            )
        # Start the pipeline for this job
        return self.env.process(self._job_flow(job))

    def _log_resource(self, job, resource_type: str, t_start: float, t_end: float):
        """Gantt용 trace 한 줄 추가 (자원 기준)."""
        if t_end <= t_start:
            return
        job_id = getattr(job, "id_job", None)
        plat_id = getattr(job, "platform_id", None)

        self.trace.append({
            "job_id": job_id,
            "platform_id": plat_id,
            "id": f"Job {job_id}" if job_id is not None else str(plat_id or "UNKNOWN"),
            "resource_type": resource_type,   
            "stage": resource_type,          
            "t0": float(t_start),
            "t1": float(t_end),
        })



    # Internal pipeline
    def _job_flow(self, job: Job):
        """
        Full stage pipeline for a single job:

            Preprocess → Print → AutoPost → ManualPost → PlatformClean
        """

        # 1) Preprocess (get clean platform, prep operations)
        platform_token = yield self.env.process(self.preproc_stage.run(job))

        # 2) Print
        yield self.env.process(self.print_stage.run(job))

        # 3) Automated post-process (wash1, wash2, dry, UV)
        yield self.env.process(self.auto_post_stage.run(job))

        # 4) Manual finishing
        yield self.env.process(self.manual_stage.run(job))

        # 5) Platform cleaning
        yield self.env.process(self.platform_clean_stage.run(job, platform_token))

        if self.logger:
            self.logger.log_event(
                "Factory",
                f"Job {job.id_job} fully completed through stage pipeline.",
            )


