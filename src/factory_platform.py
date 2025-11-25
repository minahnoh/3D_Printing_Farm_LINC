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

        # Shared KPI container (platform-level)
        self.kpi = kpi if kpi is not None else KPI()

        # PlatformManager uses platform_clean & demand configuration
        self.platform_manager = PlatformManager(env, plat_cfg=FACTORY_PLATFORM_CLEAN, demand_cfg=demand_cfg, logger=self.logger)

        # Instantiate all stages
        self.preproc_stage          = Preprocess(env, cfg=self.cfg, platM=self.platform_manager, kpi=self.kpi, logger=self.logger)
        self.print_stage            = Print(env, cfg=self.cfg["print"], kpi=self.kpi, logger=self.logger)
        self.auto_post_stage        = AutoPost(env, cfg=self.cfg, kpi=self.kpi, logger=self.logger)
        self.manual_stage           = ManualPost(env, cfg=self.cfg, kpi=self.kpi, logger=self.logger)
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
