import simpy
from typing import Dict, Any


class PlatformManager:
    """
    Platform resource manager.

    - Provides clean platforms on request (SimPy Store)
    - Assigns a platform to a Job and records the mapping
    - Releases platforms back to pool when finished and clears mapping
    """

    def __init__(self, env, plat_cfg, demand_cfg, logger=None):
        self.env = env
        self.logger = logger
        
        # Determine platform count (use whichever is larger: demand config or platform config)
        demand_init = int(demand_cfg.get("initial_platforms", 0) or 0)
        clean_init = int(plat_cfg.get("initial_platforms", 0) or 0)
        self.initial_plat = max(demand_init, clean_init) or 1

        # SimPy resource pool
        self.clean_platforms = simpy.Store(env, capacity=self.initial_plat)

        # Create platform tokens
        self.tokens = []
        for i in range(self.initial_plat):
            token = {"id": f"PLAT-{i+1}", "cycle": 0}
            self.tokens.append(token)
            self.clean_platforms.items.append(token)

        # Job <-> Platform mapping
        self.job_to_platform: Dict[int, str] = {}
        self.platform_to_job: Dict[str, int] = {}

        """
        if self.logger:
            self.logger.log_event("PlatformManager", f"Initialized with {self.initial_plat} platforms.")
        """

    def get_clean_platform(self):
        """Return next available clean platform."""
        
        """"
        if self.logger:
            self.logger.log_event("PlatformManager", "Platform request issued. Waiting for availability...")
        """
        return self.clean_platforms.get()


    def assign(self, job_id: int, platform_id) -> str:
        """Record that the specified job uses the given platform."""
        
        # Platform id normalization (can pass dict or string)
        platform_id = (
            platform_id.get("id")
            if isinstance(platform_id, dict)
            else str(platform_id)
        )

        # Safety checks
        if platform_id in self.platform_to_job:
            raise ValueError(f"[ERROR] Platform {platform_id} already assigned to another job.")
        if job_id in self.job_to_platform:
            raise ValueError(f"[ERROR] Job {job_id} already has a platform assigned.")

        # Record mapping
        self.job_to_platform[job_id] = platform_id
        self.platform_to_job[platform_id] = job_id

        ## Debugging
        if self.logger:
            self.logger.log_event(
                "PlatformManager",
                f"Assigned platform {platform_id} → Job {job_id}"
            )

        return platform_id


    def release_platform(self, token: Dict[str, Any]):
        """Return a platform to the pool and clear its job mapping."""
        platform_id = token.get("id")

        # Remove mapping
        job_id = self.platform_to_job.pop(platform_id, None)
        if job_id is not None:
            self.job_to_platform.pop(job_id, None)
        """
        if self.logger:
            self.logger.log_event(
                "PlatformManager",
                f"Platform {platform_id} released back to pool (previous job={job_id})"
            )
        """
        return self.env.process(self._put_clean_platform(token))


    def _put_clean_platform(self, token: Dict[str, Any]):
        """Internal helper: SimPy put operation."""
        yield self.clean_platforms.put(token)

        ## Debugging
        """
        if self.logger:
            self.logger.log_event(
                "PlatformManager",
                f"Platform {token['id']} is now available again."
            )
        """