import simpy
from typing import Optional
from log_SimPy import Logger


class BaseProcessor:
    def __init__(self, env, name,capacity, proc_time, logger= None):
        self.env = env
        self.name = name
        self.capacity = max(1, int(capacity))
        self.proc_time = float(proc_time)
        self.resource = simpy.Resource(env, capacity=self.capacity)
        self.logger = logger

        # tracking
        self.busy_time = 0.0
        self.total_jobs = 0

   
    def _log(self, job_id: str, text: str):
        """Calls external logger in the required format."""
        if self.logger:
            self.logger.log_event(
                event_type=self.name,
                message=f"{job_id} | {text}"
            )


    def seize(self, job_id: str):
        """Request a machine/worker and wait for availability."""

        req = self.resource.request()
        yield req
        return req
    

    def work(self, job_id: str, duration: Optional[float] = None):
        """Simulate job processing inside processor."""

        duration = duration if duration is not None else self.proc_time
        start_time = self.env.now

        self._log(job_id, f"Processing started ({duration:.1f} min)")

        yield self.env.timeout(duration)

        elapsed = self.env.now - start_time
        self.busy_time += elapsed
        self.total_jobs += 1

        self._log(job_id, f"Processing finished (elapsed={elapsed:.1f} min)")


    def release(self, req, job_id: str):
        self.resource.release(req)

   
    def process(self, job_id, duration: Optional[float] = None):
        """Full micro-cycle: seize → work → release."""
        req = yield from self.seize(job_id)
        yield from self.work(job_id, duration)
        self.release(req, job_id)

    def utilization(self, sim_time: float) -> float:
        """Return utilization ratio between 0 and 1."""
        if sim_time <= 0:
            return 0.0
        return self.busy_time / (sim_time * self.capacity)

    def summary(self) -> dict:
        """Return a JSON-safe stats summary."""
        return {
            "name": self.name,
            "capacity": self.capacity,
            "processed_jobs": self.total_jobs,
            "busy_time_min": round(self.busy_time, 2),
        }