class PrinterDispatcher:
    def __init__(self, env, ready_store, stacker_store, P, kpi, logger=None):
        self.env = env
        self.ready = ready_store
        self.stacker = stacker_store
        self.P = P
        self.kpi = kpi
        self.logger = logger

    def run(self):
        while True:
            item = yield self.ready.get()

            job_id = item["job_id"]
            platform_id = item["platform"]

            if self.logger:
                self.logger.log_event(
                    "Dispatcher",
                    f"[DISPATCH] Job {job_id}, Platform {platform_id} → stacker"
                )

            n_parts = self.P["print"].get("max_parts_per_platform", 8)
            payload = {
                "platform": platform_id,
                "n_parts": n_parts,
                "created_at": item.get("created_at", self.env.now),
                "arrived_stacker_at": self.env.now,
                "_platform_token": item["_platform_token"],
                "job_id": job_id
            }

            yield self.stacker.put(payload)

            self.kpi.started_platforms += 1
            self.kpi.finished_platforms += 1
            self.kpi.started_parts += n_parts
            self.kpi.completed_parts += n_parts
            self.kpi.max_stacker_wip = max(self.kpi.max_stacker_wip, len(self.stacker.items))
