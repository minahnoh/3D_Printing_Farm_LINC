import simpy

class PreprocessingManager:
    def __init__(self, env, preproc_resource, t_preproc_func, clean_platform_manager, mapper, kpi, logger=None):
        self.env = env
        self.preproc = preproc_resource
        self.t_preproc = t_preproc_func
        self.clean_plat = clean_platform_manager
        self.mapper = mapper
        self.kpi = kpi
        self.logger = logger

        self.pre_in = simpy.Store(env, capacity=9999)
        self.ready = simpy.Store(env, capacity=9999)

    def run(self):
        while True:
            item = yield self.pre_in.get()

            # 1) use preproc
            req = self.preproc.request()
            t0 = self.env.now
            yield req

            wait = self.env.now - t0
            self.kpi.add_wait("preproc", wait)

            t = self.t_preproc()
            self.kpi.preproc_busy_min += t
            yield self.env.timeout(t)

            # 2) clean platform
            token = yield self.clean_plat.get_clean_platform()
            token["cycle"] += 1
            platform_id = f"{token['id']}-R{token['cycle']}"

            item["_platform_token"] = token
            job_id = item.get("job_id", platform_id)
            item["job_id"] = job_id
            item["platform"] = platform_id

            # 3) mapping
            self.mapper.assign(job_id, platform_id)

            # validation: confirm correct Job ↔ Platform mapping
            if self.logger:
                self.logger.log_event("Preproc", f"Assigned {job_id} to {platform_id}")

            yield self.ready.put(item)
            self.kpi.n_preproc_jobs += 1
