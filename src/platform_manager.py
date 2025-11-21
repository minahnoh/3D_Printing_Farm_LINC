import simpy

class PlatformManager:
    def __init__(self, env, plat_cfg, demand_cfg, logger=None):
        self.env = env
        self.logger = logger
        
        demand_init = int(demand_cfg.get("initial_platforms", 0) or 0)
        clean_init = int(plat_cfg.get("initial_platforms", 0) or 0)
        self.initial_plat = max(demand_init, clean_init) or 1

        self.clean_platforms = simpy.Store(env, capacity=self.initial_plat)
        self.tokens = []

        for i in range(self.initial_plat):
            token = {"id": f"PLAT-{i+1}", "cycle": 0}
            self.tokens.append(token)
            self.clean_platforms.items.append(token)

    def get_clean_platform(self):
        def _proc():
            token = yield self.clean_platforms.get()
            return token
        return self.env.process(_proc())

    def return_platform(self, token):
        def _proc():
            yield self.clean_platforms.put(token)
        return self.env.process(_proc())

    def total_platforms(self):
        return len(self.tokens)
