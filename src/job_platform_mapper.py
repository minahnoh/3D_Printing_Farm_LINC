class JobPlatformMapper:
    def __init__(self):
        self.job_to_platform = {}
        self.platform_to_job = {}

    def assign(self, job_id, platform_id):
        if platform_id in self.platform_to_job:
            raise ValueError(f"Platform {platform_id} already assigned.")
        if job_id in self.job_to_platform:
            raise ValueError(f"Job {job_id} already has platform.")

        self.job_to_platform[job_id] = platform_id
        self.platform_to_job[platform_id] = job_id

    def get_platform(self, job_id):
        return self.job_to_platform.get(job_id)

    def get_job(self, platform_id):
        return self.platform_to_job.get(platform_id)
