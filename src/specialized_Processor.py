import simpy
from base_Processor import BaseProcessor


class Printer(BaseProcessor):
    def __init__(self, env: simpy.Environment, capacity: int, proc_time: float):
        super().__init__(env, name="Printer", capacity=capacity, proc_time=proc_time)


class WashM1(BaseProcessor):
    def __init__(self, env: simpy.Environment, capacity: int, wash_time: float):
        super().__init__(env, name="WashM1", capacity=capacity, proc_time=wash_time)


class WashM2(BaseProcessor):
    def __init__(self, env: simpy.Environment, capacity: int, wash_time: float):
        super().__init__(env, name="WashM2", capacity=capacity, proc_time=wash_time)


class Dryer(BaseProcessor):
    def __init__(self, env: simpy.Environment, capacity: int, dry_time: float):
        super().__init__(env, name="Dryer", capacity=capacity, proc_time=dry_time)


class UVStation(BaseProcessor):
    def __init__(self, env: simpy.Environment, capacity: int, uv_time: float):
        super().__init__(env, name="UV", capacity=capacity, proc_time=uv_time)


class AMR(BaseProcessor):
    def __init__(self, env: simpy.Environment, capacity: int, default_travel_time: float = 0.0):
        super().__init__(env, name="AMRPool", capacity=capacity, proc_time=default_travel_time)


class Worker(BaseProcessor):
    def __init__(self, env: simpy.Environment, capacity: int, default_task_time: float = 0.0):
        super().__init__(env, name="Worker", capacity=capacity, proc_time=default_task_time)


class PlatformWasher(BaseProcessor):
    def __init__(self, env: simpy.Environment, capacity: int, wash_time: float):
        super().__init__(env, name="PlatformWasher", capacity=capacity, proc_time=wash_time)
