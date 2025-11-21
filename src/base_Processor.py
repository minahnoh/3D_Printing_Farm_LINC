import simpy

# Base class for all factory processors (common behaviors
class BaseProcessor:
    """Base class providing shared behavior for all Factory machines."""
    def __init__(self, env, name, capacity=1, proc_time=0.0):
        self.env = env
        self.name = name
        self.capacity = capacity
        self.proc_time = proc_time
        self.resource = simpy.Resource(env, capacity=capacity)

    def process(self, item=None):
        """Default processing behavior: simple timeout."""
        yield self.env.timeout(self.proc_time)


# Preprocessing Equipment
class PreprocServer(BaseProcessor):
    def __init__(self, env, capacity, proc_time):
        super().__init__(env, "Preprocessing", capacity, proc_time)


# 3D Printer
class PrinterMachine(BaseProcessor):
    def __init__(self, env, printer_id, print_time):
        name = f"Printer-{printer_id}"
        super().__init__(env, name, capacity=1, proc_time=print_time)
        self.printer_id = printer_id


# Washing Machines (M1, M2)
class WasherM1(BaseProcessor):
    def __init__(self, env, capacity, wash_time):
        super().__init__(env, "Washer-M1", capacity, wash_time)


class WasherM2(BaseProcessor):
    def __init__(self, env, capacity, wash_time):
        super().__init__(env, "Washer-M2", capacity, wash_time)


# Drying Units
class Dryer(BaseProcessor):
    def __init__(self, env, capacity, dry_time):
        super().__init__(env, "Dryer", capacity, dry_time)



# UV Cure Units
class UVUnit(BaseProcessor):
    def __init__(self, env, capacity, uv_time):
        super().__init__(env, "UV-Unit", capacity, uv_time)




# Platform Washer
class PlatformWasher(BaseProcessor):
    def __init__(self, env, capacity, wash_time):
        super().__init__(env, "Platform-Washer", capacity, wash_time)
