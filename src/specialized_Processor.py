from base_Processor import Worker, Machine
from config_SimPy import *


class Worker_Inspect(Worker):
    def __init__(self, id_worker):
        super().__init__(
            id_worker, f"Inspector_{id_worker}", PROC_TIME_INSPECT)


class Mach_3DPrint(Machine):
    def __init__(self, id_machine):
        super().__init__(id_machine, "Proc_Build",
                         f"3DPrinter_{id_machine}", PROC_TIME_BUILD, CAPACITY_MACHINE_BUILD)


class Mach_Wash(Machine):
    def __init__(self, id_machine):
        super().__init__(id_machine, "Proc_Wash",
                         f"Washer_{id_machine}", PROC_TIME_WASH, CAPACITY_MACHINE_WASH)


class Mach_Dry(Machine):
    def __init__(self, id_machine):
        super().__init__(id_machine, "Proc_Dry",
                         f"Dryer_{id_machine}", PROC_TIME_DRY, CAPACITY_MACHINE_DRY)
