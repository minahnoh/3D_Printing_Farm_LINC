import simpy
from config_SimPy import *


class Worker:
    """
    Worker class to represent a worker in the manufacturing process
    One type of processor in the simulation

    Attributes:
        type_processor (str): Type of processor (Worker)
        id_worker (int): Worker ID
        name_worker (str): Worker name
        available_status (bool): Worker availability status
        working_job (Job): Job currently being processed
        processing_time (int): Time taken to process a job
        busy_time (int): Total time spent processing jobs
        last_status_change (int): Time of last status change
    """

    def __init__(self, id_worker, name_worker, processing_time):
        self.type_processor = "Worker"
        self.id_worker = id_worker
        self.name_worker = name_worker
        self.available_status = True
        self.working_job = None
        self.processing_time = processing_time
        self.busy_time = 0
        self.last_status_change = 0


class Machine:
    """
    Machine class to represent a machine in the manufacturing process
    One type of processor in the simulation

    Attributes:
        type_processor (str): Type of processor (Machine)
        id_machine (int): Machine ID
        name_process (str): Process name
        name_machine (str): Machine name
        available_status (bool): Machine availability status
        list_working_jobs (list): List of jobs currently being processed
        capacity_jobs (int): Maximum number of jobs that can be processed simultaneously
        processing_time (int): Time taken to process a job
        busy_time (int): Total time spent processing jobs
        last_status_change (int): Time of last status change
        allows_job_addition_during_processing (bool): Flag to allow job addition during processing
    """

    def __init__(self, id_machine, name_process, name_machine, processing_time, capacity_jobs=1):
        self.type_processor = "Machine"
        self.id_machine = id_machine
        self.name_process = name_process
        self.name_machine = name_machine
        self.available_status = True
        self.list_working_jobs = []
        self.capacity_jobs = capacity_jobs
        self.processing_time = processing_time
        self.busy_time = 0
        self.last_status_change = 0
        self.allows_job_addition_during_processing = False

# ============================================================
#   NEW: Concrete Machine Types (Washer, Dryer, UV, Preproc...)
# ============================================================

class WasherM1(Machine):
    def __init__(self, env, capacity, wash_time):
        super().__init__(
            id_machine=101,
            name_process="wash_m1",
            name_machine="Washer_M1",
            processing_time=wash_time,
            capacity_jobs=capacity
        )
        self.resource = ProcessorResource(env, self)


class WasherM2(Machine):
    def __init__(self, env, capacity, wash_time):
        super().__init__(
            id_machine=102,
            name_process="wash_m2",
            name_machine="Washer_M2",
            processing_time=wash_time,
            capacity_jobs=capacity
        )
        self.resource = ProcessorResource(env, self)


class Dryer(Machine):
    def __init__(self, env, capacity, dry_time):
        super().__init__(
            id_machine=103,
            name_process="dry",
            name_machine="Dryer",
            processing_time=dry_time,
            capacity_jobs=capacity
        )
        self.resource = ProcessorResource(env, self)


class UVUnit(Machine):
    def __init__(self, env, capacity, uv_time):
        super().__init__(
            id_machine=104,
            name_process="uv",
            name_machine="UV_Unit",
            processing_time=uv_time,
            capacity_jobs=capacity
        )
        self.resource = ProcessorResource(env, self)


class PreprocServer(Machine):
    def __init__(self, env, capacity, proc_time):
        super().__init__(
            id_machine=105,
            name_process="preproc",
            name_machine="Preprocessing_Server",
            processing_time=proc_time,
            capacity_jobs=capacity
        )
        self.resource = ProcessorResource(env, self)


class PrinterMachine(Machine):
    def __init__(self, env, printer_id, print_time):
        super().__init__(
            id_machine=200 + printer_id,
            name_process="print",
            name_machine=f"Printer_{printer_id}",
            processing_time=print_time,
            capacity_jobs=1
        )
        self.resource = ProcessorResource(env, self)


class PlatformWasher(Machine):
    def __init__(self, env, capacity, wash_time):
        super().__init__(
            id_machine=106,
            name_process="platform_wash",
            name_machine="PlatformWasher",
            processing_time=wash_time,
            capacity_jobs=capacity
        )
        self.resource = ProcessorResource(env, self)




class ProcessorResource(simpy.Resource):
    """
    Integrated processor (Machine, Worker) resource management class that inherits SimPy Resource

    Attributes: 
        processor_type (str): Type of processor (Machine/Worker)
        id (int): Processor ID
        name (str): Processor name
        allows_job_addition_during_processing (bool): Flag to allow job addition during processing
        current_jobs (list): List of jobs currently being processed (Machines)
        current_job (Job): Job currently being processed (Worker)
        processing_time (int): Time taken to process a job
        processing_started (bool): Flag to prevent further resource allocation after processing starts

    """

    def __init__(self, env, processor):
        # Check processor type and set properties
        self.processor_type = getattr(processor, 'type_processor', 'Unknown')

        # Set capacity - Machine uses capacity_jobs, Worker always 1
        if self.processor_type == "Machine":
            capacity = getattr(processor, 'capacity_jobs', 1)
            self.id = getattr(processor, 'id_machine', 0)
            self.name = getattr(processor, 'name_machine', 'Machine')
            # Flag for allowing job addition during processing
            self.allows_job_addition_during_processing = getattr(
                processor, 'allows_job_addition_during_processing', True)
            # Current jobs being processed
            self.current_jobs = []
        elif self.processor_type == "Worker":
            capacity = 1  # Worker always processes one job at a time
            self.id = getattr(processor, 'id_worker', 0)
            self.name = getattr(processor, 'name_worker', 'Worker')
            # Worker never allows job addition during processing
            self.allows_job_addition_during_processing = False
            # Current job being processed
            self.current_job = None
            self.current_jobs = []  # Added for consistency

        # Initialize Resource
        super().__init__(env, capacity=capacity)

        self.processor = processor
        self.processing_time = getattr(processor, 'processing_time', 10)

        # Flag to prevent further resource allocation after processing starts
        self.processing_started = False

    def request(self, *args, **kwargs):
        """
        Override resource request - Check if addition during processing is allowed
        """
        # If already processing and addition not allowed, reject request
        if self.processing_started and not self.allows_job_addition_during_processing:
            # Return a dummy event that mimics SimPy request but waits indefinitely
            dummy_event = self._env.event()
            dummy_event.callbacks.append(
                lambda _: None)  # Add callback to set to infinite wait state
            return dummy_event

        # Set flag when job is first assigned to resource
        if not self.processing_started and self.count == 0:
            self.processing_started = True

        # Process basic request
        return super().request(*args, **kwargs)

    def release(self, request):
        """
        Override resource release - Handle job completion
        """
        result = super().release(request)

        # Reset processing flag when all jobs are complete
        if self.count == 0:
            self.processing_started = False
            if self.processor_type == "Machine":
                self.current_jobs = []
            else:  # Worker
                self.current_job = None
                self.current_jobs = []

        return result

    @property
    def is_available(self):
        """Check if processor is available"""
        # Not available if processing and additions not allowed
        if self.processing_started and not self.allows_job_addition_during_processing:
            return False

        # Available if capacity has room
        return self.count < self.capacity  # Use count attribute instead of count()

    def start_job(self, job):
        """Process job start"""
        if self.processor_type == "Machine":
            # Add job to Machine
            self.current_jobs.append(job)
        else:  # Worker
            # Set Worker's current job
            self.current_job = job
            self.current_jobs = [job]  # Add to list for consistency

        # Set workstation info in job
        if self.processor_type == "Machine":
            job.workstation["Machine"] = self.id
        else:  # Worker
            job.workstation["Worker"] = self.id

    def get_jobs(self):
        """Return list of currently processing jobs"""
        if self.processor_type == "Machine":
            return self.current_jobs
        else:  # Worker
            return [self.current_job] if self.current_job else []

    def finish_jobs(self):
        """Process job completion"""
        jobs = self.get_jobs()

        if self.processor_type == "Machine":
            self.current_jobs = []
        else:  # Worker
            self.current_job = None
            self.current_jobs = []

        return jobs
