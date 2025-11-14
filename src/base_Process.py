from base_Job import JobStore
from base_Processor import ProcessorResource

class Process:
    """
    Base manufacturing process class for SimPy simulation

    Attributes:
        name_process (str): Process identifier
        env (simpy.Environment): Simulation environment
        logger (Logger): Event logger
        list_processors (list): List of processors (Machines, Workers)
        job_store (JobStore): Job queue management
        processor_resources (dict): Processor resources (Machine, Worker)
        completed_jobs (list): List of completed jobs
        next_process (Process): Next process in the flow
        resource_trigger (simpy.Event): Resource trigger event
        job_added_trigger (simpy.Event): Job added trigger event
        process (simpy.Process): Main process execution
    """

    def __init__(self, name_process, env, logger=None):
        self.name_process = name_process
        self.env = env
        self.logger = logger
        self.list_processors = []  # Processor list

        # Implement queue with JobStore (Inherits SimPy Store)
        self.job_store = JobStore(env, f"{name_process}_JobStore")

        # Processor resource management
        self.processor_resources = {}  # {processor_id: ProcessorResource}

        # Track completed jobs
        self.completed_jobs = []

        # Next process
        self.next_process = None

        # Add new events
        self.resource_trigger = env.event()
        self.job_added_trigger = env.event()

        # Start simulation process
        self.process = env.process(self.run())

        # if self.logger:
        #     self.logger.log_event(
        #         "Process", f"Process {self.name_process} created")

    def connect_to_next_process(self, next_process):
        """Connect directly to next process. Used for process initialization."""
        self.next_process = next_process
        # if self.logger:
        #     self.logger.log_event(
        #         "Process", f"Process {self.name_process} connected to {next_process.name_process}")

        #  validation
        # print(f"Process {self.name_process} connected to {next_process.name_process}")

    def register_processor(self, processor):
        """Register processor (Machine or Worker). Used for process initialization."""
        # Add to processor list
        self.list_processors.append(processor)

        # Create ProcessorResource (integrated resource management)
        processor_resource = ProcessorResource(self.env, processor)

        # Determine id based on processor type
        if processor.type_processor == "Machine":
            processor_id = f"Machine_{processor.id_machine}"
        else:  # Worker
            processor_id = f"Worker_{processor.id_worker}"

        # Store resource
        self.processor_resources[processor_id] = processor_resource

        # if self.logger:
        #     self.logger.log_event(
        #         "Resource", f"Registered {processor.type_processor} {processor_name} to process {self.name_process}")
        
        # validation
        #if processor_resource.processor_type == "Machine":
        #   print("Resource", f"Registered {processor_resource.name} | Capacity {processor_resource.capacity} | Processing time {processor_resource.processing_time} | to process {self.name_process}")
        #else:
        #   print("Resource", f"Registered {processor_resource.name} | Processing time {processor_resource.processing_time} |to process {self.name_process}")
    
    def add_to_queue(self, job):
        """Add job to queue"""
        job.time_waiting_start = self.env.now
        job.workstation["Process"] = self.name_process

        # Add job to JobStore
        self.job_store.put(job)

        # Trigger job added event
        self.job_added_trigger.succeed()
        # Create new trigger immediately
        self.job_added_trigger = self.env.event()

        # if self.logger:
        #   self.logger.log_event(
        #       "Queue", f"Added job {job.id_job} to {self.name_process} queue. Queue length: {self.job_store.size}")

    def run(self):
        """Event-based process execution"""
        # print(f"[DEBUG] {self.name_process}: run started")

        # Initial check (important!): Check if queue already has jobs at start
        if not self.job_store.is_empty:
            # print(
            #     f"[DEBUG] {self.name_process}: jobs in initial queue, attempting resource allocation")
            yield self.env.process(self.seize_resources())

        while True:
            # print(f"[DEBUG] {self.name_process}: waiting for events")
            # Wait for events: until job is added or resource is released
            yield self.job_added_trigger | self.resource_trigger
            # print(
            #     f"[DEBUG] {self.name_process}: event triggered! time={self.env.now}")

            # If there are jobs in queue, attempt to allocate resources
            if not self.job_store.is_empty:
                yield self.env.process(self.seize_resources())

    def seize_resources(self):
        """
        Allocate available resources (machines or workers) to jobs in queue
        """
        #print(
        #   f"[DEBUG] {self.name_process}: seize_resources called, time={self.env.now}")

        # Find available processors
        available_processors = [
            res for res in self.processor_resources.values() if res.is_available]

        #print(
        #   f"[DEBUG] {self.name_process}: available processors={len(available_processors)}")
        # Debug: Print status of each resource
        #for res_id, res in self.processor_resources.items():
        #    print(
        #        f"[DEBUG] Processor {res_id}: is_available={res.is_available}, capacity={res.capacity}")

        # If queue is empty or no available processors, stop
        if self.job_store.is_empty or not available_processors:
            # print(
            #     f"[DEBUG] {self.name_process}: job allocation stopped - queue empty={self.job_store.is_empty}, no processors={not available_processors}")
            return

        # List of jobs assigned to each processor
        processor_assignments = []

        # Try processing with all processors
        for processor_resource in available_processors:
            # print(
            #     f"[DEBUG] {self.name_process}: attempting to process with {processor_resource.name}")
            # Determine number of jobs to assign (up to capacity)
            remaining_capacity = processor_resource.capacity - processor_resource.count
            jobs_to_assign = []

            # Assign jobs
            try:
                for i in range(min(remaining_capacity, self.job_store.size)):
                    if not self.job_store.is_empty:
                        # print(
                        #    f"[DEBUG] {self.name_process}: attempting to get capacity {i+1}")
                        job = yield self.job_store.get()
                        # print(
                        #    f"[DEBUG] {self.name_process}: retrieved job {job.id_job}")
                        jobs_to_assign.append(job)
            except Exception as e:
                # Continue if unable to get job from JobStore
                print(f"[ERROR] {self.name_process}: failed to get job: {e}")

            # Assign jobs to processor
            if jobs_to_assign:
                processor_assignments.append(
                    (processor_resource, jobs_to_assign))
                # yield self.env.process(self.delay_resources(processor_resource, jobs_to_assign))

        # Process jobs with assigned processors in parallel
        for processor_resource, jobs in processor_assignments:
            self.env.process(self.delay_resources(processor_resource, jobs))

    def delay_resources(self, processor_resource, jobs):
        """
        Process jobs with processor (integrated for Machine, Worker)
        Takes processing time into account 

        Args:
            processor_resource (ProcessorResource): Processor resource (Machine, Worker)
            jobs (list): List of jobs to process        
        """
        # Record time and register resources for all jobs
        for job in jobs:
            job.time_waiting_end = self.env.now

            # Register job with processor
            processor_resource.start_job(job)

            # if self.logger:
            #  self.logger.log_event(
            #      "Processing", f"Assigning job {job.id_job} to {processor_resource.name}")
               
            # Record job start time
            job.time_processing_start = self.env.now

            # Record job processing history
            process_step = self.create_process_step(job, processor_resource)    
            if not hasattr(job, 'processing_history'):
                job.processing_history = []
            job.processing_history.append(process_step)

        # Request processor resource
        request = processor_resource.request()
        yield request

        # Calculate and wait for processing time
        processing_time = processor_resource.processing_time
        yield self.env.timeout(processing_time)

        # Special processing (if needed)
        if hasattr(self, 'apply_special_processing'):
            self.apply_special_processing(processor_resource.processor, jobs)

        # Process job completion
        for job in jobs:
            job.time_processing_end = self.env.now

            # Update job history
            for step in job.processing_history:
                if step['process'] == self.name_process and step['end_time'] is None:
                    step['end_time'] = self.env.now
                    step['duration'] = self.env.now - step['start_time'] 

            # Track completed jobs
            self.completed_jobs.append(job)

            # Log record
            # if self.logger:
            #  self.logger.log_event(
            #      "Processing", f"Completed processing job {job.id_job} on {processor_resource.name}")
            # validaiton code
            # if self.logger:
            #  self.logger.log_event(
            #       "Validation", f"{self.name_process}: {processor_resource.name} started job{job.id_job} at {job.time_processing_start} and finished at {self.env.now}, duration:{step['duration']}"
            #       )
            # else:
            #   print("Created process step:", process_step)   

            # Send job to next process
            self.send_job_to_next(job)

        # Release resources
        self.release_resources(processor_resource, request)

    def release_resources(self, processor_resource, request):
        """
        Release processor resources and process job completion

        Args:
            processor_resource (ProcessorResource): Processor resource (Machine, Worker)
            request (simpy.Request): Resource request 

        """
        # Release processor resource
        processor_resource.release(request)
        processor_resource.finish_jobs()

        # Trigger resource release event (for event-based approach)
        if hasattr(self, 'resource_trigger'):
            self.resource_trigger.succeed()
            # Create new trigger immediately
            self.resource_trigger = self.env.event()

        # if self.logger:
        #   self.logger.log_event(
        #       "Resource", f"Released {processor_resource.name} in {self.name_process}")

    def create_process_step(self, job, processor_resource):
        """Create process step for job history"""
        return {
            'process': self.name_process,
            'resource_type': processor_resource.processor_type,
            'resource_id': processor_resource.id,
            'resource_name': processor_resource.name,
            'start_time': job.time_processing_start,
            'end_time': None,
            'duration': None
        }

    def send_job_to_next(self, job):
        """Send job to next process"""
        if self.next_process:
        # if self.logger:
        #       self.logger.log_event(
        #           "Process Flow", f"Moving job {job.id_job} from {self.name_process} to {self.next_process.name_process}")
            # Add job to next process queue
            self.next_process.add_to_queue(job)
            return True
        else:
            # Final process or no next process set
            if self.logger:
               self.logger.log_event(
                   "Process Flow", f"Job {job.id_job} completed at {self.name_process} (final process)")

            # vadlidation code to check the remaining defective item
            # print("======================================================================")      
            # print(f"The number of remaining defecitve item: {len(self.defective_items)}")       
            # print("======================================================================")    

            return False

            