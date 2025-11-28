from typing import Optional
from base_Job import Job
from base_Customer import OrderReceiver
from config_SimPy import PALLET_SIZE_LIMIT, POLICY_ORDER_TO_JOB
from factory_platform import Factory
from config_SimPy import CFG

class Manager(OrderReceiver):
    """
    The class that receives Orders from the Customer,
    converts them into Jobs, and forwards the Jobs to the stage-based Factory.

    Responsibilities:
        - Implement the OrderReceiver interface (receive_order).
        - Record the start time of each Order.
        - Split Orders into one or more Jobs based on pallet capacity
          and POLICY_ORDER_TO_JOB.
        - Submit each Job to the Factory via `factory.submit_job(job)`.

    Attributes:
        env:    SimPy environment.
        logger: Logger instance used to record events (can be None).
        next_job_id: Sequential counter for assigning unique job IDs.
        completed_orders: List of Orders that have finished all processing
        factory: Stage-based Factory instance that runs the full pipeline
                 (Preprocess → Print → AutoPost → ManualPost → PlatformClean).
    """

    def __init__(self, env, logger = None) :
        self.env = env
        self.logger = logger

        # job ID counter
        self.next_job_id: int = 1

        # keep track of fully completed orders
        self.completed_orders = []

        # Create the stage-based factory (wrapping PlatformManager + stages + KPI)
        self.cfg = CFG
        self.factory = Factory(env=self.env, cfg=self.cfg, logger=self.logger)

        if self.logger:
            self.logger.log_event(
                "Manager",
                "Stage-based Factory created and ready to receive jobs.",
            )


    def receive_order(self, order):
        """
        Process an incoming Order from the Customer.
        """

        if self.logger:
            self.logger.log_event(
                "Order",
                f"Received Order {order.id_order} with {order.num_patients} patients",
            )

        # Record Order start time in simulation time
        order.time_start = self.env.now

        # Convert Order -> Jobs and push them into the factory
        self._create_jobs_for_factory(order)

        return order

    
    def _create_jobs_for_factory(self, order):
        """
        Convert an Order into Jobs based on the batching policy.

        Current behavior:
            For each patient in the Order:
                - Let N = number of items (aligners) for that patient.

                - If N <= PALLET_SIZE_LIMIT:
                      -> create a single Job with all items.

                - If N > PALLET_SIZE_LIMIT and
                  POLICY_ORDER_TO_JOB == "MAX_PER_JOB":
                      -> split into multiple Jobs, each containing
                         at most PALLET_SIZE_LIMIT items.

        Each created Job is immediately submitted to the Factory via
        `self.factory.submit_job(job)`.
        """
        for patient in order.list_patients:
            patient_items = patient.list_items
            num_items = len(patient_items)

            if num_items <= PALLET_SIZE_LIMIT:
                job = Job(self.next_job_id, patient_items)
                self.next_job_id += 1

                if self.logger:
                    self.logger.log_event(
                        "Manager",
                        (
                            f"Created job {job.id_job} for patient {patient.id_patient} "
                            f"with {num_items} items"
                        ),
                    )

                # Send job into the stage-based factory pipeline
                self.factory.submit_job(job)

            else:
                if POLICY_ORDER_TO_JOB == "MAX_PER_JOB":
                    items_per_job = PALLET_SIZE_LIMIT

                    for i in range(0, num_items, items_per_job):
                        job_items = patient_items[i : i + items_per_job]
                        job = Job(self.next_job_id, job_items)
                        self.next_job_id += 1

                        if self.logger:
                            self.logger.log_event(
                                "Manager",
                                (
                                    f"Created split job {job.id_job} for patient "
                                    f"{patient.id_patient} with {len(job_items)} items"
                                ),
                            )

                        self.factory.submit_job(job)

