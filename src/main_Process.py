import random
import simpy
from typing import Optional
from base_Customer import Item
from base_Job import Job
from log_SimPy import Logger
from manager import Manager
from base_Customer import Customer
from KPI import KPI
from factory_platform import Factory
from config_SimPy import (
    SIM_TIME,
    FACTORY_PLATFORM_CLEAN,
    FACTORY_AUTO_POST,
    FACTORY_PRINT,
    FACTORY_MANUAL_OPS
)



def build_caps_from_cfg() -> dict:
    """
    Build a resource capacity dictionary from configuration values.

    This dictionary is later passed into KPI.to_dict() so that
    utilization metrics can be calculated per resource.
    """
    print_cfg = FACTORY_PRINT
    auto_cfg = FACTORY_AUTO_POST
    plat_cfg = FACTORY_PLATFORM_CLEAN
    manual_cfg = FACTORY_MANUAL_OPS

    caps = {
        "printers": print_cfg.get("printer_count"),
        "wash_m1": auto_cfg.get("washers_m1"),
        "wash_m2": auto_cfg.get("washers_m2"),
        "dryers": auto_cfg.get("dryers"),
        "uv_units": auto_cfg.get("uv_units"),
        "amr": auto_cfg.get("amr_count"),
        "platform_washers": plat_cfg.get("washers"),
        "manual_workers": manual_cfg.get("workers")
    }

    return caps


def run_process_pipeline_validation(sim_duration: Optional[int] = None):
    """
    Validation mode that uses the full demand logic instead of synthetically generating Jobs.

    This ensures:
        - Orders are generated based on `order_cycle_min` in config.
        - Jobs are created only by the Manager based on Order content.
        - Only as many Jobs as demand produces are mapped to platforms.
    """

    print("================ Process Pipeline Validation (Factory) ================")

    # 1) Create simulation environment
    env = simpy.Environment()
    logger = Logger(env)

    # 2) Create Manager (this automatically creates Factory and KPI)
    manager = Manager(env=env, logger=logger)
    factory = manager.factory   # Just reference it (do NOT recreate Factory)
    kpi: KPI = factory.kpi      # Shared KPI instance

    # 3) Create Customer that generates Orders on a schedule
    customer = Customer(env=env, order_receiver=manager, logger=logger)

    # 4) Determine simulation duration
    if sim_duration is None:
        sim_duration = SIM_TIME

    logger.log_event(
        "VALIDATION",
        f"Running sim with config-based order generation (order_cycle_min applied)."
    )

    print(f"\nStarting simulation for {sim_duration} minutes...")
    env.run(until=sim_duration)

    # 5) Reporting
    print("\n================ Simulation Results ================")
    print(f"Total Jobs completed     : {kpi.finished_platforms}")
    print(f"Total Jobs started       : {kpi.started_platforms}")

    caps = build_caps_from_cfg()
    kpi_dict = kpi.to_dict(horizon_min=sim_duration, caps=caps)


    """
    print("\nKPI Summary:")
    for key, value in kpi_dict.items():
        if key == "utilization":
            print("  utilization:")
            for rname, u in value.items():
                if u is None:
                    print(f"    - {rname}: None")
                else:
                    print(f"    - {rname}: {u:.3f}")
        else:
            print(f"  {key}: {value}")
    """        
    """
    if logger:
        logger.log_event(
            "VALIDATION",
            f"Validation complete: created={len(jobs)}, completed={kpi.finished_platforms}"
        )
    """
    print("\n================ Test Ended ================")


if __name__ == "__main__":
    random.seed(42)

    # Run validation using config values only
    run_process_pipeline_validation()
