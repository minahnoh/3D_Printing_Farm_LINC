import random
import simpy
from log_SimPy import Logger
from KPI import KPI 
from manager import Manager
from base_Customer import Customer

from config_SimPy import (
    SIM_TIME,
    FACTORY_PLATFORM_CLEAN,
    FACTORY_AUTO_POST,
    FACTORY_PRINT,
    FACTORY_MANUAL_OPS,
)


def build_caps_from_cfg() -> dict:
    """s
    Build a capacity dictionary used by KPI.to_dict()
    to compute resource utilizations.

    This helper mirrors the logic used in main_Process.py,
    but is now used for the full-factory simulation.
    """
    print_cfg = FACTORY_PRINT
    auto_cfg = FACTORY_AUTO_POST
    plat_cfg = FACTORY_PLATFORM_CLEAN
    manual_cfg = FACTORY_MANUAL_OPS

    caps = {
        # automatic resources
        "printers": print_cfg.get("printer_count"),
        "wash_m1": auto_cfg.get("washers_m1"),
        "wash_m2": auto_cfg.get("washers_m2"),
        "dryers": auto_cfg.get("dryers"),
        "uv_units": auto_cfg.get("uv_units"),
        "amr": auto_cfg.get("amr_count"),
        "platform_washers": plat_cfg.get("washers"),

        # logical / manual resources
        "manual_workers": manual_cfg.get("workers")
        
    }
    return caps


def run_full_simulation(sim_duration: int = SIM_TIME, show_gantt = False):
    """
    Run a full simulation with:
        Customer → Manager → Factory (stage-based).

    Steps:
        1) Create SimPy environment and Logger
        2) Create Manager (which internally creates the Factory)
        3) Create Customer and attach Manager as OrderReceiver
        4) Let Customer generate Orders over the horizon
        5) Run env.run(until=sim_duration)
        6) Collect KPI from manager.factory.kpi and print a summary
    """
    print("================ Full Factory Simulation ================")

   
    env = simpy.Environment()
    logger = Logger(env)

    manager = Manager(env=env, logger=logger)

    customer = Customer(env=env, order_receiver=manager, logger=logger)

    logger.log_event(
        "SIM_START",
        f"Simulation started for {sim_duration} minutes "
        f"(Customer → Manager → Factory pipeline)."
    )

    env.run(until=sim_duration)

    logger.log_event(
        "SIM_END",
        f"Simulation ended at t={sim_duration} minutes."
    )

    # Collect KPI from the Factory inside Manager
    factory = manager.factory
    kpi: KPI = factory.kpi

    print("\n================ Simulation Results ================")
    print(f"Finished platforms (KPI): {kpi.finished_platforms}")
    print(f"Started platforms (KPI) : {kpi.started_platforms}")

    caps = build_caps_from_cfg()
    kpi_dict = kpi.to_dict(horizon_min=sim_duration, caps=caps)

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

    logger.log_event(
        "SIM_SUMMARY",
        f"Finished platforms={kpi.finished_platforms}, "
        f"started_platforms={kpi.started_platforms}"
    )

    if show_gantt:
        # 구 factory.trace 방식 (필요하면 유지)
        if logger and hasattr(factory, "trace") and getattr(factory, "trace", None):
            logger.visualize_trace_gantt(factory.trace, title="Factory Trace Gantt (Factory.trace)")

        # stage 기반 logger.trace_events 간트차트
        if logger.trace_events:
            logger.visualize_trace_gantt(logger.trace_events, title="Factory Trace Gantt (trace_events)")

    print("\n================ Full Simulation Finished ================")

    # 여기서 결과를 dict로 반환 → app.py에서 그대로 JSON으로 내려줄 수 있음
    result = {
        "kpi":       kpi_dict,
        "caps":      caps,
        "trace_events": logger.trace_events,  # 프론트에서 간트차트 그릴 때 사용
        # 필요하면 추가:
        # "finished_platforms": kpi.finished_platforms,
        # "started_platforms":  kpi.started_platforms,
    }

    return result


if __name__ == "__main__":
    # Fix random seed for reproducibility
    random.seed(42)

    # Run full factory simulation
    run_full_simulation(sim_duration=SIM_TIME, show_gantt=True)

