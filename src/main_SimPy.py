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
    """
    Build a capacity dictionary used by KPI.to_dict()
    to compute resource utilizations.
    """
    auto_cfg = FACTORY_AUTO_POST
    plat_cfg = FACTORY_PLATFORM_CLEAN
    print_cfg = FACTORY_PRINT
    pre_cfg = FACTORY_PRINT  # (원 코드 유지)
    manual_cfg = FACTORY_MANUAL_OPS

    caps = {
        "printers": print_cfg.get("printer_count"),
        "preproc_servers": pre_cfg.get("preproc_servers", pre_cfg.get("servers", 1)),
        "wash_m1": auto_cfg.get("washers_m1"),
        "wash_m2": auto_cfg.get("washers_m2"),
        "dryers": auto_cfg.get("dryers"),
        "uv_units": auto_cfg.get("uv_units"),
        "amr": auto_cfg.get("amr_count"),
        "platform_washers": plat_cfg.get("washers"),
        "manual_workers": manual_cfg.get("workers"),
    }
    return caps


def run_full_simulation(sim_duration: int = SIM_TIME, show_gantt=False):
    """
    Run a full simulation with:
        Customer → Manager → Factory (stage-based).
    """
    print("================ Full Factory Simulation ================")

    env = simpy.Environment()
    logger = Logger(env)

    manager = Manager(env, logger=logger)
    customer = Customer(env, manager, logger=logger)

    env.run(until=sim_duration)

    logger.log_event(
        "SIM_END",
        f"Simulation ended at t={sim_duration} minutes."
    )

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

    logger.log_event("SIM_SUMMARY", f"KPI={kpi_dict}")

    if show_gantt:
        if logger.trace_events:
            logger.visualize_trace_gantt(logger.trace_events, title="Factory Trace Gantt (trace_events)")

    print("\n================ Full Simulation Finished ================")

    # 프론트(result.html)가 원하는 키 포함해서 반환
    result = {
        "kpi": kpi_dict,
        "caps": caps,
        "trace_events": logger.trace_events,
        "stacker_wip_history": getattr(kpi, "stacker_wip_history", []),
        "amr_route_counts": getattr(kpi, "amr_route_counts", {}),
        "scrap_by_stage": getattr(kpi, "scrap_by_stage", {}),
    }

    return result


if __name__ == "__main__":
    random.seed(42)
    run_full_simulation(sim_duration=SIM_TIME, show_gantt=True)
