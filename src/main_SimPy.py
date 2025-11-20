# main.py
import simpy
import random
from base_Customer import Customer
from manager import Manager
from log_SimPy import Logger
from config_SimPy import *


def run_simulation(sim_duration=SIM_TIME): 
    """Run the manufacturing simulation (Factory-based pipeline)"""
    print("================ Manufacturing Process Simulation ================")

    # Setup simulation environment
    env = simpy.Environment()

    # Create logger with env
    logger = Logger(env)

    # Create manager and provide logger
    manager = Manager(env, logger)

    # Create customer to generate orders (orders -> jobs)
    Customer(env, manager, logger)

    # === Factory 공정 프로세스 시작 ===
    # 전처리: pre_in -> ready (플랫폼 토큰 할당 + Job↔Platform 매핑)
    env.process(manager.factory.preproc_worker())
    # 프린팅/후공정 자동 흐름: ready -> print_pipeline
    env.process(manager.factory.printer_dispatcher())
    # 수동 후공정(서포트 제거, 피니싱, 도장)
    env.process(manager.factory.worker_manual_ops())

    # (옵션) 도착/공급 방식: arrivals나 infinite_feeder 중 택1 사용 시
    # env.process(manager.factory.arrivals())
    # 또는
    # env.process(manager.factory.infinite_feeder(manager.factory.horizon_minutes))

    # Run simulation
    print("\nStarting simulation (Factory pipeline)...")
    print(f"Simulation will run for {sim_duration} minutes")

    env.run(until=sim_duration)

    # Collect and display results
    print("\n================ Simulation Results ================")

    # ---- ① 기존 Job 공정 통계는 일단 사용 X (구 공정) ----
    # manager_stats = manager.collect_statistics()
    # print(f"Completed jobs by process:")
    # print(f"  Build: {manager_stats['build_completed']}")
    # print(f"  Wash: {manager_stats['wash_completed']}")
    # print(f"  Dry: {manager_stats['dry_completed']}")
    # print(f"  Inspect: {manager_stats['inspect_completed']}")
    # print(f"\nRemaining defective items: {manager_stats['defective_items']}")
    # print("\nFinal queue lengths:")
    # print(f"  Build queue: {manager_stats['build_queue']}")
    # print(f"  Wash queue: {manager_stats['wash_queue']}")
    # print(f"  Dry queue: {manager_stats['dry_queue']}")
    # print(f"  Inspect queue: {manager_stats['inspect_queue']}")

    # ---- ② Factory(플랫폼 파이프라인) KPI 출력 ----
    fkpi = manager.factory.kpi

    print("Factory KPI (platform-based pipeline):")
    print(f"  Started platforms     : {fkpi.started_platforms}")
    print(f"  Finished platforms    : {fkpi.finished_platforms}")
    print(f"  Started parts         : {fkpi.started_parts}")
    print(f"  Completed parts       : {fkpi.completed_parts}")
    print(f"  Scrapped parts        : {fkpi.scrapped_parts}")
    print(f"  Resin used [kg]       : {fkpi.resin_used_kg:.3f}")
    print(f"  Resin cost [KRW]      : {fkpi.resin_cost_krw:.0f}")
    print(f"  Max stacker WIP       : {fkpi.max_stacker_wip}")
    print(f"  # preproc jobs        : {fkpi.n_preproc_jobs}")
    print(f"  # print jobs          : {fkpi.n_print_jobs}")
    print(f"  Sum platform lead time: {fkpi.sum_platform_lead_time_min:.1f} min")

    print("\nResource utilization (busy time in minutes):")
    print(f"  Printer busy          : {fkpi.printer_busy_min:.1f}")
    print(f"  Wash1 busy            : {fkpi.wash1_busy_min:.1f}")
    print(f"  Wash2 busy            : {fkpi.wash2_busy_min:.1f}")
    print(f"  Dry busy              : {fkpi.dry_busy_min:.1f}")
    print(f"  UV busy               : {fkpi.uv_busy_min:.1f}")
    print(f"  Preproc busy          : {fkpi.preproc_busy_min:.1f}")
    print(f"  Platform wash busy    : {fkpi.platform_wash_busy_min:.1f}")
    print(f"  Manual worker busy    : {fkpi.manual_busy_min:.1f}")
    print(f"  Manual-move busy      : {fkpi.pre_manual_busy_min:.1f}")

    print("\nAMR statistics:")
    print(f"  Total AMR travel time : {fkpi.amr_travel_min:.1f} min")
    print(f"  # AMR moves           : {fkpi.n_amr_moves}")

    # (선택) 대기시간 통계 출력
    if getattr(fkpi, "wait_stats", None):
        print("\nWait times by resource (minutes):")
        for res_name, wait in fkpi.wait_stats.items():
            print(f"  {res_name:15s}: {wait:.1f}")

    # ---- ③ Logger 시각화 부분은 당장은 그대로 두고, 나중에 Factory용으로 확장 ----
    if DETAILED_STATS_ENABLED or GANTT_CHART_ENABLED or VIS_STAT_ENABLED:
        print("\nCollecting detailed statistics (legacy process-based logger)...")
        processes = manager.get_processes()  # 여기 안에 기존 Proc들만 있으면 거의 빈 결과일 수 있음
        stats = logger.collect_statistics(processes)

        if GANTT_CHART_ENABLED or VIS_STAT_ENABLED:
            logger.visualize_statistics(stats, processes)

    print("\n================ Simulation Ended ================")



if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Run the simulation
    run_simulation()
