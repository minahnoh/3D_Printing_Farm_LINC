# main.py
import simpy
import random
from base_Customer import Customer
from manager import Manager
from log_SimPy import Logger
from config_SimPy import *


def run_simulation(sim_duration=SIM_TIME):
    """Run the manufacturing simulation"""
    print("================ Manufacturing Process Simulation ================")

    # Setup simulation environment
    env = simpy.Environment()

    # Create logger with env
    logger = Logger(env)

    # Create manager and provide logger
    manager = Manager(env, logger)

    # Create customer to generate orders
    Customer(env, manager, logger)

    # Run simulation
    print("\nStarting simulation...")
    print(f"Simulation will run for {sim_duration} minutes")

    # Run simulation
    env.run(until=sim_duration)

    # Collect and display results
    print("\n================ Simulation Results ================")

    # Get basic statistics from manager
    manager_stats = manager.collect_statistics()

    # Basic results
    print(f"Completed jobs by process:")
    print(f"  Build: {manager_stats['build_completed']}")
    print(f"  Wash: {manager_stats['wash_completed']}")
    print(f"  Dry: {manager_stats['dry_completed']}")
    print(f"  Inspect: {manager_stats['inspect_completed']}")

    print(f"\nRemaining defective items: {manager_stats['defective_items']}")

    # Queue statistics
    print("\nFinal queue lengths:")
    print(f"  Build queue: {manager_stats['build_queue']}")
    print(f"  Wash queue: {manager_stats['wash_queue']}")
    print(f"  Dry queue: {manager_stats['dry_queue']}")
    print(f"  Inspect queue: {manager_stats['inspect_queue']}")

    # Collect detailed statistics and visualize if enabled
    if DETAILED_STATS_ENABLED or GANTT_CHART_ENABLED or VIS_STAT_ENABLED:
        print("\nCollecting detailed statistics...")
        processes = manager.get_processes()
        stats = logger.collect_statistics(processes)

        # Visualize results if enabled
        if GANTT_CHART_ENABLED or VIS_STAT_ENABLED:
            logger.visualize_statistics(stats, processes)

    print("\n================ Simulation Ended ================")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Run the simulation
    run_simulation()
