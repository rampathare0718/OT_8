"""
Track J: Software Engineering Benchmarks & APFD Optimization
Evaluates:
- Baseline Full Test Suite vs 50% CI Budget Minimization
- Greedy Additional Coverage vs 0-1 ILP Branch & Bound
- APFD Fault Detection Metric & Code Coverage Preservation
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
from .test_suite_data import build_codebase_test_suite
from .tsmp_solvers import calculate_apfd, GreedyTSMPSolver, ILPTSMPSolver

def run_track_j_experiments(output_dir="results"):
    print("=" * 60)
    print("RUNNING TRACK J: SOFTWARE ENGINEERING OPTIMIZATION (TSMP)")
    print("=" * 60)

    tests, total_funcs, total_faults = build_codebase_test_suite(seed=42)
    total_duration = sum(t.duration_ms for t in tests)
    total_possible_cov = len(set.union(*[t.covered_functions for t in tests]))

    # 1. Baseline Full Suite (Arbitrary Order)
    base_apfd = calculate_apfd(tests, total_faults)

    # 2. Constrained CI Budget: 50% of total test duration
    ci_budget = 0.50 * total_duration

    # Greedy Additional Coverage Solver
    t0 = time.perf_counter()
    greedy_tests, greedy_cov = GreedyTSMPSolver().solve(tests, ci_budget, total_funcs)
    t_greedy = (time.perf_counter() - t0) * 1000.0
    greedy_dur = sum(t.duration_ms for t in greedy_tests)
    greedy_apfd = calculate_apfd(greedy_tests, total_faults)

    # 0-1 ILP Solver
    t0 = time.perf_counter()
    ilp_tests, ilp_cov = ILPTSMPSolver().solve(tests, ci_budget, total_funcs)
    t_ilp = (time.perf_counter() - t0) * 1000.0
    ilp_dur = sum(t.duration_ms for t in ilp_tests)
    ilp_apfd = calculate_apfd(ilp_tests, total_faults)

    results = [
        {
            "Track": "J",
            "Strategy": "Baseline Full Test Suite",
            "Test_Count": len(tests),
            "Execution_Time_ms": total_duration,
            "Time_Saved_%": 0.0,
            "Coverage_Functions": len(set.union(*[t.covered_functions for t in tests])),
            "Coverage_%": (total_possible_cov / total_funcs) * 100.0,
            "APFD_Fault_Metric": base_apfd,
            "Solver_Runtime_ms": 0.0
        },
        {
            "Track": "J",
            "Strategy": "Greedy Additional Coverage (50% Budget)",
            "Test_Count": len(greedy_tests),
            "Execution_Time_ms": greedy_dur,
            "Time_Saved_%": ((total_duration - greedy_dur) / total_duration) * 100.0,
            "Coverage_Functions": len(greedy_cov),
            "Coverage_%": (len(greedy_cov) / total_funcs) * 100.0,
            "APFD_Fault_Metric": greedy_apfd,
            "Solver_Runtime_ms": t_greedy
        },
        {
            "Track": "J",
            "Strategy": "0-1 ILP Set Cover (50% Budget)",
            "Test_Count": len(ilp_tests),
            "Execution_Time_ms": ilp_dur,
            "Time_Saved_%": ((total_duration - ilp_dur) / total_duration) * 100.0,
            "Coverage_Functions": len(ilp_cov),
            "Coverage_%": (len(ilp_cov) / total_funcs) * 100.0,
            "APFD_Fault_Metric": ilp_apfd,
            "Solver_Runtime_ms": t_ilp
        }
    ]

    # Plot APFD and Time Saved
    strats = ["Baseline", "Greedy (50% Time)", "0-1 ILP (50% Time)"]
    durations = [total_duration, greedy_dur, ilp_dur]
    apfds = [base_apfd, greedy_apfd, ilp_apfd]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(strats, durations, color=["slategray", "cornflowerblue", "mediumseagreen"], alpha=0.8, width=0.45)
    ax1.set_ylabel("CI Execution Time (ms)", color="navy")
    ax1.tick_params(axis="y", labelcolor="navy")

    ax2 = ax1.twinx()
    ax2.plot(strats, apfds, "r-D", linewidth=2.5, markersize=8, label="APFD Fault Score")
    ax2.set_ylabel("APFD Score (higher is better)", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.set_ylim(0, 1.0)

    plt.title("Track J: Test Suite Minimization Time vs APFD Fault Detection")
    plt.grid(True, alpha=0.2)
    plt.savefig(f"{output_dir}/track_j_tsmp_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close()

    df_j = pd.DataFrame(results)
    df_j.to_csv(f"{output_dir}/track_j_results.csv", index=False)
    print(f"Track J completed successfully! Results written to {output_dir}/track_j_results.csv")
    print(df_j.to_string(index=False))
    return df_j

if __name__ == "__main__":
    run_track_j_experiments()

