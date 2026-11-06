"""
Track E: Case Study Benchmarks & Verification
Compares Baseline Heuristic vs Exact Linear Programming vs Genetic Algorithm
Generates 24-hour battery dispatch curves and strategy-selection table.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .bess_problem import BESSProblem
from .solvers import HeuristicBaselineController, ExactLPSolver, GeneticAlgorithmSolver

def run_track_e_experiments(output_dir="results"):
    print("=" * 60)
    print("RUNNING TRACK E: REAL-PROCESS CASE STUDY (BESS ARBITRAGE)")
    print("=" * 60)

    prob = BESSProblem(seed=42)
    results = []

    # 1. Baseline Heuristic Controller
    t0 = time.perf_counter()
    p_ch_base, p_dis_base, cost_base, prof_base, feas_base, soc_base = HeuristicBaselineController().solve(prob)
    t_base = (time.perf_counter() - t0) * 1000.0

    results.append({
        "Track": "E",
        "Method": "Rule-Based Heuristic (Baseline)",
        "Net_Cost_$": cost_base,
        "Net_Profit_$": prof_base,
        "Feasible": feas_base,
        "Runtime_ms": t_base,
        "Daily_Cycles": np.sum(p_ch_base) / prob.E_cap,
        "Optimality_Gap_%": 0.0 # Will compute relative to LP
    })

    # 2. Exact Linear Programming (LP Highs)
    t0 = time.perf_counter()
    p_ch_lp, p_dis_lp, cost_lp, prof_lp, feas_lp, soc_lp = ExactLPSolver().solve(prob)
    t_lp = (time.perf_counter() - t0) * 1000.0

    results.append({
        "Track": "E",
        "Method": "Exact Linear Programming (HiGHS)",
        "Net_Cost_$": cost_lp,
        "Net_Profit_$": prof_lp,
        "Feasible": feas_lp,
        "Runtime_ms": t_lp,
        "Daily_Cycles": np.sum(p_ch_lp) / prob.E_cap,
        "Optimality_Gap_%": 0.0
    })

    # 3. Genetic Algorithm (Metaheuristic)
    t0 = time.perf_counter()
    p_ch_ga, p_dis_ga, cost_ga, prof_ga, feas_ga, soc_ga = GeneticAlgorithmSolver(generations=120, seed=42).solve(prob)
    t_ga = (time.perf_counter() - t0) * 1000.0

    results.append({
        "Track": "E",
        "Method": "Genetic Algorithm (Evolutionary)",
        "Net_Cost_$": cost_ga,
        "Net_Profit_$": prof_ga,
        "Feasible": feas_ga,
        "Runtime_ms": t_ga,
        "Daily_Cycles": np.sum(p_ch_ga) / prob.E_cap,
        "Optimality_Gap_%": 0.0
    })

    # Calculate true improvement and optimality gaps
    # Profit improvement: (Prof_opt - Prof_base)
    for r in results:
        r["Optimality_Gap_%"] = ((r["Net_Cost_$"] - cost_lp) / abs(cost_lp)) * 100.0

    # Plot 24-hour dispatch curves
    hours = np.arange(24)
    fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    # Subplot 1: Market conditions (Price & Solar)
    ax1 = axs[0]
    ax1.plot(hours, prob.prices, "r-o", label="Wholesale Electricity Price ($/MWh)")
    ax1.set_ylabel("Price ($/MWh)", color="r")
    ax1.tick_params(axis="y", labelcolor="r")
    ax1_twin = ax1.twinx()
    ax1_twin.fill_between(hours, prob.solar_profile, color="gold", alpha=0.35, label="Solar PV Profile (MW)")
    ax1_twin.set_ylabel("Solar (MW)", color="goldenrod")
    ax1_twin.tick_params(axis="y", labelcolor="goldenrod")
    ax1.set_title("Market Context: CAISO Wholesale Duck Curve & Solar Generation")
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Optimized Battery Dispatch (LP vs Baseline)
    ax2 = axs[1]
    ax2.bar(hours - 0.2, p_ch_lp, width=0.4, color="forestgreen", label="Optimized Charge P_ch (LP)")
    ax2.bar(hours + 0.2, -p_dis_lp, width=0.4, color="crimson", label="Optimized Discharge P_dis (LP)")
    ax2.plot(hours, p_ch_base - p_dis_base, "k--", label="Baseline Net Dispatch")
    ax2.set_ylabel("Battery Power (MW)")
    ax2.set_title("Optimized Dispatch Schedule (Positive=Charge, Negative=Discharge)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Subplot 3: State of Charge (SoC) Trajectory
    ax3 = axs[2]
    ax3.plot(hours, soc_lp[:24] * 100, "b-s", label="Exact LP SoC Trajectory")
    ax3.plot(hours, soc_base[:24] * 100, "g--^", label="Baseline Rule SoC")
    ax3.axhline(prob.SoC_min * 100, color="gray", linestyle=":", label="SoC Safety Limits (10% - 90%)")
    ax3.axhline(prob.SoC_max * 100, color="gray", linestyle=":")
    ax3.set_xlabel("Hour of Day (0 - 23)")
    ax3.set_ylabel("Battery SoC (%)")
    ax3.set_title("State of Charge Conservation & Cycling Health")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/track_e_bess_dispatch.png", dpi=150)
    plt.close()

    df_e = pd.DataFrame(results)
    df_e.to_csv(f"{output_dir}/track_e_results.csv", index=False)
    print(f"Track E completed successfully! Results written to {output_dir}/track_e_results.csv")
    print(df_e.to_string(index=False))
    return df_e

if __name__ == "__main__":
    run_track_e_experiments()

