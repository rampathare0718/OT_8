"""
Track B: Classical QUBO Solvers & Benchmarking
Provides:
1. Exact Brute-Force QUBO solver (ground truth)
2. Simulated Annealing QUBO solver
3. Tabu Search QUBO solver
4. Full benchmark suite comparing discrete methods
"""

import time
import numpy as np
import pandas as pd
from .discrete_solvers import (
    KnapsackProblem, GreedyKnapsackSolver,
    DynamicProgrammingKnapsackSolver, BranchAndBoundKnapsackSolver
)
from .qubo_formulation import (
    build_maxcut_qubo, build_feature_selection_qubo,
    binary_to_spin, spin_to_binary
)

class ExactQUBOSolver:
    """Exact enumeration of all 2^N states to establish certified global minimum."""
    def solve(self, qubo):
        n = qubo.n
        best_x = None
        best_energy = float("inf")

        for idx in range(2**n):
            # Bit extraction
            x = np.array([(idx >> i) & 1 for i in range(n)], dtype=int)
            e = qubo.energy(x)
            if e < best_energy:
                best_energy = e
                best_x = x

        return best_x, best_energy

class SimulatedAnnealingQUBOSolver:
    """Classical Simulated Annealing (Metropolis-Hastings schedule)."""
    def __init__(self, t_initial=10.0, t_final=0.01, cooling_rate=0.95, steps_per_temp=50, seed=42):
        self.t_initial = t_initial
        self.t_final = t_final
        self.cooling_rate = cooling_rate
        self.steps_per_temp = steps_per_temp
        self.rng = np.random.default_rng(seed)

    def solve(self, qubo):
        n = qubo.n
        # Initial random state
        curr_x = self.rng.integers(0, 2, size=n)
        curr_energy = qubo.energy(curr_x)

        best_x = curr_x.copy()
        best_energy = curr_energy

        temp = self.t_initial
        while temp > self.t_final:
            for _ in range(self.steps_per_temp):
                flip_idx = self.rng.integers(0, n)
                cand_x = curr_x.copy()
                cand_x[flip_idx] = 1 - cand_x[flip_idx]
                cand_energy = qubo.energy(cand_x)

                dE = cand_energy - curr_energy
                if dE < 0 or self.rng.uniform(0, 1) < np.exp(-dE / temp):
                    curr_x = cand_x
                    curr_energy = cand_energy
                    if curr_energy < best_energy:
                        best_energy = curr_energy
                        best_x = curr_x.copy()

            temp *= self.cooling_rate

        return best_x, best_energy

class TabuSearchQUBOSolver:
    """Tabu Search with 1-flip neighborhood and recency-based tenure."""
    def __init__(self, tenure=5, max_iter=200, seed=42):
        self.tenure = tenure
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

    def solve(self, qubo):
        n = qubo.n
        curr_x = self.rng.integers(0, 2, size=n)
        curr_energy = qubo.energy(curr_x)

        best_x = curr_x.copy()
        best_energy = curr_energy

        tabu_list = np.zeros(n, dtype=int)

        for it in range(1, self.max_iter + 1):
            best_cand_idx = -1
            best_cand_energy = float("inf")

            for i in range(n):
                cand_x = curr_x.copy()
                cand_x[i] = 1 - cand_x[i]
                cand_energy = qubo.energy(cand_x)

                # Aspiration criterion: allow if improves overall best
                is_tabu = (tabu_list[i] >= it)
                if not is_tabu or cand_energy < best_energy:
                    if cand_energy < best_cand_energy:
                        best_cand_energy = cand_energy
                        best_cand_idx = i

            if best_cand_idx != -1:
                curr_x[best_cand_idx] = 1 - curr_x[best_cand_idx]
                curr_energy = best_cand_energy
                tabu_list[best_cand_idx] = it + self.tenure

                if curr_energy < best_energy:
                    best_energy = curr_energy
                    best_x = curr_x.copy()

        return best_x, best_energy

def run_track_b_experiments(output_dir="results"):
    print("=" * 60)
    print("RUNNING TRACK B: DISCRETE OPTIMIZATION EXPERIMENTS")
    print("=" * 60)

    results = []

    # 1. 0/1 Knapsack Benchmarking
    values = [40, 50, 100, 95, 30, 25, 60, 45, 80, 70]
    weights = [10, 20, 30, 40, 15, 12, 25, 18, 35, 28]
    capacity = 100
    knapsack = KnapsackProblem(values, weights, capacity)

    # Greedy
    t0 = time.perf_counter()
    g_sol, g_val, g_w = GreedyKnapsackSolver().solve(knapsack)
    t_greedy = (time.perf_counter() - t0) * 1000.0

    # Dynamic Programming
    t0 = time.perf_counter()
    dp_sol, dp_val, dp_w = DynamicProgrammingKnapsackSolver().solve(knapsack)
    t_dp = (time.perf_counter() - t0) * 1000.0

    # Branch-and-Bound
    t0 = time.perf_counter()
    bb_sol, bb_val, bb_w, nodes = BranchAndBoundKnapsackSolver().solve(knapsack)
    t_bb = (time.perf_counter() - t0) * 1000.0

    results.append({
        "Track": "B",
        "Problem": "0/1 Knapsack (N=10, C=100)",
        "Solver": "Greedy (Value/Weight)",
        "Objective_Value": g_val,
        "Feasible": g_w <= capacity,
        "Runtime_ms": t_greedy,
        "Optimality_Gap_%": ((dp_val - g_val) / dp_val) * 100.0
    })
    results.append({
        "Track": "B",
        "Problem": "0/1 Knapsack (N=10, C=100)",
        "Solver": "Dynamic Programming",
        "Objective_Value": dp_val,
        "Feasible": dp_w <= capacity,
        "Runtime_ms": t_dp,
        "Optimality_Gap_%": 0.0
    })
    results.append({
        "Track": "B",
        "Problem": "0/1 Knapsack (N=10, C=100)",
        "Solver": f"Branch-and-Bound ({nodes} nodes)",
        "Objective_Value": bb_val,
        "Feasible": bb_w <= capacity,
        "Runtime_ms": t_bb,
        "Optimality_Gap_%": 0.0
    })

    # 2. Max-Cut QUBO Benchmark (5-node Butterfly Graph)
    # Adjacency: 0-1, 0-2, 1-2, 2-3, 3-4, 2-4
    A_graph = np.zeros((5, 5))
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (2, 4)]
    for u, v in edges:
        A_graph[u, v] = 1.0
        A_graph[v, u] = 1.0

    qubo_maxcut = build_maxcut_qubo(A_graph)

    # Exact QUBO
    t0 = time.perf_counter()
    exact_x, exact_e = ExactQUBOSolver().solve(qubo_maxcut)
    t_exact = (time.perf_counter() - t0) * 1000.0

    # Simulated Annealing QUBO
    t0 = time.perf_counter()
    sa_x, sa_e = SimulatedAnnealingQUBOSolver().solve(qubo_maxcut)
    t_sa = (time.perf_counter() - t0) * 1000.0

    # Tabu Search QUBO
    t0 = time.perf_counter()
    tabu_x, tabu_e = TabuSearchQUBOSolver().solve(qubo_maxcut)
    t_tabu = (time.perf_counter() - t0) * 1000.0

    results.append({
        "Track": "B",
        "Problem": "Max-Cut QUBO (5-node graph)",
        "Solver": "Exact Brute Force",
        "Objective_Value": -exact_e,  # Max cut value is negative energy
        "Feasible": True,
        "Runtime_ms": t_exact,
        "Optimality_Gap_%": 0.0
    })
    results.append({
        "Track": "B",
        "Problem": "Max-Cut QUBO (5-node graph)",
        "Solver": "Simulated Annealing",
        "Objective_Value": -sa_e,
        "Feasible": True,
        "Runtime_ms": t_sa,
        "Optimality_Gap_%": ((-exact_e - (-sa_e)) / (-exact_e)) * 100.0
    })
    results.append({
        "Track": "B",
        "Problem": "Max-Cut QUBO (5-node graph)",
        "Solver": "Tabu Search",
        "Objective_Value": -tabu_e,
        "Feasible": True,
        "Runtime_ms": t_tabu,
        "Optimality_Gap_%": ((-exact_e - (-tabu_e)) / (-exact_e)) * 100.0
    })

    df_b = pd.DataFrame(results)
    df_b.to_csv(f"{output_dir}/track_b_results.csv", index=False)
    print(f"Track B completed successfully! Results written to {output_dir}/track_b_results.csv")
    print(df_b.to_string(index=False))
    return df_b, qubo_maxcut, exact_x, exact_e

if __name__ == "__main__":
    run_track_b_experiments()

