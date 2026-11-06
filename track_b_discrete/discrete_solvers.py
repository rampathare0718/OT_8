"""
Track B: Discrete Optimization Solvers
Implements:
1. Greedy Heuristic
2. Dynamic Programming (0/1 Knapsack & Job Scheduling)
3. Branch-and-Bound (Exact combinatorial search)
4. Integer Linear Programming (ILP via Branch-and-Bound Simplex relaxation)
"""

import time
import numpy as np

class KnapsackProblem:
    """Standard 0/1 Knapsack Problem: maximize sum v_i x_i s.t. sum w_i x_i <= W, x_i in {0, 1}."""
    def __init__(self, values, weights, capacity):
        self.values = np.array(values, dtype=float)
        self.weights = np.array(weights, dtype=float)
        self.capacity = float(capacity)
        self.n = len(values)

    def evaluate(self, x):
        """Returns (value, weight, is_feasible)."""
        x = np.array(x, dtype=int)
        total_w = np.sum(x * self.weights)
        total_v = np.sum(x * self.values)
        return total_v, total_w, (total_w <= self.capacity)

class GreedyKnapsackSolver:
    """Greedy heuristic based on value-to-weight ratio density."""
    def solve(self, problem: KnapsackProblem):
        ratios = problem.values / np.maximum(problem.weights, 1e-9)
        sorted_indices = np.argsort(ratios)[::-1]
        
        solution = np.zeros(problem.n, dtype=int)
        curr_weight = 0.0
        curr_val = 0.0

        for idx in sorted_indices:
            if curr_weight + problem.weights[idx] <= problem.capacity:
                solution[idx] = 1
                curr_weight += problem.weights[idx]
                curr_val += problem.values[idx]

        return solution, curr_val, curr_weight

class DynamicProgrammingKnapsackSolver:
    """Exact pseudo-polynomial Dynamic Programming solver."""
    def solve(self, problem: KnapsackProblem):
        W = int(problem.capacity)
        N = problem.n
        weights = problem.weights.astype(int)
        values = problem.values

        dp = np.zeros((N + 1, W + 1), dtype=float)

        for i in range(1, N + 1):
            w_i = weights[i - 1]
            v_i = values[i - 1]
            for w in range(W + 1):
                if w >= w_i:
                    dp[i, w] = max(dp[i - 1, w], dp[i - 1, w - w_i] + v_i)
                else:
                    dp[i, w] = dp[i - 1, w]

        # Backtrack solution vector
        solution = np.zeros(N, dtype=int)
        curr_w = W
        for i in range(N, 0, -1):
            if dp[i, curr_w] != dp[i - 1, curr_w]:
                solution[i - 1] = 1
                curr_w -= weights[i - 1]

        total_val = dp[N, W]
        total_weight = np.sum(solution * problem.weights)
        return solution, total_val, total_weight

class BranchAndBoundKnapsackSolver:
    """Exact Branch-and-Bound solver using LP continuous relaxation bounds."""
    def __init__(self):
        self.best_val = 0.0
        self.best_sol = None
        self.nodes_explored = 0

    def solve(self, problem: KnapsackProblem):
        self.best_val = 0.0
        self.best_sol = np.zeros(problem.n, dtype=int)
        self.nodes_explored = 0

        # Sort items by value density for sharper LP upper bounds
        ratios = problem.values / np.maximum(problem.weights, 1e-9)
        order = np.argsort(ratios)[::-1]
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(problem.n)

        sorted_v = problem.values[order]
        sorted_w = problem.weights[order]
        capacity = problem.capacity
        n = problem.n

        def upper_bound(idx, curr_w, curr_v):
            if curr_w > capacity:
                return 0.0
            bound = curr_v
            w = curr_w
            for i in range(idx, n):
                if w + sorted_w[i] <= capacity:
                    w += sorted_w[i]
                    bound += sorted_v[i]
                else:
                    # Fractional item relaxation (Linear Programming relaxation)
                    bound += (capacity - w) * (sorted_v[i] / sorted_w[i])
                    break
            return bound

        def branch(idx, curr_w, curr_v, sol):
            self.nodes_explored += 1
            if curr_w <= capacity and curr_v > self.best_val:
                self.best_val = curr_v
                self.best_sol = sol.copy()

            if idx >= n:
                return

            ub = upper_bound(idx, curr_w, curr_v)
            if ub <= self.best_val:
                return  # Prune branch

            # Branch 1: Include item idx (if fits)
            if curr_w + sorted_w[idx] <= capacity:
                sol[idx] = 1
                branch(idx + 1, curr_w + sorted_w[idx], curr_v + sorted_v[idx], sol)
                sol[idx] = 0

            # Branch 2: Exclude item idx
            branch(idx + 1, curr_w, curr_v, sol)

        curr_sol = np.zeros(n, dtype=int)
        branch(0, 0.0, 0.0, curr_sol)

        # Map back to original order
        final_solution = self.best_sol[inv_order]
        total_weight = np.sum(final_solution * problem.weights)
        return final_solution, self.best_val, total_weight, self.nodes_explored

