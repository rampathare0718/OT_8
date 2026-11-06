"""
Track E: Solvers for BESS Dispatch Optimization
1. Baseline Heuristic (Rule-based Threshold Controller)
2. Exact LP / MILP Solver (via Scipy Highs Interior-Point / Simplex)
3. Genetic Algorithm (Metaheuristic Global Search)
"""

import time
import numpy as np
from scipy.optimize import linprog
from .bess_problem import BESSProblem

class HeuristicBaselineController:
    """Greedy rule controller: charges during low price hours, discharges during peak."""
    def solve(self, prob: BESSProblem):
        T = prob.T
        P_ch = np.zeros(T)
        P_dis = np.zeros(T)

        p_low = np.percentile(prob.prices, 25)
        p_high = np.percentile(prob.prices, 75)

        curr_soc = prob.SoC_0
        for t in range(T):
            price = prob.prices[t]
            # Greedy charge
            if price <= p_low and curr_soc < prob.SoC_max:
                max_ch_energy = (prob.SoC_max - curr_soc) * prob.E_cap / prob.eta_ch
                p_c = min(prob.P_max, max_ch_energy / prob.dt)
                P_ch[t] = p_c
                curr_soc += (prob.eta_ch * p_c * prob.dt) / prob.E_cap
            # Greedy discharge
            elif price >= p_high and curr_soc > prob.SoC_min:
                max_dis_energy = (curr_soc - prob.SoC_min) * prob.E_cap * prob.eta_dis
                p_d = min(prob.P_max, max_dis_energy / prob.dt)
                # Check feeder
                p_d = min(p_d, prob.P_feeder - prob.solar_profile[t])
                p_d = max(0.0, p_d)
                P_dis[t] = p_d
                curr_soc -= (p_d * prob.dt) / (prob.eta_dis * prob.E_cap)

        cost, profit, is_feas, soc = prob.evaluate_dispatch(P_ch, P_dis)
        return P_ch, P_dis, cost, profit, is_feas, soc

class ExactLPSolver:
    """
    Formulates BESS as Linear Programming:
    Variables x = [P_ch(0..23), P_dis(0..23), SoC(1..24)] in R^72
    min c^T x s.t. A_eq x = b_eq, A_ub x <= b_ub, lb <= x <= ub
    Guarantees global mathematical optimality.
    """
    def solve(self, prob: BESSProblem):
        T = prob.T
        N_vars = 3 * T  # 24 P_ch, 24 P_dis, 24 SoC

        # Objective vector c:
        # P_ch: Price(t) + C_deg
        # P_dis: -Price(t) + C_deg
        # SoC: 0
        c = np.zeros(N_vars)
        for t in range(T):
            c[t] = (prob.prices[t] + prob.C_deg) * prob.dt
            c[T + t] = (-prob.prices[t] + prob.C_deg) * prob.dt

        # Equality constraints: SoC conservation
        # SoC(t) - SoC(t-1) - eta_ch * P_ch(t) * dt / E_cap + P_dis(t) * dt / (eta_dis * E_cap) = 0
        A_eq = np.zeros((T, N_vars))
        b_eq = np.zeros(T)

        for t in range(T):
            # Coefficients for P_ch(t) and P_dis(t)
            A_eq[t, t] = -(prob.eta_ch * prob.dt) / prob.E_cap
            A_eq[t, T + t] = prob.dt / (prob.eta_dis * prob.E_cap)
            # Coefficient for SoC(t)
            A_eq[t, 2 * T + t] = 1.0
            # Coefficient for SoC(t-1)
            if t == 0:
                b_eq[0] = prob.SoC_0
            else:
                A_eq[t, 2 * T + (t - 1)] = -1.0

        # Inequality constraints:
        # 1. Feeder injection: P_dis(t) - P_ch(t) <= P_feeder - P_solar(t)
        # 2. End of day balance: SoC(T) >= SoC(0) => -SoC(T) <= -SoC(0)
        A_ub = np.zeros((T + 1, N_vars))
        b_ub = np.zeros(T + 1)

        for t in range(T):
            A_ub[t, t] = -1.0  # -P_ch
            A_ub[t, T + t] = 1.0  # +P_dis
            b_ub[t] = prob.P_feeder - prob.solar_profile[t]

        # End of day balance
        A_ub[T, 2 * T + (T - 1)] = -1.0
        b_ub[T] = -prob.SoC_0

        # Bounds:
        # 0 <= P_ch <= P_max
        # 0 <= P_dis <= P_max
        # SoC_min <= SoC <= SoC_max
        bounds = []
        for t in range(T):
            bounds.append((0.0, prob.P_max))
        for t in range(T):
            bounds.append((0.0, prob.P_max))
        for t in range(T):
            bounds.append((prob.SoC_min, prob.SoC_max))

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        if not res.success:
            raise RuntimeError(f"Linear program failed: {res.message}")

        sol = res.x
        P_ch_opt = sol[:T]
        P_dis_opt = sol[T:2*T]

        cost, profit, is_feas, soc = prob.evaluate_dispatch(P_ch_opt, P_dis_opt)
        return P_ch_opt, P_dis_opt, cost, profit, is_feas, soc

class GeneticAlgorithmSolver:
    """Evolutionary metaheuristic with tournament selection and Gaussian mutation."""
    def __init__(self, pop_size=60, generations=150, mutation_rate=0.15, seed=42):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng(seed)

    def solve(self, prob: BESSProblem):
        T = prob.T
        # Individual: vector of size 2*T (P_ch, P_dis)
        pop = self.rng.uniform(0, prob.P_max, size=(self.pop_size, 2 * T))
        best_ind = None
        best_cost = float("inf")

        for gen in range(self.generations):
            fitness = []
            for ind in pop:
                p_c = ind[:T]
                p_d = ind[T:]
                c, _, _, _ = prob.evaluate_dispatch(p_c, p_d)
                fitness.append(c)
                if c < best_cost:
                    best_cost = c
                    best_ind = ind.copy()

            # Tournament selection
            new_pop = [best_ind.copy()]  # Elitism
            while len(new_pop) < self.pop_size:
                i1, i2 = self.rng.integers(0, self.pop_size, 2)
                parent1 = pop[i1] if fitness[i1] < fitness[i2] else pop[i2]
                i3, i4 = self.rng.integers(0, self.pop_size, 2)
                parent2 = pop[i3] if fitness[i3] < fitness[i4] else pop[i4]

                # Uniform crossover
                mask = self.rng.integers(0, 2, size=2 * T).astype(bool)
                child = np.where(mask, parent1, parent2)

                # Mutation
                mutate_mask = self.rng.uniform(0, 1, size=2 * T) < self.mutation_rate
                child[mutate_mask] += self.rng.normal(0, 2.0, size=np.sum(mutate_mask))
                child = np.clip(child, 0.0, prob.P_max)

                new_pop.append(child)

            pop = np.array(new_pop)

        P_ch_ga = best_ind[:T]
        P_dis_ga = best_ind[T:]
        cost, profit, is_feas, soc = prob.evaluate_dispatch(P_ch_ga, P_dis_ga)
        return P_ch_ga, P_dis_ga, cost, profit, is_feas, soc

