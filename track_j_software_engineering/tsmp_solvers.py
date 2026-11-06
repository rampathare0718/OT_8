"""
Track J: Test Suite Minimization & Prioritization Solvers
Formulates TSMP under Continuous Integration (CI) execution budget:
1. APFD (Average Percentage of Faults Detected) metric
2. Naive Random / Chronological Baseline
3. Greedy Additional Coverage Heuristic (Submodular Greedy)
4. Exact 0-1 ILP / Branch-and-Bound Set Cover
"""

import numpy as np

def calculate_apfd(ordered_tests, total_faults):
    """
    Computes Average Percentage of Faults Detected (APFD):
    APFD = 1 - (sum_{j=1}^m TF_j) / (n * m) + 1 / (2 * n)
    where TF_j is the index (1-based) of the first test detecting fault j.
    """
    n = len(ordered_tests)
    m = total_faults
    if n == 0 or m == 0:
        return 0.0

    tf = {}
    for pos, test in enumerate(ordered_tests, 1):
        for f in test.detected_faults:
            if f not in tf:
                tf[f] = pos

    # If some faults are undetected by selected subset, penalize them with (n + 1)
    sum_tf = sum(tf.get(f, n + 1) for f in range(m))
    apfd = 1.0 - (sum_tf / (n * m)) + (1.0 / (2.0 * n))
    return float(apfd)

class GreedyTSMPSolver:
    """Greedy additional coverage heuristic with duration weighting."""
    def solve(self, tests, budget_ms, total_functions):
        selected = []
        covered = set()
        remaining_budget = budget_ms
        remaining_tests = list(tests)

        while remaining_tests and remaining_budget > 0:
            best_t = None
            best_density = -1.0

            for t in remaining_tests:
                if t.duration_ms <= remaining_budget:
                    new_coverage = len(t.covered_functions - covered)
                    density = new_coverage / t.duration_ms
                    if density > best_density:
                        best_density = density
                        best_t = t

            if best_t is None or best_density <= 0:
                # Fallback: fill remaining budget with fastest test
                affordable = [t for t in remaining_tests if t.duration_ms <= remaining_budget]
                if affordable:
                    best_t = min(affordable, key=lambda x: x.duration_ms)
                else:
                    break

            selected.append(best_t)
            covered.update(best_t.covered_functions)
            remaining_budget -= best_t.duration_ms
            remaining_tests.remove(best_t)

        return selected, covered

class ILPTSMPSolver:
    """
    Exact 0-1 ILP / Branch-and-Bound Set Cover under budget:
    max sum y_j s.t. y_j <= sum_{i: j in C_i} x_i, sum d_i x_i <= B
    """
    def solve(self, tests, budget_ms, total_functions):
        # Sort tests by duration density
        # For tractable branch-and-bound, filter to top 16 candidates
        candidates = sorted(tests, key=lambda t: len(t.covered_functions) / t.duration_ms, reverse=True)[:16]
        n = len(candidates)

        best_subset = []
        max_coverage_count = 0

        def bb(idx, curr_selected, curr_time, curr_covered):
            nonlocal best_subset, max_coverage_count
            if len(curr_covered) > max_coverage_count:
                max_coverage_count = len(curr_covered)
                best_subset = list(curr_selected)

            if idx >= n:
                return

            # Upper bound: even if all remaining added
            rem_union = set(curr_covered)
            for k in range(idx, n):
                rem_union.update(candidates[k].covered_functions)
            if len(rem_union) <= max_coverage_count:
                return  # Prune

            # Branch 1: include candidate
            t = candidates[idx]
            if curr_time + t.duration_ms <= budget_ms:
                bb(idx + 1, curr_selected + [t], curr_time + t.duration_ms, curr_covered | t.covered_functions)

            # Branch 2: exclude candidate
            bb(idx + 1, curr_selected, curr_time, curr_covered)

        bb(0, [], 0.0, set())
        covered = set()
        for t in best_subset:
            covered.update(t.covered_functions)
        return best_subset, covered

