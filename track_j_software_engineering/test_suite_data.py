"""
Track J: Test Suite Minimization & Prioritization (TSMP)
Dataset & Codebase Test Mapping across all 10 tracks (2,500+ LOC codebase).
Defines tests, execution durations (ms), function coverage sets, and seeded faults.
"""

import numpy as np

class TestCase:
    def __init__(self, test_id, name, duration_ms, covered_functions, detected_faults):
        self.test_id = test_id
        self.name = name
        self.duration_ms = duration_ms
        self.covered_functions = set(covered_functions)
        self.detected_faults = set(detected_faults)

def build_codebase_test_suite(seed=42):
    rng = np.random.default_rng(seed)
    # 30 modular test cases spanning Tracks A to I
    test_names = [
        # Track A
        "test_gd_convergence", "test_adam_momentum", "test_newton_hessian", "test_mlp_backprop",
        # Track B
        "test_greedy_knapsack", "test_dp_knapsack", "test_bb_knapsack", "test_maxcut_qubo", "test_sa_qubo",
        # Track C
        "test_qaoa_cut", "test_vqe_ground_energy", "test_param_shift_rule", "test_qng_metric", "test_barren_plateaus",
        # Track D
        "test_submodular_diminishing", "test_greedy_nemhauser_bound", "test_work_stealing_speedup", "test_fedavg_convergence", "test_mgda_pareto",
        # Track E
        "test_bess_feasibility", "test_bess_lp_arbitrage", "test_bess_ga_dispatch",
        # Track F
        "test_tac_interpreter", "test_constant_folding", "test_cse_pass", "test_dce_pass", "test_reg_allocation_ilp",
        # Track G
        "test_xv6_round_robin", "test_xv6_mlfq_priority", "test_xv6_fairness_index",
        # Track H
        "test_tpch_schema_gen", "test_index_knapsack_selection", "test_explain_query_plan",
        # Track I
        "test_deutsch_jozsa", "test_grover_search", "test_shor_order_finding", "test_vqc_classification"
    ]

    total_functions = 60
    total_faults = 12

    tests = []
    for idx, name in enumerate(test_names):
        # Durations between 5ms and 150ms
        dur = float(rng.integers(8, 120))
        # Coverage: 3 to 12 functions
        n_cov = rng.integers(4, 14)
        cov = rng.choice(total_functions, size=n_cov, replace=False)
        # Detected faults: each test catches 0 to 3 seeded faults
        n_flt = rng.integers(0, 3)
        flts = rng.choice(total_faults, size=n_flt, replace=False) if n_flt > 0 else []

        tests.append(TestCase(idx, name, dur, cov, flts))

    return tests, total_functions, total_faults

