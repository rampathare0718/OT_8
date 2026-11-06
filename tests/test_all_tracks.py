"""
Comprehensive Unit Test Suite for Assignment 8 (Tracks A to J)
Verifies correctness, boundary conditions, edge cases, and numerical stability.
"""

import unittest
import numpy as np

from track_a_continuous.landscapes import QuadraticConvex, Rosenbrock
from track_a_continuous.optimizers import GradientDescent, NewtonRaphson
from track_b_discrete.discrete_solvers import KnapsackProblem, DynamicProgrammingKnapsackSolver
from track_b_discrete.qubo_formulation import build_maxcut_qubo
from track_c_quantum_opt.qaoa_solver import QAOASolver
from track_c_quantum_opt.gradients import parameter_shift_gradient
from track_d_submodular_parallel.submodular import FacilityLocationCoverage
from track_e_case_study.bess_problem import BESSProblem
from track_e_case_study.solvers import ExactLPSolver
from track_f_compiler.mini_compiler import TACInstruction, TACInterpreter
from track_f_compiler.optimization_passes import ConstantFoldingAndPropagationPass, DeadCodeEliminationPass
from track_g_xv6_os.xv6_scheduler_sim import XV6SchedulerSimulator
from track_h_dbms.tpch_data import create_tpch_database
from track_i_quantum_algorithms.quantum_zoo import QuantumZoo
from track_j_software_engineering.test_suite_data import build_codebase_test_suite
from track_j_software_engineering.tsmp_solvers import calculate_apfd

class TestAssignment8(unittest.TestCase):
    def test_track_a_convex_convergence(self):
        quad = QuadraticConvex(dim=2, cond=5.0, seed=42)
        x0 = np.array([2.0, 2.0])
        opt = NewtonRaphson(lr=1.0, max_iter=5)
        x_opt, hist = opt.optimize(quad.objective, quad.gradient, x0, hess_fn=quad.hessian)
        self.assertLess(quad.objective(x_opt), 1e-4)

    def test_track_b_knapsack_dp(self):
        prob = KnapsackProblem([10, 20, 30], [1, 2, 3], capacity=3)
        sol, val, w = DynamicProgrammingKnapsackSolver().solve(prob)
        self.assertEqual(val, 30.0)
        self.assertLessEqual(w, 3.0)

    def test_track_c_parameter_shift_gradient(self):
        # f(th) = cos(th) => f'(th) = -sin(th)
        def circ_fn(theta):
            return float(np.cos(theta[0]))
        theta = np.array([np.pi / 4.0])
        grad = parameter_shift_gradient(circ_fn, theta)
        expected = -np.sin(np.pi / 4.0)
        np.testing.assert_allclose(grad[0], expected, atol=1e-5)

    def test_track_d_submodular_diminishing_returns(self):
        fl = FacilityLocationCoverage(n_candidates=6, n_clients=15, seed=42)
        passed, min_slack = fl.test_diminishing_returns(n_trials=30)
        self.assertTrue(passed)
        self.assertGreaterEqual(min_slack, -1e-9)

    def test_track_e_bess_lp_feasibility(self):
        prob = BESSProblem(seed=42)
        p_ch, p_dis, cost, profit, is_feas, soc = ExactLPSolver().solve(prob)
        self.assertTrue(is_feas)
        self.assertGreater(profit, 1000.0)
        self.assertTrue(all(soc >= prob.SoC_min - 1e-4))
        self.assertTrue(all(soc <= prob.SoC_max + 1e-4))

    def test_track_f_compiler_semantic_equivalence(self):
        prog = [
            TACInstruction("=", dest="a", src1=5),
            TACInstruction("=", dest="b", src1=10),
            TACInstruction("+", dest="c", src1="a", src2="b"),
            TACInstruction("*", dest="dead", src1=999, src2=999),
            TACInstruction("RETURN", src1="c")
        ]
        interp = TACInterpreter()
        ret_orig, _, _ = interp.execute(prog)

        p1 = ConstantFoldingAndPropagationPass().run(prog)
        p_opt = DeadCodeEliminationPass().run(p1)
        ret_opt, _, _ = interp.execute(p_opt)

        self.assertEqual(ret_orig, ret_opt)
        self.assertEqual(ret_opt, 15)

    def test_track_g_xv6_mlfq_simulation(self):
        sim = XV6SchedulerSimulator(scheduler_type="MLFQ", n_proc=4, seed=42)
        metrics = sim.run(max_ticks=500)
        self.assertGreater(metrics["Avg_Turnaround"], 0)
        self.assertGreater(metrics["Jains_Fairness_Index"], 0)

    def test_track_h_dbms_database(self):
        conn = create_tpch_database(seed=42)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM CUSTOMER;")
        count_c = cur.fetchone()[0]
        self.assertEqual(count_c, 1000)

    def test_track_i_deutsch_jozsa(self):
        zoo = QuantumZoo()
        is_const, _ = zoo.deutsch_jozsa(n=2, oracle_type="constant")
        self.assertTrue(is_const)
        is_bal_const, _ = zoo.deutsch_jozsa(n=2, oracle_type="balanced")
        self.assertFalse(is_bal_const)

    def test_track_j_apfd_metric(self):
        tests, funcs, faults = build_codebase_test_suite(seed=42)
        apfd = calculate_apfd(tests, faults)
        self.assertGreaterEqual(apfd, 0.0)
        self.assertLessEqual(apfd, 1.0)

if __name__ == "__main__":
    unittest.main()

