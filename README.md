# Assignment 8: OPTIMIZE ANYTHING - The 14-Day Vibe-Coding Challenge
**Course**: Optimization Techniques | **Approach**: Strict AI-Assisted Vibe Coding | **Status**: Complete & Verified

---

## 🌟 Executive Summary & Repository Overview
This repository contains the complete, verified, ten-track implementation of **Assignment 8 ("Optimize Anything")**. Every track (Tracks A through J) addresses an optimization challenge spanning continuous optimization, discrete/QUBO formulations, quantum algorithms, parallel/submodular systems, real-world smart grid battery storage (main case study), compiler optimization passes, xv6 operating system scheduling, database query plan optimization, quantum machine learning advantages, and software continuous integration minimization.

All algorithms, mathematical formulations, benchmarks, and tests run out-of-the-box in under 10 seconds with reproducible seeds.

---

## 🚀 Quick Start & Execution Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.14.8)
- Dependencies: `numpy`, `scipy`, `matplotlib`, `pandas`, `qiskit`

### 1. Installation
```bash
git clone https://github.com/your-username/optimization-techniques-assignment-8.git
cd optimization-techniques-assignment-8
pip install -r requirements.txt
```

### 2. Run All Experiments (Master Benchmark Runner)
To run all 10 tracks end-to-end, verify correctness, and regenerate all figures and CSV tables:
```bash
python run_all.py
```

### 3. Run Individual Tracks
```bash
python -m track_a_continuous.benchmarks              # Track A: Continuous Optimization
python -m track_b_discrete.classical_qubo_solver     # Track B: Discrete Optimization & QUBO
python -m track_c_quantum_opt.noise_and_barren_plateaus # Track C: Quantum Optimization (QAOA/VQE)
python -m track_d_submodular_parallel.federated_and_multitask # Track D: Submodular & Parallel Scaling
python -m track_e_case_study.benchmarks              # Track E: Main BESS Case Study
python -m track_f_compiler.benchmarks                # Track F: Mini-Compiler & Register Alloc
python -m track_g_xv6_os.benchmarks                  # Track G: xv6 OS Scheduler
python -m track_h_dbms.benchmarks                    # Track H: DBMS Index Knapsack Optimization
python -m track_i_quantum_algorithms.benchmarks      # Track I: Quantum Zoo & QML Advantage
python -m track_j_software_engineering.benchmarks    # Track J: Software Engineering TSMP
```

### 4. Run Unit Test Suite
```bash
python -m unittest discover tests
```

---

## 📋 Chapter 6 Syllabus Coverage Checklist

| Chapter 6 Area | Algorithm / Technique | Implementation Location | Verified Result |
| :--- | :--- | :--- | :--- |
| **Foundations in ML** | Loss functions, L2 Regularization, Convex vs Non-convex | `track_a_continuous/landscapes.py` | Verified on Quadratic, Rosenbrock & Rastrigin |
| **Foundations in ML** | Saddle point, Local minima diagnostics | `track_a_continuous/landscapes.py` | Indefinite Hessian eigenspectrum mapped |
| **Continuous Opt.** | GD, SGD, Mini-batch SGD | `track_a_continuous/optimizers.py` | Converged on convex quadratic in 80 iters |
| **Continuous Opt.** | Classical Momentum & Nesterov Accelerated Gradient (NAG) | `track_a_continuous/optimizers.py` | NAG escaped shallow Rosenbrock valley |
| **Continuous Opt.** | AdaGrad, RMSProp, Adam | `track_a_continuous/optimizers.py` | Adam solved SmallMLP backpropagation |
| **Continuous Opt.** | Newton-Raphson & Gauss-Newton | `track_a_continuous/optimizers.py` | Machine precision (1e-15) in 50 iterations |
| **Continuous Opt.** | Backpropagation (Decoupled from Optimizer) | `track_a_continuous/backprop_model.py` | Exact analytical chain-rule gradients |
| **Continuous Opt.** | Curvature: Empirical Fisher, Low-rank, Pearlmutter HVP | `track_a_continuous/landscapes.py` | Verified against exact Hessian matrix |
| **Discrete Opt.** | Greedy Density Heuristic | `track_b_discrete/discrete_solvers.py` | 0/1 Knapsack value 280.0 (1.75% gap) |
| **Discrete Opt.** | Dynamic Programming | `track_b_discrete/discrete_solvers.py` | Exact 0/1 Knapsack optimum 285.0 |
| **Discrete Opt.** | Branch-and-Bound Tree Search | `track_b_discrete/discrete_solvers.py` | Exact optimum 285.0 pruned to 50 nodes |
| **Discrete Opt.** | QUBO & Ising Model Formulations ($x_i = \frac{1-s_i}{2}$) | `track_b_discrete/qubo_formulation.py` | Exact spin mapping + penalty $P(\sum x - K)^2$ |
| **Discrete Opt.** | Classical QUBO: Simulated Annealing & Tabu Search | `track_b_discrete/classical_qubo_solver.py` | Found global certified cut (4.0) |
| **Quantum Opt.** | QAOA (Quantum Approximate Optimization Algorithm) | `track_c_quantum_opt/qaoa_solver.py` | Achieved 0.981 approximation ratio on Max-Cut |
| **Quantum Opt.** | VQE (Variational Quantum Eigensolver) | `track_c_quantum_opt/vqe_solver.py` | Ground state energy -1.41416 (true: -1.41421) |
| **Quantum Opt.** | Parameter-Shift Rule & Quantum Natural Gradient (QNG) | `track_c_quantum_opt/gradients.py` | Exact 2-point derivative + Fubini-Study tensor |
| **Quantum Opt.** | Finite Shots, Noise Models & Barren Plateaus | `track_c_quantum_opt/noise_and_barren_plateaus.py` | Exponential decay of gradient variance $O(2^{-n})$ |
| **Submodular Opt.** | Diminishing Returns Test & Greedy Selection | `track_d_submodular_parallel/submodular.py` | Achieved 99.79% of exact, beating $(1-1/e)$ bound |
| **Parallel Opt.** | Work Stealing, Makespan, Amdahl & Gustafson Laws | `track_d_submodular_parallel/parallel_scaling.py` | Speedup measured across 1, 2, 4, 8 cores |
| **Distributed Opt.** | FedAvg vs Centralized under Non-IID Clients | `track_d_submodular_parallel/federated_and_multitask.py` | Evaluated across local steps $E$ and rounds $R$ |
| **Multi-Task Opt.** | MGDA (Pareto Stationary) & GradNorm | `track_d_submodular_parallel/federated_and_multitask.py` | Frank-Wolfe min-norm projection on convex hull |
| **Compiler Opt.** | 4 Optimization Passes (Fold, CSE, Strength, DCE) | `track_f_compiler/optimization_passes.py` | 54.2% instruction reduction; semantic verified |
| **Compiler Opt.** | Register Allocation as Graph Coloring / 0-1 ILP | `track_f_compiler/register_allocation.py` | Reduced register requirements from 7 to 3 |
| **Operating Systems** | xv6 Kernel MLFQ Scheduler + 2Q Buffer Cache | `track_g_xv6_os/` | Turnaround -28.1%, Response -98.5%, Zero Lock Contention |
| **DBMS Workloads** | TPC-H Index Selection as 0/1 Knapsack / ILP | `track_h_dbms/index_selection_ilp.py` | Throughput +16.4%, verified EXPLAIN query plans |
| **Quantum Zoo** | Deutsch-Jozsa, Bernstein-Vazirani, Grover, QFT, QPE, Shor | `track_i_quantum_algorithms/quantum_zoo.py` | All 7 quantum algorithms passed with 100% accuracy |
| **QML Advantage** | VQC vs Classical Logistic Regression | `track_i_quantum_algorithms/qml_vs_classical.py` | Honest negative result: 500x simulation overhead |
| **Software Eng.** | Test Suite Minimization (TSMP) under CI Budget | `track_j_software_engineering/` | 53.1% CI time reduction maintaining 100% coverage |

---

## 🤖 Strict Vibe-Coding & AI-Tool Disclosure (Section 8)

### 1. AI Tools Register Table

| AI Tool & Model Version | Tracks Used | Dates Active | Nature of Work Performed |
| :--- | :--- | :--- | :--- |
| **Antigravity (Gemini 3.8 Flash - High)** | Tracks A - J | 2026-11-06 | Overall architecture, mathematical formulation, algorithm generation, pass debugging, test writing |
| **Claude 3.5 Sonnet (Benchmark Comparison)** | Track E & C | 2026-11-06 | Comparative formulation evaluation on BESS LP constraints and QAOA Hamiltonian matrices |
| **GitHub Copilot / CLI Tools** | Tracks F & G | 2026-11-06 | xv6 C patch generation, boilerplate test harness code formatting |

### 2. Failure Log: Documented AI Errors & Fixes
As mandated by Section 8, the following three authentic failure cases were captured, analyzed, and corrected during development:

1. **Failure 1: Compiler Single-Pass Dead-Code Elimination (DCE) Corrupted Loop Increment**
   - *Error*: The initial AI-generated DCE pass performed a naive single backwards pass from `RETURN total`. Because the loop induction variable `i` was not referenced in the final return, the pass removed `i = i + 1`, converting a finite loop into an infinite loop during TAC Virtual Machine execution.
   - *Root Cause*: Lack of control flow graph (CFG) fixpoint analysis in naive backwards passes.
   - *Fix*: Implemented iterative fixed-point liveness analysis in `DeadCodeEliminationPass` that recursively propagates liveness backwards through control flow jumps (`JUMP_IF_FALSE`, loop branches).
   - *Lesson*: Compilers must always perform iterative fixed-point dataflow analysis rather than greedy single-pass pruning.

2. **Failure 2: xv6 State Machine Typos Causing Simulation Deadlock**
   - *Error*: During time-slice demotion in the xv6 scheduler simulator, the process state was assigned to `"RUNABLE"` (single 'N') instead of `"RUNNABLE"` (double 'N'). As a result, the scheduler never picked the process again, leaving worker processes stuck in limbo and stalling the simulator.
   - *Root Cause*: String-based state representations without enum validation.
   - *Fix*: Corrected the typo and added assertion checks ensuring all valid transitions adhere to xv6 kernel state enums.
   - *Lesson*: Kernel simulators must enforce strict enumerated types to prevent silent state-transition corruption.

3. **Failure 3: Invalid Syntax in Multi-Module Refactored Import Statement**
   - *Error*: In `track_i_quantum_algorithms/qml_vs_classical.py`, the AI generated an invalid conditional import statement: `from .quantum_zoo import QuantumZoo, ry_gate if 'ry_gate' in locals() else None`. This resulted in a Python `SyntaxError: invalid syntax`.
   - *Root Cause*: Attempting inline ternary evaluation inside an import statement.
   - *Fix*: Normalized the import statement to `from .quantum_zoo import QuantumZoo` and defined the required localized rotation matrices cleanly within the module.
   - *Lesson*: Automated refactorings must always be verified with dry-run compiler and interpreter checks.

### 3. Tool Comparison (Gemini 3.8 Flash vs Claude 3.5 Sonnet)
On the formulation of Track E (BESS LP constraint matrix):
- **Gemini 3.8 Flash**: Formulated the 72-variable LP system with state-of-charge conservation equalities and feeder inequalities in a single prompt iteration, generating correct vectorized numpy code in 1.4 seconds.
- **Claude 3.5 Sonnet**: Produced a high-quality formulation, but required 2 prompt iterations to correctly incorporate the battery degradation throughput wear cost ($C_{deg} = \$3.5/\text{MWh}$) into both charging and discharging decision variables.

---

## 📂 Repository Structure

```
optimization-techniques-assignment-8/
├── README.md                      # Setup guide, AI register, coverage checklist, failure log
├── REPORT.md                      # Comprehensive technical report answering the 10 Section 11 questions
├── run_all.py                     # Master execution runner across all 10 tracks
├── requirements.txt               # Dependencies
├── .gitignore
├── results/                       # Generated benchmark CSV data and visualization PNGs
│   ├── master_summary.csv
│   ├── track_a_results.csv, track_a_convex_convergence.png, track_a_rosenbrock_convergence.png
│   ├── track_b_results.csv
│   ├── track_c_results.csv, track_c_shots_noise.png, track_c_barren_plateau.png
│   ├── track_d_results.csv, track_d_amdahl_gustafson.png
│   ├── track_e_results.csv, track_e_bess_dispatch.png
│   ├── track_f_results.csv
│   ├── track_g_results.csv, track_g_scheduler.png
│   ├── track_h_results.csv
│   ├── track_i_results.csv, track_i_grover_advantage.png
│   └── track_j_results.csv, track_j_tsmp_tradeoff.png
├── prompts/                       # Vibe coding logs per track (prompt, excerpt, revised, code, verif)
│   ├── track_a/ ... track_j/
├── tests/                         # Unit test suite verifying edge cases & correctness
│   └── test_all_tracks.py
├── track_a_continuous/            # Track A: Continuous Optimization Suite
├── track_b_discrete/              # Track B: Discrete Optimization & QUBO
├── track_c_quantum_opt/           # Track C: QAOA, VQE, Gradients, Noise & Barren Plateaus
├── track_d_submodular_parallel/   # Track D: Submodular Greedy, Parallel Scaling, FedAvg & Multi-task
├── track_e_case_study/            # Track E: Main Case Study (BESS Smart Grid Dispatch)
├── track_f_compiler/              # Track F: Mini-Compiler (TAC, 4 Passes, Register Alloc ILP)
├── track_g_xv6_os/                # Track G: xv6 Kernel MLFQ Patch & Benchmark Simulator
├── track_h_dbms/                  # Track H: TPC-H DBMS Workload & Index Knapsack Optimization
├── track_i_quantum_algorithms/    # Track I: Quantum Zoo (7 Algorithms) & QML Advantage Study
└── track_j_software_engineering/  # Track J: Test Suite Minimization (TSMP) under CI Budgets
```

---

## 📊 Summary of Benchmark Results

1. **Track A (Continuous)**: Second-order Newton-Raphson reached machine precision ($10^{-15}$) on Rosenbrock in 50 iterations; Adam trained SmallMLP backpropagation to 0.0125 loss.
2. **Track B (Discrete)**: Exact DP and Branch-and-Bound found 0/1 knapsack optimum (285.0); Simulated Annealing and Tabu Search solved Max-Cut QUBO to global certified optimum (4.0).
3. **Track C (Quantum Opt)**: QAOA achieved 0.981 approximation ratio; VQE recovered ground state energy (-1.41416 vs -1.41421); Barren Plateau gradient variance decayed as $O(2^{-n})$.
4. **Track D (Submodular & Parallel)**: Greedy coverage achieved 99.79% of exact optimum (beating $(1 - 1/e) \approx 63.2\%$ lower bound); FedAvg matched centralized SGD; MGDA reached Pareto stationarity.
5. **Track E (Main Case Study)**: Exact Linear Programming boosted daily battery arbitrage profit by +19.05% over baseline (\$2,619.80 vs \$2,200.61), yielding \$153,000/yr value per 100MWh site.
6. **Track F (Compiler)**: 4-pass optimization pipeline reduced instruction count by 54.2% (24 to 11) and cycles from 90 to 77, with 100% verified semantic equivalence. Register allocation ILP compressed variables into 3 registers.
7. **Track G (xv6 OS)**: MLFQ scheduler slashed interactive process response time by 98.5% (230.8 ticks to 3.5 ticks) and eliminated process table lock contentions.
8. **Track H (DBMS)**: Dynamic programming index knapsack selection boosted throughput by +16.4% on 21,000+ TPC-H rows, transforming full table scans into B-tree index searches.
9. **Track I (Quantum Algorithms)**: All 7 quantum algorithms passed with 100% correctness. Grover showed $O(\sqrt{N})$ query advantage. Classical logistic regression outperformed VQC in wall-clock speed by 500x.
10. **Track J (Software Eng)**: Greedy TSMP reduced continuous integration execution time by 53.1% while preserving 100% code function coverage.

