# Live Demonstration Script: Assignment 8 (Optimize Anything)
**Project Title**: Optimization Techniques — The 14-Day Vibe-Coding Challenge  
**Demonstration Mode**: Live Terminal Walkthrough / Viva Presentation  

---

## 🎬 Part 1: Quick 2-Minute Live Demo Walkthrough

### 1. Show the Environment & Test Suite (15 seconds)
Run the automated test suite to prove that all modules pass edge case and regression checks:
```bash
python -m unittest discover tests
```
> **Speaker note**: *"Here, all 10 unit tests across continuous, discrete, quantum, and systems tracks pass in 0.25 seconds."*

### 2. Run the Master Benchmark Runner (20 seconds)
```bash
python run_all.py
```
> **Speaker note**: *"In just 8 seconds, `run_all.py` benchmarks all 10 tracks, reproduces every dataset, checks semantic equivalence, and updates the dashboard."*

---

## 🔬 Part 2: Highlighting the Core Improvements

### 1. Main Case Study (Track E: Battery Energy Storage Arbitrage)
- **The Problem**: A 100 MWh / 25 MW utility battery on the CAISO grid used a naive heuristic threshold rule to charge and discharge based on hourly electricity prices. It charged too early, missed peak price spikes, and risked exceeding the 40 MW substation feeder rating.
- **The Optimization**: We formulated a 72-variable Linear Program ($P_{ch}, P_{dis}, SoC \in \mathbb{R}^{24}$) incorporating battery degradation wear ($C_{deg} = \$3.5/\text{MWh}$), SoC conservation, and feeder limits, solved via the HiGHS simplex solver.
- **Evidence of Improvement**:
  - Baseline Heuristic: **\$2,200.61 net daily profit**
  - Exact Linear Program: **\$2,619.80 net daily profit**
  - **Gain**: **+\$419.19 / day (+19.05% gain)**, delivering **\$153,000 in additional annual revenue per site** with zero feeder violations.

### 2. Operating System Optimization (Track G: xv6 MLFQ Scheduler)
- **The Problem**: The default xv6 teaching OS uses a single-queue round-robin scheduler that scans the entire process table with spinlocks. CPU-bound jobs cause severe starvation and high response latency for interactive processes.
- **The Optimization**: We designed a 3-queue Multi-Level Feedback Queue (MLFQ) with quantum demotion ($Q_0=1\,\text{tick}, Q_1=2, Q_2=4$) and priority aging every 100 ticks.
- **Evidence of Improvement**:
  - Interactive response time slashed by **98.5%** (230.8 ticks down to 3.5 ticks).
  - Average turnaround time reduced by **28.1%** (294.1 ticks down to 211.4 ticks).
  - Process table lock contention dropped from **869 lock collisions to 0**.

### 3. Compiler Optimization (Track F: 4 Passes & Register Allocation)
- **The Problem**: TAC intermediate representation contains dead instructions, unrolled loops, and redundant common subexpressions.
- **The Optimization**: Implemented Constant Folding, CSE, Strength Reduction, and Fixed-Point Dead Code Elimination, plus Welsh-Powell and 0-1 ILP Register Allocation.
- **Evidence of Improvement**:
  - Instruction count cut by **54.2%** (24 instructions down to 11).
  - Virtual machine cycles reduced by **14.4%** (90 cycles down to 77).
  - Registers compressed from 7 virtual variables to 3 physical registers.
  - Semantic output verified bit-for-bit (both returned 264).

### 4. Quantum Machine Learning Reality Check (Track I)
- **The Finding**: On Grover Search, quantum query complexity is provably $O(\sqrt{N})$. However, on classical simulators, Variational Quantum Classifiers (VQC) required **9,020 circuit evaluations (444 ms)** compared to **0.88 ms** for classical Logistic Regression (a **500x simulation slowdown**).
- **Key Takeaway**: Demonstrates an honest negative result—QML provides no practical advantage without physical fault-tolerant hardware and fast quantum RAM (QRAM).

---

## 🛡️ Part 3: Vibe-Coding Governance & Failure Disclosures
If the evaluator asks how AI was used:
1. **AI Register**: Disclosed in `README.md` (Gemini 3.8 Flash, Claude 3.5 Sonnet).
2. **Commit Traceability**: Every git commit is prefixed with `[AI:...]` or `[Manual:...]`.
3. **Documented Failures**:
   - Compiler DCE stripped `i = i + 1` causing infinite loops (fixed with iterative fixpoint analysis).
   - xv6 simulator had a typo (`RUNABLE` vs `RUNNABLE`) stalling process queues (fixed with strict enum validation).
   - Quantum import statement had syntax error from inline ternary operator (fixed with clean direct imports).

