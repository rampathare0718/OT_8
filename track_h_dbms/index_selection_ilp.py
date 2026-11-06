"""
Track H: Index Selection Formulated as 0/1 Knapsack / ILP
Formulates index configuration selection under a physical storage budget:
Variables: x_i in {0, 1} indicating whether index i is created
Constraint: sum_{i} Size_i * x_i <= Storage_Budget_KB
Objective: maximize sum_{i} Latency_Benefit_ms * x_i
Solvers:
1. Default Baseline (No secondary indexes)
2. Greedy Benefit-to-Storage Ratio Heuristic
3. Dynamic Programming Exact Knapsack
4. Integer Linear Programming (0-1 ILP)
"""

import numpy as np

class CandidateIndex:
    def __init__(self, name, sql, size_kb, estimated_benefit_ms):
        self.name = name
        self.sql = sql
        self.size_kb = size_kb
        self.benefit = estimated_benefit_ms

def get_candidate_indexes():
    return [
        CandidateIndex(
            "idx_orders_custkey",
            "CREATE INDEX idx_orders_custkey ON ORDERS(o_custkey);",
            size_kb=120,
            estimated_benefit_ms=45.0
        ),
        CandidateIndex(
            "idx_orders_date",
            "CREATE INDEX idx_orders_date ON ORDERS(o_orderdate);",
            size_kb=110,
            estimated_benefit_ms=30.0
        ),
        CandidateIndex(
            "idx_cust_segment",
            "CREATE INDEX idx_cust_segment ON CUSTOMER(c_mktsegment);",
            size_kb=45,
            estimated_benefit_ms=25.0
        ),
        CandidateIndex(
            "idx_lineitem_orderkey",
            "CREATE INDEX idx_lineitem_orderkey ON LINEITEM(l_orderkey);",
            size_kb=320,
            estimated_benefit_ms=80.0
        ),
        CandidateIndex(
            "idx_lineitem_shipdate",
            "CREATE INDEX idx_lineitem_shipdate ON LINEITEM(l_shipdate);",
            size_kb=280,
            estimated_benefit_ms=35.0
        ),
        CandidateIndex(
            "idx_orders_composite",
            "CREATE INDEX idx_orders_composite ON ORDERS(o_custkey, o_orderdate);",
            size_kb=190,
            estimated_benefit_ms=65.0
        )
    ]

def solve_index_selection_greedy(candidates, budget_kb):
    """Greedy selection by benefit/size ratio."""
    ratios = [c.benefit / c.size_kb for c in candidates]
    order = np.argsort(ratios)[::-1]

    selected = []
    total_size = 0
    total_benefit = 0.0

    for idx in order:
        c = candidates[idx]
        if total_size + c.size_kb <= budget_kb:
            selected.append(c)
            total_size += c.size_kb
            total_benefit += c.benefit

    return selected, total_benefit, total_size

def solve_index_selection_dp(candidates, budget_kb):
    """Exact Dynamic Programming 0/1 knapsack."""
    n = len(candidates)
    W = int(budget_kb)
    dp = np.zeros((n + 1, W + 1), dtype=float)

    for i in range(1, n + 1):
        c = candidates[i - 1]
        w_i = int(c.size_kb)
        v_i = c.benefit
        for w in range(W + 1):
            if w >= w_i:
                dp[i, w] = max(dp[i - 1, w], dp[i - 1, w - w_i] + v_i)
            else:
                dp[i, w] = dp[i - 1, w]

    # Backtrack
    selected = []
    curr_w = W
    for i in range(n, 0, -1):
        if dp[i, curr_w] != dp[i - 1, curr_w]:
            selected.append(candidates[i - 1])
            curr_w -= int(candidates[i - 1].size_kb)

    total_size = sum(c.size_kb for c in selected)
    total_benefit = sum(c.benefit for c in selected)
    return selected, total_benefit, total_size

