"""
Track F: Register Allocation as Combinatorial Optimization
Formulates register allocation as Graph Coloring / 0-1 ILP:
- Variables: Virtual variables v in V
- Graph: Interference graph G = (V, E) where edges denote overlapping live ranges
- Solvers:
  1. Greedy Kempe / Welsh-Powell Heuristic
  2. Exact 0-1 Integer Linear Programming (ILP)
"""

import numpy as np

class InterferenceGraph:
    """Interference graph where nodes are virtual registers and edges denote interference."""
    def __init__(self, variables):
        self.variables = list(variables)
        self.n = len(self.variables)
        self.var_to_idx = {v: i for i, v in enumerate(self.variables)}
        self.adj = np.zeros((self.n, self.n), dtype=int)

    def add_interference(self, v1, v2):
        if v1 in self.var_to_idx and v2 in self.var_to_idx and v1 != v2:
            i1 = self.var_to_idx[v1]
            i2 = self.var_to_idx[v2]
            self.adj[i1, i2] = 1
            self.adj[i2, i1] = 1

    def solve_greedy(self):
        """Greedy Welsh-Powell graph coloring heuristic."""
        degrees = np.sum(self.adj, axis=1)
        order = np.argsort(degrees)[::-1]  # Highest degree first

        colors = -np.ones(self.n, dtype=int)
        for u in order:
            neighbor_colors = {colors[v] for v in range(self.n) if self.adj[u, v] == 1 and colors[v] != -1}
            # Assign lowest available color
            c = 0
            while c in neighbor_colors:
                c += 1
            colors[u] = c

        total_registers_needed = int(np.max(colors) + 1)
        allocation = {self.variables[i]: int(colors[i]) for i in range(self.n)}
        return allocation, total_registers_needed

    def solve_ilp(self, max_colors=10):
        """
        Exact 0-1 Integer Linear Programming solver via Branch-and-Bound:
        min sum_{c=1}^K y_c
        s.t. sum_c x_{v,c} = 1
             x_{u,c} + x_{v,c} <= 1 for (u,v) in E
             x_{v,c} <= y_c
        """
        # Initialize with greedy upper bound for aggressive pruning
        _, greedy_k = self.solve_greedy()
        min_k = greedy_k
        best_allocation = None

        current_colors = -np.ones(self.n, dtype=int)

        def can_color(node, c):
            for v in range(self.n):
                if self.adj[node, v] == 1 and current_colors[v] == c:
                    return False
            return True

        def bb_search(node, max_used_color):
            nonlocal best_allocation, min_k
            if max_used_color >= min_k:
                return  # Prune branch

            if node == self.n:
                min_k = max_used_color
                best_allocation = current_colors.copy()
                return

            for c in range(min(max_used_color + 1, min_k)):
                if can_color(node, c):
                    current_colors[node] = c
                    bb_search(node + 1, max(max_used_color, c + 1))
                    current_colors[node] = -1

        bb_search(0, 0)
        if best_allocation is None:
            return self.solve_greedy()
        allocation = {self.variables[i]: int(best_allocation[i]) for i in range(self.n)}
        return allocation, min_k
