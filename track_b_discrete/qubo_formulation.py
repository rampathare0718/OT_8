"""
Track B: QUBO and Ising Formulations
Implements:
1. Binary-to-Spin transformation (QUBO <-> Ising Hamiltonian)
2. Quadratic penalty terms for equality and inequality constraints
3. Max-Cut graph partitioning QUBO / Ising model
4. Constrained Feature Selection QUBO model
"""

import numpy as np

def binary_to_spin(x):
    """Maps binary x in {0, 1} to spin s in {+1, -1}: s = 1 - 2x."""
    return 1 - 2 * np.array(x)

def spin_to_binary(s):
    """Maps spin s in {+1, -1} to binary x in {0, 1}: x = (1 - s) / 2."""
    return ((1 - np.array(s)) / 2).astype(int)

class QUBOMatrix:
    """
    Represents a Quadratic Unconstrained Binary Optimization problem:
    min x^T Q x, x in {0, 1}^N.
    """
    def __init__(self, Q):
        self.Q = np.array(Q, dtype=float)
        # Ensure symmetric representation
        self.Q = (self.Q + self.Q.T) / 2.0
        self.n = self.Q.shape[0]

    def energy(self, x):
        """Computes objective value x^T Q x."""
        x = np.array(x, dtype=float)
        return float(x.T @ self.Q @ x)

    def to_ising(self):
        """
        Converts QUBO (x in {0,1}) to Ising Hamiltonian H(s) = sum h_i s_i + sum J_ij s_i s_j + offset.
        Using substitution: x_i = (1 - s_i) / 2.
        """
        n = self.n
        J = np.zeros((n, n), dtype=float)
        h = np.zeros(n, dtype=float)
        offset = 0.0

        for i in range(n):
            for j in range(n):
                q_ij = self.Q[i, j]
                # x_i x_j = (1 - s_i)(1 - s_j) / 4 = (1 - s_i - s_j + s_i s_j) / 4
                offset += q_ij / 4.0
                h[i] -= q_ij / 4.0
                h[j] -= q_ij / 4.0
                if i != j:
                    J[i, j] += q_ij / 4.0
                else:
                    # For i == j, s_i^2 = 1
                    pass

        # Since J is symmetric, upper triangle holds pairwise couplings
        return h, J, offset

def build_maxcut_qubo(adjacency_matrix):
    """
    Builds QUBO matrix for Max-Cut on graph G:
    Max-Cut objective: max sum_{(i,j) in E} (x_i + x_j - 2 x_i x_j)
    Equivalent minimization: min sum_{(i,j) in E} (2 x_i x_j - x_i - x_j)
    """
    A = np.array(adjacency_matrix, dtype=float)
    n = A.shape[0]
    Q = np.zeros((n, n), dtype=float)

    for i in range(n):
        degree_i = np.sum(A[i, :])
        Q[i, i] -= degree_i  # Linear term from -x_i
        for j in range(i + 1, n):
            if A[i, j] > 0:
                Q[i, j] += 2.0 * A[i, j]  # Quadratic cross term +2 x_i x_j
                Q[j, i] += 2.0 * A[i, j]

    return QUBOMatrix(Q)

def build_feature_selection_qubo(relevance, cross_correlation, target_k=3, penalty_P=10.0):
    """
    Feature Selection as QUBO:
    min -sum relevance_i x_i + alpha sum_{i<j} correlation_ij x_i x_j + P * (sum x_i - K)^2
    """
    n = len(relevance)
    Q = np.zeros((n, n), dtype=float)

    # 1. Relevance: -relevance_i * x_i
    for i in range(n):
        Q[i, i] -= relevance[i]

    # 2. Redundancy: correlation_ij * x_i x_j
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] += cross_correlation[i, j]
            Q[j, i] += cross_correlation[i, j]

    # 3. Penalty term: P * (sum x_i - K)^2
    # Expanding: P * ( sum x_i^2 + 2 sum_{i<j} x_i x_j - 2 K sum x_i + K^2 )
    # Since x_i^2 = x_i for binary variables:
    for i in range(n):
        Q[i, i] += penalty_P * (1.0 - 2.0 * target_k)
        for j in range(i + 1, n):
            Q[i, j] += 2.0 * penalty_P
            Q[j, i] += 2.0 * penalty_P

    return QUBOMatrix(Q), penalty_P * (target_k ** 2)

