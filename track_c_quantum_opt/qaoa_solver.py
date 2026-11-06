"""
Track C: QAOA (Quantum Approximate Optimization Algorithm)
Solves the Track B Max-Cut QUBO using QAOA with:
- Cost unitary U(C, gamma) = exp(-i gamma H_C)
- Mixer unitary U(M, beta) = exp(-i beta H_M)
- Parameter optimization via Parameter-Shift & SPSA
- Comparison against Track B classical baseline
"""

import numpy as np
from scipy.optimize import minimize
from .gradients import parameter_shift_gradient, spsa_gradient

# Pauli matrices
I2 = np.eye(2, dtype=complex)
X_gate = np.array([[0, 1], [1, 0]], dtype=complex)
Y_gate = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_gate = np.array([[1, 0], [0, -1]], dtype=complex)

def tensor_product_list(matrices):
    res = matrices[0]
    for m in matrices[1:]:
        res = np.kron(res, m)
    return res

class QAOASolver:
    """
    Simulates QAOA for Max-Cut on an arbitrary graph.
    Supports statevector exact expectation and finite shot sampling.
    """
    def __init__(self, adjacency_matrix, p_layers=1, shots=None, seed=42):
        self.A = np.array(adjacency_matrix, dtype=float)
        self.n = self.A.shape[0]
        self.p = p_layers
        self.shots = shots
        self.rng = np.random.default_rng(seed)
        self._build_hamiltonians()

    def _build_hamiltonians(self):
        n = self.n
        dim = 2 ** n

        # Cost Hamiltonian H_C = sum_{(i,j) in E} 0.5 * (I - Z_i Z_j)
        self.H_C = np.zeros((dim, dim), dtype=complex)
        for i in range(n):
            for j in range(i + 1, n):
                if self.A[i, j] > 0:
                    z_ops = [I2] * n
                    z_ops[i] = Z_gate
                    z_ops[j] = Z_gate
                    ZZ = tensor_product_list(z_ops)
                    self.H_C += 0.5 * self.A[i, j] * (np.eye(dim) - ZZ)

        # Mixer Hamiltonian H_M = sum_i X_i
        self.H_M = np.zeros((dim, dim), dtype=complex)
        for i in range(n):
            x_ops = [I2] * n
            x_ops[i] = X_gate
            self.H_M += tensor_product_list(x_ops)

    def get_state(self, params):
        """Constructs state |gamma, beta> = prod_{k=1}^p [e^{-i beta_k H_M} e^{-i gamma_k H_C}] |+>^n."""
        gamma = params[:self.p]
        beta = params[self.p:]

        # Initial state |+>^n
        dim = 2 ** self.n
        psi = np.ones(dim, dtype=complex) / np.sqrt(dim)

        for k in range(self.p):
            # Cost evolution (diagonal in computational basis)
            diag_HC = np.diag(self.H_C)
            psi = np.exp(-1j * gamma[k] * diag_HC) * psi

            # Mixer evolution
            # e^{-i beta H_M} = prod_i (cos(beta) I - i sin(beta) X_i)
            # Since H_M terms commute, we can exponentiate directly or via matrix exponential
            w, v = np.linalg.eigh(self.H_M)
            U_M = v @ np.diag(np.exp(-1j * beta[k] * w)) @ v.conj().T
            psi = U_M @ psi

        return psi

    def expectation_value(self, params):
        """Computes <psi | H_C | psi>."""
        psi = self.get_state(params)
        probs = np.abs(psi) ** 2

        if self.shots is None:
            # Exact statevector expectation
            return float(np.real(psi.conj().T @ self.H_C @ psi))
        else:
            # Finite measurement shot simulation
            dim = 2 ** self.n
            samples = self.rng.choice(dim, size=self.shots, p=probs)
            diag_HC = np.real(np.diag(self.H_C))
            return float(np.mean(diag_HC[samples]))

    def objective_to_minimize(self, params):
        """Max-Cut is maximization, so objective is -<H_C>."""
        return -self.expectation_value(params)

    def optimize(self, method="COBYLA", maxiter=100):
        # Initial parameters
        init_params = np.concatenate([
            self.rng.uniform(0, np.pi, self.p),
            self.rng.uniform(0, np.pi / 2, self.p)
        ])

        res = minimize(
            self.objective_to_minimize,
            init_params,
            method=method,
            options={"maxiter": maxiter}
        )

        opt_params = res.x
        opt_cut = -res.fun

        # Find most probable bitstring
        final_psi = self.get_state(opt_params)
        probs = np.abs(final_psi) ** 2
        best_bit_idx = np.argmax(probs)
        best_bitstring = [(best_bit_idx >> (self.n - 1 - i)) & 1 for i in range(self.n)]

        return opt_params, opt_cut, best_bitstring, probs[best_bit_idx]

