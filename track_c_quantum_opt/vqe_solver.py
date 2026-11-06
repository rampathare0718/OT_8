"""
Track C: VQE (Variational Quantum Eigensolver)
Implements:
- Hardware-efficient Parameterized Quantum Circuit (PQC)
- Ground-state energy estimation for 2-qubit and 4-qubit spin Hamiltonians
- Optimization via Parameter-Shift, SPSA, and Quantum Natural Gradient (QNG)
"""

import numpy as np
from .gradients import parameter_shift_gradient, spsa_gradient, quantum_natural_gradient_step

# 2x2 Pauli definitions
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def ry_gate(theta):
    """Single qubit Ry(theta) = cos(theta/2) I - i sin(theta/2) Y."""
    return np.array([
        [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
        [np.sin(theta / 2.0),  np.cos(theta / 2.0)]
    ], dtype=complex)

def cz_gate():
    """Controlled-Z gate between 2 qubits."""
    return np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)

class VQESolver:
    """
    VQE solver using Hardware-Efficient Ansatz (Ry layers + CZ entanglers).
    """
    def __init__(self, hamiltonian, n_qubits=2, layers=2, seed=42):
        self.H = np.array(hamiltonian, dtype=complex)
        self.n = n_qubits
        self.layers = layers
        self.num_params = self.n * (self.layers + 1)
        self.rng = np.random.default_rng(seed)
        self.exact_ground_energy = float(np.min(np.linalg.eigvalsh(self.H)))

    def get_state(self, params):
        dim = 2 ** self.n
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1.0  # |00...0>

        # Layer 0: Initial single-qubit Ry rotations
        idx = 0
        U_layer0 = ry_gate(params[idx])
        for q in range(1, self.n):
            U_layer0 = np.kron(U_layer0, ry_gate(params[idx + q]))
        idx += self.n
        psi = U_layer0 @ psi

        # Entangled layers
        CZ = cz_gate()
        for l in range(self.layers):
            # CZ entanglement on adjacent pairs (0-1)
            psi = CZ @ psi
            # Ry rotation layer
            U_rot = ry_gate(params[idx])
            for q in range(1, self.n):
                U_rot = np.kron(U_rot, ry_gate(params[idx + q]))
            idx += self.n
            psi = U_rot @ psi

        return psi

    def energy(self, params):
        psi = self.get_state(params)
        return float(np.real(psi.conj().T @ self.H @ psi))

    def optimize_parameter_shift(self, lr=0.1, max_iter=60):
        params = self.rng.uniform(-np.pi, np.pi, self.num_params)
        history = [self.energy(params)]

        for it in range(max_iter):
            grad = parameter_shift_gradient(self.energy, params)
            params = params - lr * grad
            history.append(self.energy(params))

        return params, history[-1], history

    def optimize_qng(self, lr=0.1, max_iter=40):
        params = self.rng.uniform(-np.pi, np.pi, self.num_params)
        history = [self.energy(params)]

        for it in range(max_iter):
            params, _ = quantum_natural_gradient_step(self.energy, self.get_state, params, lr=lr)
            history.append(self.energy(params))

        return params, history[-1], history

