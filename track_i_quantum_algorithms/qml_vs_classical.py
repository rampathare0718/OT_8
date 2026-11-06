"""
Track I: QML vs Classical Classification & Algorithmic Advantage Benchmarks
Compares:
1. Variational Quantum Classifier (VQC) vs Classical Logistic Regression
2. Grover Quantum Search vs Classical Linear Search (Query counts & Wall-clock time)
3. Hardware Noise Model (Depolarizing error) on Quantum Circuits
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .quantum_zoo import QuantumZoo

# Helper 2x2 gates
def ry(theta):
    return np.array([
        [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
        [np.sin(theta / 2.0),  np.cos(theta / 2.0)]
    ], dtype=complex)

CZ = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)
Z0 = np.kron(np.array([[1, 0], [0, -1]], dtype=complex), np.eye(2, dtype=complex))

class VariationalQuantumClassifier:
    """2-qubit Variational Quantum Classifier (VQC) with angle encoding and CNOT/CZ entangler."""
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.params = self.rng.uniform(-np.pi, np.pi, size=4)
        self.circuit_evaluations = 0

    def predict_circuit(self, x, params):
        self.circuit_evaluations += 1
        # 1. State preparation (angle encoding)
        psi0 = np.array([1, 0, 0, 0], dtype=complex)
        U_enc = np.kron(ry(x[0]), ry(x[1]))
        psi = U_enc @ psi0

        # 2. Parameterized ansatz: Ry(th0, th1) -> CZ -> Ry(th2, th3)
        U_rot1 = np.kron(ry(params[0]), ry(params[1]))
        U_rot2 = np.kron(ry(params[2]), ry(params[3]))

        psi = U_rot1 @ psi
        psi = CZ @ psi
        psi = U_rot2 @ psi

        # 3. Expectation value <Z0>
        exp_z0 = float(np.real(psi.conj().T @ Z0 @ psi))
        return exp_z0

    def fit(self, X, y, lr=0.1, epochs=30):
        self.circuit_evaluations = 0
        N = len(X)
        for ep in range(epochs):
            for i in range(N):
                xi = X[i]
                yi = y[i]

                # Parameter-shift gradient for each theta_k
                grads = np.zeros(4)
                for k in range(4):
                    p_plus = self.params.copy()
                    p_minus = self.params.copy()
                    p_plus[k] += np.pi / 2.0
                    p_minus[k] -= np.pi / 2.0
                    exp_plus = self.predict_circuit(xi, p_plus)
                    exp_minus = self.predict_circuit(xi, p_minus)
                    # d<Z0>/d_theta
                    grads[k] = 0.5 * (exp_plus - exp_minus)

                # Loss: 0.5 * (pred - yi)^2
                curr_pred = self.predict_circuit(xi, self.params)
                loss_grad = (curr_pred - yi) * grads
                self.params -= lr * loss_grad

    def score(self, X, y):
        preds = np.array([1 if self.predict_circuit(xi, self.params) >= 0 else -1 for xi in X])
        return float(np.mean(preds == y))

class ClassicalLogisticRegression:
    """Equivalent 2-parameter classical linear classifier: y = sign(w^T x + b)."""
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
        self.w = self.rng.standard_normal(2)
        self.b = 0.0
        self.evaluations = 0

    def fit(self, X, y, lr=0.1, epochs=30):
        N = len(X)
        for ep in range(epochs):
            for i in range(N):
                self.evaluations += 1
                xi = X[i]
                yi = y[i]
                margin = yi * (np.dot(self.w, xi) + self.b)
                # Perceptron / Logistic loss gradient
                if margin < 1.0:
                    self.w += lr * yi * xi
                    self.b += lr * yi

    def score(self, X, y):
        preds = np.array([1 if (np.dot(self.w, xi) + self.b) >= 0 else -1 for xi in X])
        return float(np.mean(preds == y))
