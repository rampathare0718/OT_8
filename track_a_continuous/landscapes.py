"""
Track A: Optimization Landscapes & Curvature Approximations
Provides convex, strongly convex, non-convex (Rosenbrock, Saddle point, Rastrigin)
and curvature approximation tools (HVP, Fisher, Low-rank).
"""

import numpy as np

class QuadraticConvex:
    """Convex / Strongly Convex Quadratic: f(x) = 0.5 * x^T A x - b^T x."""
    def __init__(self, A=None, b=None, dim=2, cond=10.0, seed=42):
        if A is None:
            rng = np.random.default_rng(seed)
            Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
            eigs = np.linspace(1.0, cond, dim)
            self.A = Q @ np.diag(eigs) @ Q.T
            self.b = rng.standard_normal(dim)
        else:
            self.A = np.array(A, dtype=float)
            self.b = np.array(b, dtype=float) if b is not None else np.zeros(self.A.shape[0])
        eigs = np.linalg.eigvalsh(self.A)
        self.mu = float(np.min(eigs))
        self.L = float(np.max(eigs))
        self.cond = self.L / self.mu

    def objective(self, x):
        return 0.5 * float(x.T @ self.A @ x - self.b.T @ x)

    def gradient(self, x):
        return self.A @ x - self.b

    def hessian(self, x):
        return self.A.copy()

class Rosenbrock:
    """Non-convex Rosenbrock function: f(x, y) = 100*(y - x^2)^2 + (1 - x)^2."""
    def objective(self, x):
        return 100.0 * (x[1] - x[0]**2)**2 + (1.0 - x[0])**2

    def gradient(self, x):
        g0 = -400.0 * x[0] * (x[1] - x[0]**2) - 2.0 * (1.0 - x[0])
        g1 = 200.0 * (x[1] - x[0]**2)
        return np.array([g0, g1])

    def hessian(self, x):
        h00 = -400.0 * x[1] + 1200.0 * x[0]**2 + 2.0
        h01 = -400.0 * x[0]
        h10 = -400.0 * x[0]
        h11 = 200.0
        return np.array([[h00, h01], [h10, h11]])

class SaddlePointLandscape:
    """Saddle point objective: f(x, y) = x^2 - y^2 (indefinite Hessian at origin)."""
    def objective(self, x):
        return float(x[0]**2 - x[1]**2)

    def gradient(self, x):
        return np.array([2.0 * x[0], -2.0 * x[1]])

    def hessian(self, x):
        return np.array([[2.0, 0.0], [0.0, -2.0]])

class RastriginMultimodal:
    """Multimodal non-convex objective with abundant local minima: f(x) = 10d + sum(x_i^2 - 10*cos(2*pi*x_i))."""
    def objective(self, x):
        d = len(x)
        return 10.0 * d + np.sum(x**2 - 10.0 * np.cos(2 * np.pi * x))

    def gradient(self, x):
        return 2.0 * x + 20.0 * np.pi * np.sin(2 * np.pi * x)

    def hessian(self, x):
        diag = 2.0 + 40.0 * (np.pi ** 2) * np.cos(2 * np.pi * x)
        return np.diag(diag)

# Curvature approximations
def hessian_vector_product(grad_fn, x, v, eps=1e-5):
    """
    Pearlmutter's HVP trick:
    H(x) v approx (grad(x + eps * v) - grad(x - eps * v)) / (2 * eps)
    Avoids O(d^2) Hessian storage for high-dimensional models.
    """
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return np.zeros_like(x)
    return (grad_fn(x + eps * v) - grad_fn(x - eps * v)) / (2.0 * eps)

def empirical_fisher_matrix(sample_gradients):
    """
    Empirical Fisher Information Matrix:
    F = 1/N sum_{i=1}^N g_i g_i^T.
    Serves as a positive semi-definite curvature approximation.
    """
    G = np.array(sample_gradients)  # shape (N, d)
    return (G.T @ G) / len(G)

def low_rank_hessian_approx(H, rank=2):
    """Low-rank spectral approximation of Hessian via top-k eigenpairs."""
    eigvals, eigvecs = np.linalg.eigh(H)
    idx = np.argsort(np.abs(eigvals))[::-1][:rank]
    H_low_rank = eigvecs[:, idx] @ np.diag(eigvals[idx]) @ eigvecs[:, idx].T
    return H_low_rank

