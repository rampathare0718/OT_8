"""
Track A: Continuous Optimization Suite
Implements Chapter 6 continuous optimization algorithms:
- First-order: GD, SGD, Mini-batch SGD, Momentum, NAG, AdaGrad, RMSProp, Adam
- Second-order: Newton-Raphson, Gauss-Newton
"""

import numpy as np

class ContinuousOptimizer:
    """Base class for continuous optimizers."""
    def __init__(self, lr=0.01, max_iter=200, tol=1e-6, seed=42):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(seed)
        self.history = []

    def optimize(self, objective_fn, grad_fn, x0, **kwargs):
        raise NotImplementedError

class GradientDescent(ContinuousOptimizer):
    """Standard Batch Gradient Descent (GD)."""
    def optimize(self, objective_fn, grad_fn, x0, **kwargs):
        x = np.array(x0, dtype=float, copy=True)
        self.history = [x.copy()]
        for it in range(self.max_iter):
            g = grad_fn(x)
            if np.linalg.norm(g) < self.tol:
                break
            x = x - self.lr * g
            self.history.append(x.copy())
        return x, self.history

class SGD(ContinuousOptimizer):
    """Stochastic Gradient Descent (with optional batch/stochastic noise simulation)."""
    def __init__(self, lr=0.01, batch_size=1, noise_std=0.05, max_iter=200, tol=1e-6, seed=42):
        super().__init__(lr, max_iter, tol, seed)
        self.batch_size = batch_size
        self.noise_std = noise_std

    def optimize(self, objective_fn, grad_fn, x0, data=None, **kwargs):
        x = np.array(x0, dtype=float, copy=True)
        self.history = [x.copy()]
        for it in range(self.max_iter):
            if data is not None and hasattr(grad_fn, "batch_grad"):
                # Real mini-batch gradient
                X, y = data
                idx = self.rng.choice(len(X), size=min(self.batch_size, len(X)), replace=False)
                g = grad_fn.batch_grad(x, X[idx], y[idx])
            else:
                # Stochastic gradient with controlled noise variance
                noise = self.rng.normal(0, self.noise_std, size=x.shape)
                g = grad_fn(x) + noise
            x = x - self.lr * g
            self.history.append(x.copy())
            if np.linalg.norm(g) < self.tol:
                break
        return x, self.history

class Momentum(ContinuousOptimizer):
    """Polyak Classical Momentum Optimizer."""
    def __init__(self, lr=0.01, beta=0.9, max_iter=200, tol=1e-6, seed=42):
        super().__init__(lr, max_iter, tol, seed)
        self.beta = beta

    def optimize(self, objective_fn, grad_fn, x0, **kwargs):
        x = np.array(x0, dtype=float, copy=True)
        v = np.zeros_like(x)
        self.history = [x.copy()]
        for it in range(self.max_iter):
            g = grad_fn(x)
            v = self.beta * v + self.lr * g
            x = x - v
            self.history.append(x.copy())
            if np.linalg.norm(g) < self.tol:
                break
        return x, self.history

class NAG(ContinuousOptimizer):
    """Nesterov Accelerated Gradient (look-ahead momentum)."""
    def __init__(self, lr=0.01, beta=0.9, max_iter=200, tol=1e-6, seed=42):
        super().__init__(lr, max_iter, tol, seed)
        self.beta = beta

    def optimize(self, objective_fn, grad_fn, x0, **kwargs):
        x = np.array(x0, dtype=float, copy=True)
        v = np.zeros_like(x)
        self.history = [x.copy()]
        for it in range(self.max_iter):
            # Look-ahead evaluation point
            x_lookahead = x - self.beta * v
            g = grad_fn(x_lookahead)
            v = self.beta * v + self.lr * g
            x = x - v
            self.history.append(x.copy())
            if np.linalg.norm(g) < self.tol:
                break
        return x, self.history

class AdaGrad(ContinuousOptimizer):
    """Adaptive Gradient Algorithm (sum of squared historical gradients)."""
    def __init__(self, lr=0.1, eps=1e-8, max_iter=200, tol=1e-6, seed=42):
        super().__init__(lr, max_iter, tol, seed)
        self.eps = eps

    def optimize(self, objective_fn, grad_fn, x0, **kwargs):
        x = np.array(x0, dtype=float, copy=True)
        G = np.zeros_like(x)
        self.history = [x.copy()]
        for it in range(self.max_iter):
            g = grad_fn(x)
            G += g ** 2
            x = x - (self.lr / (np.sqrt(G) + self.eps)) * g
            self.history.append(x.copy())
            if np.linalg.norm(g) < self.tol:
                break
        return x, self.history

class RMSProp(ContinuousOptimizer):
    """Root Mean Square Propagation (exponentially decaying average of squared gradients)."""
    def __init__(self, lr=0.01, decay_rate=0.9, eps=1e-8, max_iter=200, tol=1e-6, seed=42):
        super().__init__(lr, max_iter, tol, seed)
        self.decay_rate = decay_rate
        self.eps = eps

    def optimize(self, objective_fn, grad_fn, x0, **kwargs):
        x = np.array(x0, dtype=float, copy=True)
        s = np.zeros_like(x)
        self.history = [x.copy()]
        for it in range(self.max_iter):
            g = grad_fn(x)
            s = self.decay_rate * s + (1.0 - self.decay_rate) * (g ** 2)
            x = x - (self.lr / (np.sqrt(s) + self.eps)) * g
            self.history.append(x.copy())
            if np.linalg.norm(g) < self.tol:
                break
        return x, self.history

class Adam(ContinuousOptimizer):
    """Adaptive Moment Estimation (first and second moments with bias correction)."""
    def __init__(self, lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=200, tol=1e-6, seed=42):
        super().__init__(lr, max_iter, tol, seed)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def optimize(self, objective_fn, grad_fn, x0, **kwargs):
        x = np.array(x0, dtype=float, copy=True)
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        self.history = [x.copy()]
        for it in range(1, self.max_iter + 1):
            g = grad_fn(x)
            m = self.beta1 * m + (1.0 - self.beta1) * g
            v = self.beta2 * v + (1.0 - self.beta2) * (g ** 2)
            m_hat = m / (1.0 - self.beta1 ** it)
            v_hat = v / (1.0 - self.beta2 ** it)
            x = x - (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat
            self.history.append(x.copy())
            if np.linalg.norm(g) < self.tol:
                break
        return x, self.history

class NewtonRaphson(ContinuousOptimizer):
    """Classical Second-Order Newton's Method using exact Hessian inversion with regularization."""
    def __init__(self, lr=1.0, reg=1e-4, max_iter=100, tol=1e-6, seed=42):
        super().__init__(lr, max_iter, tol, seed)
        self.reg = reg

    def optimize(self, objective_fn, grad_fn, x0, hess_fn=None, **kwargs):
        x = np.array(x0, dtype=float, copy=True)
        self.history = [x.copy()]
        for it in range(self.max_iter):
            g = grad_fn(x)
            if np.linalg.norm(g) < self.tol:
                break
            H = hess_fn(x)
            # Damped Hessian for stability when non-positive definite
            H_damped = H + self.reg * np.eye(len(x))
            try:
                step = np.linalg.solve(H_damped, g)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(H_damped) @ g
            x = x - self.lr * step
            self.history.append(x.copy())
        return x, self.history

class GaussNewton(ContinuousOptimizer):
    """
    Gauss-Newton method for non-linear least squares:
    min 0.5 * ||r(x)||^2.
    Approximates Hessian by J(x)^T J(x).
    """
    def __init__(self, lr=1.0, reg=1e-5, max_iter=100, tol=1e-6, seed=42):
        super().__init__(lr, max_iter, tol, seed)
        self.reg = reg

    def optimize(self, residual_fn, jacobian_fn, x0, **kwargs):
        x = np.array(x0, dtype=float, copy=True)
        self.history = [x.copy()]
        for it in range(self.max_iter):
            r = residual_fn(x)
            J = jacobian_fn(x)
            grad = J.T @ r
            if np.linalg.norm(grad) < self.tol:
                break
            H_approx = J.T @ J + self.reg * np.eye(len(x))
            try:
                step = np.linalg.solve(H_approx, grad)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(H_approx) @ grad
            x = x - self.lr * step
            self.history.append(x.copy())
        return x, self.history

