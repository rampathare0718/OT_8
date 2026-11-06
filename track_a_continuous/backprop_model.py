"""
Track A: Backpropagation & Neural Network Curvature
Demonstrates backpropagation as an exact gradient computation procedure
strictly distinct from the parameter update optimizer.
Computes per-sample gradients, Fisher Information, and Hessian-Vector Products.
"""

import numpy as np

class SmallMLP:
    """
    Two-layer Multi-Layer Perceptron for non-linear regression/classification:
    x -> Dense(in, hidden) -> ReLU -> Dense(hidden, out) -> Linear
    """
    def __init__(self, in_dim=2, hidden_dim=4, out_dim=1, seed=42):
        self.rng = np.random.default_rng(seed)
        # Xavier initialization
        self.W1 = self.rng.standard_normal((in_dim, hidden_dim)) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.standard_normal((hidden_dim, out_dim)) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(out_dim)

    def get_params_vector(self):
        """Flattens all parameters into a single vector theta."""
        return np.concatenate([self.W1.ravel(), self.b1.ravel(), self.W2.ravel(), self.b2.ravel()])

    def set_params_vector(self, theta):
        """Unflattens parameter vector theta into network layers."""
        idx = 0
        w1_size = self.W1.size
        self.W1 = theta[idx:idx+w1_size].reshape(self.W1.shape)
        idx += w1_size

        b1_size = self.b1.size
        self.b1 = theta[idx:idx+b1_size].reshape(self.b1.shape)
        idx += b1_size

        w2_size = self.W2.size
        self.W2 = theta[idx:idx+w2_size].reshape(self.W2.shape)
        idx += w2_size

        b2_size = self.b2.size
        self.b2 = theta[idx:idx+b2_size].reshape(self.b2.shape)

    def forward(self, X):
        """Forward pass caching intermediate activations."""
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.maximum(0, self.Z1)  # ReLU
        self.Z2 = self.A1 @ self.W2 + self.b2
        return self.Z2

    def compute_loss(self, X, y, reg=1e-4):
        """Mean Squared Error loss + L2 regularization."""
        preds = self.forward(X)
        diff = preds - y
        mse = 0.5 * np.mean(diff ** 2)
        l2 = 0.5 * reg * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return mse + l2

    def backpropagation(self, X, y, reg=1e-4):
        """
        Backpropagation algorithm:
        Computes analytical gradients dL/dtheta via chain rule.
        Strictly decoupled from optimizer update rules.
        """
        N = X.shape[0]
        preds = self.forward(X)
        dZ2 = (preds - y) / N  # shape (N, out_dim)

        dW2 = self.A1.T @ dZ2 + reg * self.W2
        db2 = np.sum(dZ2, axis=0)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.Z1 > 0).astype(float)  # ReLU derivative

        dW1 = X.T @ dZ1 + reg * self.W1
        db1 = np.sum(dZ1, axis=0)

        grad_vector = np.concatenate([dW1.ravel(), db1.ravel(), dW2.ravel(), db2.ravel()])
        return grad_vector

    def per_sample_gradients(self, X, y, reg=1e-4):
        """Computes individual sample gradients for Empirical Fisher calculation."""
        N = X.shape[0]
        grads = []
        for i in range(N):
            xi = X[i:i+1]
            yi = y[i:i+1]
            g = self.backpropagation(xi, yi, reg=reg)
            grads.append(g)
        return np.array(grads)

