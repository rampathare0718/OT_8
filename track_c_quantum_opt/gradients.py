"""
Track C: Quantum Gradient Computation
Implements:
1. Exact Parameter-Shift Rule: d<H>/d_theta = (<H>(theta + pi/2) - <H>(theta - pi/2)) / 2
2. SPSA (Simultaneous Perturbation Stochastic Approximation) for noisy quantum evaluations
3. Quantum Natural Gradient (QNG) using Fubini-Study metric tensor F(theta)
"""

import numpy as np

def parameter_shift_gradient(circuit_fn, params, shift=np.pi / 2.0):
    """
    Evaluates exact analytic gradient of expectation value <H>(theta)
    using the 2-point parameter-shift rule for Pauli generators:
    d<H>/d theta_i = (<H>(theta + (pi/2) e_i) - <H>(theta - (pi/2) e_i)) / 2
    """
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += shift
        params_minus[i] -= shift

        val_plus = circuit_fn(params_plus)
        val_minus = circuit_fn(params_minus)
        grad[i] = (val_plus - val_minus) / (2.0 * np.sin(shift))
    return grad

def spsa_gradient(circuit_fn, params, ck=0.1, seed=None):
    """
    SPSA gradient approximation:
    Requires only 2 evaluations regardless of parameter dimension!
    Delta_k drawn from Rademacher distribution {-1, +1}^p.
    g_k = (circuit_fn(theta + ck * Delta) - circuit_fn(theta - ck * Delta)) / (2 * ck * Delta)
    """
    rng = np.random.default_rng(seed)
    delta = rng.choice([-1.0, 1.0], size=len(params))
    val_plus = circuit_fn(params + ck * delta)
    val_minus = circuit_fn(params - ck * delta)
    grad = (val_plus - val_minus) / (2.0 * ck * delta)
    return grad

def fubini_study_metric(state_fn, params, eps=1e-3):
    """
    Computes Fubini-Study metric tensor F_ij(theta) = Re[<d_i psi | d_j psi> - <d_i psi | psi><psi | d_j psi>].
    Used by Quantum Natural Gradient (QNG) to navigate quantum state manifold.
    """
    p = len(params)
    psi0 = state_fn(params)
    dpsi = []
    for i in range(p):
        p_plus = params.copy()
        p_plus[i] += eps
        psi_plus = state_fn(p_plus)
        dpsi_i = (psi_plus - psi0) / eps
        dpsi.append(dpsi_i)

    F = np.zeros((p, p), dtype=float)
    for i in range(p):
        for j in range(p):
            inner_di_dj = np.vdot(dpsi[i], dpsi[j])
            inner_di_psi = np.vdot(dpsi[i], psi0)
            inner_psi_dj = np.vdot(psi0, dpsi[j])
            F[i, j] = np.real(inner_di_dj - inner_di_psi * inner_psi_dj)

    return F

def quantum_natural_gradient_step(circuit_fn, state_fn, params, lr=0.1, reg=1e-3):
    """Computes QNG update: theta_{t+1} = theta_t - lr * (F + reg * I)^{-1} grad."""
    grad = parameter_shift_gradient(circuit_fn, params)
    F = fubini_study_metric(state_fn, params)
    F_reg = F + reg * np.eye(len(params))
    try:
        nat_grad = np.linalg.solve(F_reg, grad)
    except np.linalg.LinAlgError:
        nat_grad = np.linalg.pinv(F_reg) @ grad
    return params - lr * nat_grad, grad

