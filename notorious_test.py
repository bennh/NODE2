# notorious_test.py

import numpy as np
import casadi as ca

def make_notorious_integrator(mu: float = 60.0):
    """
    Create an integrator for the Section 6 'notorious test example'.

    System:
        dx1/dt = x2
        dx2/dt = mu^2 * x1 - (mu^2 + p^2) * sin(p * t)

    Initial conditions:
        x1(0) = 0
        x2(0) = pi

    Parameter:
        p: to be estimated
    """

    t = ca.MX.sym('t')
    x = ca.MX.sym('x', 2)
    p = ca.MX.sym('p')

    x1, x2 = x[0], x[1]

    rhs = ca.vertcat(
        x2,
        mu**2 * x1 - (mu**2 + p**2) * ca.sin(p * t)
    )

    dae = {'x': x, 'p': ca.vertcat(t, p), 'ode': rhs}
    opts = {'tf': 0.1}
    integrator = ca.integrator('integrator', 'cvodes', dae, opts)

    return integrator

def get_initial_state():
    """Return the initial state for the test system."""
    return np.array([0.0, np.pi])

def get_true_parameters():
    """Return the true value of parameter p."""
    return np.array([np.pi])

def simulate_notorious(p: float, mu: float = 60.0, N: int = 10, noise_std: float = 0.0, seed: int = 42):
    """
    Simulate noisy measurements for the notorious test system.

    Parameters:
    - p: true parameter value
    - mu: stiffness parameter (default: 60)
    - N: number of time intervals (N+1 points)
    - noise_std: std. deviation of Gaussian noise
    - seed: random seed

    Returns:
    - t_grid: time points
    - y_meas: noisy measurements
    - y_true: true trajectory (no noise)
    """
    integrator = make_notorious_integrator(mu)
    x0 = get_initial_state()
    t_grid = np.linspace(0, 1, N + 1)

    y_true = [x0]
    for k in range(N):
        res = integrator(x0=x0, p=ca.vertcat(t_grid[k], p))
        x0 = res['xf'].full().flatten()
        y_true.append(x0)

    y_true = np.array(y_true)

    rng = np.random.default_rng(seed)
    y_meas = y_true + noise_std * rng.standard_normal(y_true.shape)

    return t_grid, y_meas, y_true
