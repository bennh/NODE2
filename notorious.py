import casadi as ca
import numpy as np

def make_notorious_integrator(dt: float, mu: float) -> ca.Function:
    """
    Construct CasADi integrator for the Notorious test example (Section 6 of Bock et al. 2007).
    dx1/dt = x2
    dx2/dt = mu^2 * x1 - (mu^2 + p^2) * sin(p * t)
    Initial: x1(0) = 0, x2(0) = pi
    Parameter: p (to be estimated)
    mu: fixed known (default: 60)
    """
    x = ca.MX.sym('x', 2)      # [x1, x2]
    p = ca.MX.sym('p')         # parameter to estimate (scalar)
    t = ca.MX.sym('t')         # current time

    dxdt = ca.vertcat(
        x[1],
        mu**2 * x[0] - (mu**2 + p**2) * ca.sin(p * t)
    )
    dae = {'x': x, 'p': ca.vertcat(p, t), 'ode': dxdt}
    return ca.integrator('notorious_int', 'cvodes', dae, {'tf': dt})

def get_initial_state() -> np.ndarray:
    """
    Returns the initial state for the Notorious test example.
    """
    return np.array([0.0, np.pi])  # x1(0)=0, x2(0)=pi

def get_true_parameters() -> float:
    """
    Returns the true parameter p for the Notorious test example.
    """
    return np.pi

def simulate_notorious_data(p: float, mu: float, t_grid: np.ndarray) -> np.ndarray:
    """
    Simulate the exact solution for the Notorious test example.
    """
    # Analytical solution from the paper
    x1 = np.sin(p * t_grid)
    x2 = p * np.cos(p * t_grid)
    return np.stack([x1, x2], axis=1)

def simulate_notorious_measurements(mu: float, p: float, t_grid: np.ndarray, noise_level=0.05, seed=42) -> np.ndarray:
    """
    Simulate noisy measurement data for the Notorious test example.
    Measurements at points t_grid (excluding t=0).
    """
    np.random.seed(seed)
    y_clean = simulate_notorious_data(p, mu, t_grid)
    noise = np.random.randn(*y_clean.shape) * noise_level
    y_noisy = y_clean + noise
    return y_noisy

def make_multiple_shooting_integrators(t_grid: np.ndarray, mu: float) -> list:
    """
    Returns a list of integrators for each multiple shooting interval.
    Each integrator advances the state from t_k to t_{k+1}.
    """
    dts = np.diff(t_grid)
    integrators = [make_notorious_integrator(dt, mu) for dt in dts]
    return integrators