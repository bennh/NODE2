import casadi as ca
import numpy as np

def make_notorious_integrator(dt: float, mu: float) -> ca.Function:
    """
    Construct CasADi integrator for the Notorious test example (Section 6 of Bock et al. 2007).
    dx1/dt = x2
    dx2/dt = mu^2 * x1 - (mu^2 + p^2) * sin(p * t)
    Initial: x1(0) = 0, x2(0) = pi
    Parameter: p (to be estimated)
    mu: fixed known (e.g., mu = 60)
    """
    x = ca.MX.sym('x', 2)
    p = ca.MX.sym('p')
    t = ca.MX.sym('t')

    dxdt = ca.vertcat(
        x[1],
        mu**2 * x[0] - (mu**2 + p**2) * ca.sin(p * t)
    )

    dae = {'x': x, 'p': ca.vertcat(p, t), 'ode': dxdt}
    return ca.integrator('notorious_int', 'cvodes', dae, {'tf': dt})

def get_initial_state() -> np.ndarray:
    return np.array([0.0, np.pi])

def get_true_parameters() -> float:
    return np.pi

def simulate_notorious_data(p: float, mu: float, t_grid: np.ndarray) -> np.ndarray:
    x1 = np.sin(p * t_grid)
    x2 = p * np.cos(p * t_grid)
    return np.stack([x1, x2], axis=1)

def simulate_notorious_measurements(mu: float, p: float, t_grid: np.ndarray, noise_level=0.05, seed=42) -> np.ndarray:
    np.random.seed(seed)
    y_clean = simulate_notorious_data(p, mu, t_grid)
    noise = np.random.randn(*y_clean.shape) * noise_level
    y_noisy = y_clean + noise
    return y_noisy

def make_multiple_shooting_integrators(t_grid: np.ndarray, mu: float) -> list:
    dts = np.diff(t_grid)
    return [make_notorious_integrator(dt, mu) for dt in dts]

def setup_notorious_shooting(t_grid: np.ndarray,
                              nx: int,
                              np_p: int,
                              mu: float,
                              enforce_param_nonneg: bool = True):
    """
    Setup for multiple shooting on the Notorious test example.

    Parameters:
    ----------
    t_grid : np.ndarray
        Time discretization grid
    nx : int
        State dimension (typically 2)
    np_p : int
        Parameter dimension (typically 1)
    mu : float
        Known system constant
    enforce_param_nonneg : bool
        If True, adds F3: p ≥ 0

    Returns:
    -------
    w : MX
        Optimization variables [s_0, ..., s_{N-1}, p]
    X_end : List[MX]
        End states of each interval
    F2 : MX
        Matching conditions (equality constraints)
    F3 : MX
        Optional inequality constraints (e.g., p ≥ 0)
    S_vars : List[MX]
        Symbolic shooting initial state variables
    P_var : MX
        Symbolic parameter variable
    """
    N = len(t_grid) - 1
    dt_list = np.diff(t_grid)

    S_vars = [ca.MX.sym(f's_{i}', nx) for i in range(N)]
    P_var = ca.MX.sym('p', np_p)
    w = ca.vertcat(*S_vars, P_var)

    X_end = []
    F2_terms = []

    for j in range(N):
        integrator = make_notorious_integrator(dt_list[j], mu)
        t_j = t_grid[j]
        res = integrator(x0=S_vars[j], p=ca.vertcat(P_var, t_j))
        x_end = res['xf']
        X_end.append(x_end)
        if j + 1 < N:
            F2_terms.append(x_end - S_vars[j + 1])

    F2 = ca.vertcat(*F2_terms) if F2_terms else ca.MX.zeros(0)
    F3 = P_var if enforce_param_nonneg else ca.MX.zeros(0)

    return w, X_end, F2, F3, S_vars, P_var

def notorious_simulate_with_integrator(p_est: float, mu: float, t_grid: np.ndarray) -> np.ndarray:
    dts = np.diff(t_grid)
    integrators = [make_notorious_integrator(dt, mu) for dt in dts]
    x = np.array([0.0, np.pi])
    result = [x]
    for j, dt in enumerate(dts):
        t_j = t_grid[j]
        x = integrators[j](x0=ca.DM(x), p=ca.DM([p_est, t_j]))['xf'].full().flatten()
        result.append(x)
    return np.array(result)
