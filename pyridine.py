import casadi as ca
import numpy as np

def make_pyridine_integrator(dt: float) -> ca.Function:
    """Create a CasADi RK integrator for the 7-variable Pyridine system."""
    y = ca.MX.sym('y', 7)
    p = ca.MX.sym('p', 11)
    # State variables
    A, B, C, D, E, F_, G = y[0], y[1], y[2], y[3], y[4], y[5], y[6]
    # Parameters
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = ca.vertsplit(p)

    # Differential equations
    dy = ca.MX(7, 1)
    dy[0] = -p1 * A + p9 * B
    dy[1] = p1 * A - p2 * B - p3 * B * C + p7 * D - p9 * B + p10 * D * F_
    dy[2] = p2 * B - p3 * B * C - 2 * p4 * C * C - p6 * C + p8 * E + p10 * D * F_ + 2 * p11 * E * F_
    dy[3] = p3 * B * C - p5 * D - p7 * D - p10 * D * F_
    dy[4] = p4 * C * C + p5 * D - p8 * E - p11 * E * F_
    dy[5] = p3 * B * C + p4 * C * C + p6 * C - p10 * D * F_ - p11 * E * F_
    dy[6] = p6 * C + p7 * D + p8 * E

    dae = {'x': y, 'p': p, 'ode': dy}
    return ca.integrator('F', 'rk', dae, {'tf': dt})

def get_initial_state() -> np.ndarray:
    """Return the initial state for the Pyridine problem (7 states)."""
    x0 = np.zeros(7)
    # Only Pyridine has an initial concentration of 1.0
    x0[0] = 1.0
    return x0

def get_true_parameters() -> np.ndarray:
    """Return the true parameters for the Pyridine problem (11 parameters)."""
    return np.array([
        1.81, 0.894, 29.4, 9.21, 0.058,
        2.43, 0.0644, 5.55, 0.0201, 0.577, 2.15
    ])

def simulate_pyridine(integrator: ca.Function, x0: np.ndarray, p: np.ndarray, N: int) -> np.ndarray:
    """Simulate the Pyridine system over N time steps using the given integrator."""
    result = [x0]
    x = x0
    for _ in range(N):
        x = integrator(x0=x, p=p)['xf'].full().flatten()
        result.append(x)
    return np.array(result)
