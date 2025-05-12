import casadi as ca

def create_integrator(nx: int, np_p: int, ode_rhs, dt: float):
    """
    Create a CasADi CVODES integrator for an ODE system.

    Parameters:
    ----------
    nx : int
        Dimension of the state vector x.
    np_p : int
        Dimension of the parameter vector p.
    ode_rhs : Callable[[MX, MX], MX]
        A function f(x, p) returning an MX column of length nx.
    dt : float
        Time length of each shooting subinterval.

    Returns:
    -------
    integrator : casadi.Function
        An integrator you call as `res = integrator(x0=..., p=...)` returning `res['xf']`.
    """
    x = ca.MX.sym('x', nx)
    p = ca.MX.sym('p', np_p)
    dx = ode_rhs(x, p)
    dae = {'x': x, 'p': p, 'ode': dx}
    opts = {'tf': dt}
    return ca.integrator('integrator', 'cvodes', dae, opts)


def pyridine_ode(x: ca.MX, p: ca.MX) -> ca.MX:
    """
    Example Pyridine reaction ODE.

    State variables:
      x[0] = [Pyridine] (C_py)
      x[1] = [Ammonia]  (C_nh3)
      x[2] = [Pentane]  (C_pent)

    Parameters:
      p[0] = k1
      p[1] = k2

    Returns:
    -------
    dx : MX(3×1)
        Time derivatives of [C_py, C_nh3, C_pent]
    """
    C_py, C_nh3, C_pent = x[0], x[1], x[2]
    k1, k2 = p[0], p[1]
    v1 = k1 * C_py        # Pyridine → Ammonia
    v2 = k2 * C_nh3       # Ammonia → Pentane
    dC_py   = -v1
    dC_nh3  =  v1 - v2
    dC_pent =        v2
    return ca.vertcat(dC_py, dC_nh3, dC_pent)
