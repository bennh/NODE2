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


def make_lv_integrator(dt):
    x = ca.MX.sym('x', 2)
    p = ca.MX.sym('p', 4)  # [alpha, beta, gamma, delta]

    alpha, beta, gamma, delta = p[0], p[1], p[2], p[3]
    f = ca.vertcat(
        alpha * x[0] - beta * x[0] * x[1],
        delta * x[0] * x[1] - gamma * x[1]
    )

    dae = {'x': x, 'p': p, 'ode': f}
    return ca.integrator('F', 'cvodes', dae, {'tf': dt})


def make_pyridine_integrator(dt):
    y = ca.MX.sym("y", 7)
    p = ca.MX.sym("p", 11)
    A, B, C, D, E, F, G = y[0], y[1], y[2], y[3], y[4], y[5], y[6]
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = ca.vertsplit(p)

    dy = ca.MX(7, 1)
    dy[0] = -p1 * A + p9 * B
    dy[1] = p1 * A - p2 * B - p3 * B * C + p7 * D - p9 * B + p10 * D * F
    dy[2] = p2 * B - p3 * B * C - 2 * p4 * C * C - p6 * C + p8 * E + p10 * D * F + 2 * p11 * E * F
    dy[3] = p3 * B * C - p5 * D - p7 * D - p10 * D * F
    dy[4] = p4 * C * C + p5 * D - p8 * E - p11 * E * F
    dy[5] = p3 * B * C + p4 * C * C + p6 * C - p10 * D * F - p11 * E * F
    dy[6] = p6 * C + p7 * D + p8 * E

    dae = {'x': y, 'p': p, 'ode': dy}
    return ca.integrator("F", "rk", dae, {'tf': dt})
