import casadi as ca
from typing import List, Tuple

def setup_multiple_shooting(integrator: ca.Function, N: int, nx: int, np_p: int) -> Tuple[ca.MX, List[ca.MX], ca.MX, List[ca.MX], ca.MX]:
    """
    Set up a direct multiple shooting discretization.

    Parameters:
    ----------
    integrator : casadi.Function
        The integrator created by create_integrator or specific integrator for the system.
    N : int
        Number of shooting intervals.
    nx : int
        State dimension.
    np_p : int
        Parameter dimension.

    Returns:
    -------
    w       : ca.MX
        The stacked decision variable [s_0; ...; s_{N-1}; p].
    X_end   : List[ca.MX]
        List of length N: the state at the end of each interval.
    g       : ca.MX
        Continuity constraints of size (N-1)*nx (zero if N=1).
    S_vars  : List[ca.MX]
        The symbolic shooting initial states s_i.
    P_var   : ca.MX
        The symbolic parameter vector p.
    """
    # Shooting initial states s_0 ... s_{N-1}
    S_vars = [ca.MX.sym(f's_{i}', nx) for i in range(N)]
    # Parameter vector
    P_var = ca.MX.sym('p', np_p)

    X_end = []
    g_constr = []

    # Integrate each subinterval and collect end states
    for i in range(N):
        res = integrator(x0=S_vars[i], p=P_var)
        x_end = res['xf']
        X_end.append(x_end)
        # Continuity constraint: x_end = s_{i+1}
        if i < N - 1:
            g_constr.append(x_end - S_vars[i + 1])

    # Stack continuity constraints g = [x_end[0] - s_1; x_end[1] - s_2; ...]
    g = ca.vertcat(*g_constr) if g_constr else ca.MX.zeros(0)

    # Stack all decision variables [s_0; ...; s_{N-1}; p]
    w = ca.vertcat(*(S_vars + [P_var]))

    return w, X_end, g, S_vars, P_var
