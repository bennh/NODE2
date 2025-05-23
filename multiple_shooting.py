import casadi as ca
from typing import Callable, List, Tuple

def setup_multiple_shooting(
        integrator: ca.Function,
        t_shooting,
        nx: int,
        np_p: int,
        enforce_state_nonneg: bool = True,
        enforce_param_nonneg: bool = False
) -> Tuple[ca.MX, List[ca.MX], ca.MX, ca.MX, List[ca.MX], ca.MX]:
    """
    Set up a direct multiple shooting discretization.

    Returns:
    --------
    w     : MX          # Decision variable vector [s_0,...,s_{N-1}, p]
    X_end: List[MX]     # End states for each interval
    F2    : MX          # Equality constraints (continuity)
    F3    : MX          # Inequality constraints (e.g., s_i ≥ 0, p ≥ 0)
    S_vars: List[MX]    # Symbolic shooting state variables
    P_var : MX          # Symbolic parameter vector
    """
    N = len(t_shooting) - 1
    S_vars = [ca.MX.sym(f's_{i}', nx) for i in range(N)]
    P_var = ca.MX.sym('p', np_p)

    X_end, F2_terms = [], []
    for i in range(N):
        res = integrator(x0=S_vars[i], p=P_var)
        x_end = res['xf']
        X_end.append(x_end)
        if i < N - 1:
            F2_terms.append(x_end - S_vars[i + 1])

    F2 = ca.vertcat(*F2_terms) if F2_terms else ca.MX.zeros(0)

    # Decision vector
    w = ca.vertcat(*(S_vars + [P_var]))

    # Optional inequality constraints (F3)
    F3_list = []
    if enforce_state_nonneg:
        F3_list += S_vars
    if enforce_param_nonneg:
        F3_list.append(P_var)

    F3 = ca.vertcat(*F3_list) if F3_list else ca.MX.zeros(0)

    return w, X_end, F2, F3, S_vars, P_var