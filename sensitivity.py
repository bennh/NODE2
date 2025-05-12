import casadi as ca
from typing import List, Tuple

def compute_sensitivities(
    integrator: ca.Function,
    S_vars: List[ca.MX],
    P_var: ca.MX,
    N: int,
    nx: int,
    np_p: int
) -> Tuple[ca.MX, ca.MX]:
    """
    Compute sensitivities (Jacobian) for multiple shooting intervals.

    Parameters:
    ----------
    integrator : casadi.Function
        CasADi integrator for the ODE system.
    S_vars : List[MX]
        The symbolic shooting initial states s_i.
    P_var : MX
        The symbolic parameter vector p.
    N : int
        Number of shooting intervals.
    nx : int
        State dimension.
    np_p : int
        Parameter dimension.

    Returns:
    -------
    J_s : MX
        Jacobian of the terminal states w.r.t. the shooting variables.
    J_p : MX
        Jacobian of the terminal states w.r.t. the parameters.
    """
    # Prepare containers for the Jacobians
    J_s = []
    J_p = []

    # Iterate through each interval
    for i in range(N):
        # Get the end state expression
        res = integrator(x0=S_vars[i], p=P_var)
        x_end = res['xf']
        
        # Compute Jacobians
        J_s_i = ca.jacobian(x_end, S_vars[i])
        J_p_i = ca.jacobian(x_end, P_var)

        J_s.append(J_s_i)
        J_p.append(J_p_i)

    # Stack all Jacobians
    J_s = ca.block_diag(*J_s)
    J_p = ca.vertcat(*J_p)

    return J_s, J_p


def compute_full_jacobian(
    integrator: ca.Function,
    S_vars: List[ca.MX],
    P_var: ca.MX,
    N: int,
    nx: int,
    np_p: int
) -> ca.MX:
    """
    Compute the full Jacobian of the multiple shooting problem.

    Returns:
    -------
    J_full : MX
        Full Jacobian matrix [J_s, J_p].
    """
    J_s, J_p = compute_sensitivities(integrator, S_vars, P_var, N, nx, np_p)
    return ca.horzcat(J_s, J_p)
