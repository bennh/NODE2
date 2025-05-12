import casadi as ca
import numpy as np


def solve_cnlls_ipopt(w: ca.MX,
                      F1: ca.MX,
                      g: ca.MX,
                      w0: np.ndarray,
                      opts: dict = None) -> dict:
    """
    Solve a constrained nonlinear least-squares problem via IPOPT.

    Minimize 0.5*||F1(w)||^2 subject to g(w) = 0.

    Parameters:
    ----------
    w : casadi.MX
        Decision variable vector.
    F1 : casadi.MX
        Residual vector (to be squared).
    g : casadi.MX
        Constraint vector (equality constraints g(w)=0).
    w0 : numpy.ndarray
        Initial guess for w (length = w.size1()).
    opts : dict, optional
        IPOPT solver options.

    Returns:
    -------
    sol : dict
        Dictionary with keys:
        - 'x': numpy.ndarray, the optimal w.
        - 'f': final objective value.
        - 'g': final constraint violation.
        - 'ipopt': the raw IPOPT result object.
    """
    nlp = {
        'x': w,
        'f': 0.5 * ca.sumsqr(F1),
        'g': g
    }
    solver_opts = opts if opts is not None else {}
    solver = ca.nlpsol('solver', 'ipopt', nlp, solver_opts)
    sol = solver(x0=w0, lbg=np.zeros(g.size1()), ubg=np.zeros(g.size1()))
    w_opt = sol['x'].full().flatten()
    return {
        'x': w_opt,
        'f': float(sol['f']),
        'g': sol['g'].full().flatten(),
        'ipopt': sol
    }


def solve_cnlls_gauss_newton(w: ca.MX,
                              F1: ca.MX,
                              g: ca.MX,
                              w0: np.ndarray,
                              max_iter: int = 20,
                              tol: float = 1e-6,
                              qp_opts: dict = None) -> dict:
    """
    Solve a constrained nonlinear least-squares problem via Generalized Gauss-Newton.

    At each iteration k:
      - Linearize residual F1 and constraints g at w_k
      - Solve QP: min 0.5*dw^T H dw + J1^T r * dw, s.t. J2 * dw + g = 0
      - Update w_{k+1} = w_k + dw
    Terminates when ||dw|| < tol or max_iter reached.

    Parameters:
    ----------
    w : casadi.MX
        Decision variable vector.
    F1 : casadi.MX
        Residual vector to be squared.
    g : casadi.MX
        Equality constraint vector.
    w0 : numpy.ndarray
        Initial guess.
    max_iter : int
        Maximum number of GN iterations.
    tol : float
        Convergence tolerance on dw.
    qp_opts : dict, optional
        Options for the QP solver.

    Returns:
    -------
    result : dict
        - 'x': numpy.ndarray, the final iterate.
        - 'iterations': int, number of iterations used.
        - 'converged': bool, whether tol was reached.
    """
    # Create evaluation functions
    F1_fun = ca.Function('F1_fun', [w], [F1])
    g_fun  = ca.Function('g_fun',  [w], [g])
    J1_fun = ca.Function('J1_fun', [w], [ca.jacobian(F1, w)])
    J2_fun = ca.Function('J2_fun', [w], [ca.jacobian(g,  w)])

    xk = w0.copy()
    n = w.size1()
    converged = False

    for k in range(max_iter):
        # Evaluate residuals and Jacobians
        r = F1_fun(xk).full().flatten()         # m1
        h = g_fun(xk).full().flatten()          # m2
        J1 = J1_fun(xk).full()                  # m1 x n
        J2 = J2_fun(xk).full()                  # m2 x n
        
        # Build QP matrices
        H = J1.T @ J1                          # n x n
        c = J1.T @ r                           # n
        # Define QP variable dw
        dw = ca.MX.sym('dw', n)
        qp = {
            'x': dw,
            'f': 0.5 * ca.mtimes([dw.T, H, dw]) + c.T @ dw,
            'g': J2 @ dw + h
        }
        qp_solver = ca.qpsol('qp_solver', 'qpoases', qp, qp_opts or {})
        sol_qp = qp_solver(lbg=np.zeros(h.size), ubg=np.zeros(h.size))
        dw_opt = sol_qp['x'].full().flatten()
        
        # Update
        xk = xk + dw_opt
        if np.linalg.norm(dw_opt) < tol:
            converged = True
            break

    return {
        'x': xk,
        'iterations': k+1,
        'converged': converged
    }
