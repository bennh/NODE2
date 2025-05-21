import casadi as ca
import numpy as np

def solve_cnlls_gauss_newton(w: ca.MX,
                              F1: ca.MX,
                              g: ca.MX,
                              w0: np.ndarray,
                              max_iter: int = 30,
                              tol: float = 1e-8,
                              qp_opts: dict = None) -> dict:
    """
    Simplest Gauss-Newton method for constrained nonlinear least squares:
        min_w 0.5 * ||F1(w)||^2   s.t. g(w) = 0

    Parameters
    ----------
    w : casadi.MX
        Decision variable.
    F1 : casadi.MX
        Residual vector.
    g : casadi.MX
        Equality constraint vector.
    w0 : np.ndarray
        Initial guess.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on ||dw||.
    qp_opts : dict
        Options for qpsol.

    Returns
    -------
    dict
        - 'x': final iterate
        - 'iterations': number of iterations
        - 'converged': whether ||dw|| < tol
    """
    n = w.size1()
    F1_fun = ca.Function("F1_fun", [w], [F1])
    g_fun  = ca.Function("g_fun", [w], [g])
    J1_fun = ca.Function("J1_fun", [w], [ca.jacobian(F1, w)])
    J2_fun = ca.Function("J2_fun", [w], [ca.jacobian(g, w)])

    xk = w0.copy()
    converged = False

    for k in range(max_iter):
        r  = F1_fun(xk).full().flatten()
        h  = g_fun(xk).full().flatten()
        J1 = J1_fun(xk).full()
        J2 = J2_fun(xk).full()

        H = J1.T @ J1
        c = J1.T @ r

        dw = ca.MX.sym("dw", n)
        qp = {
            "x": dw,
            "f": 0.5 * ca.mtimes([dw.T, H, dw]) + ca.dot(c.T, dw),
            "g": J2 @ dw + h
        }

        solver = ca.qpsol("solver", "qpoases", qp, qp_opts or {})
        sol = solver(lbg=np.zeros_like(h), ubg=np.zeros_like(h))
        dw_opt = sol["x"].full().flatten()

        xk += dw_opt
        if np.linalg.norm(dw_opt) < tol:
            converged = True
            break

    return {
        "x": xk,
        "iterations": k + 1,
        "converged": converged
    }
