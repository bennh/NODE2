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
        Equality constraint vector.
    w0 : numpy.ndarray
        Initial guess for w.
    opts : dict, optional
        Options for IPOPT solver.

    Returns:
    -------
    result : dict
        - 'x': numpy.ndarray, the optimal w.
        - 'f': float, the objective value 0.5*||F1||^2 at solution.
        - 'g': numpy.ndarray, the equality constraint residual at solution.
        - 'ipopt': the raw IPOPT solution dictionary.
    """
    # Create CASADI NLP problem and solve with IPOPT
    nlp = {'x': w, 'f': 0.5 * ca.sumsqr(F1), 'g': g}
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
      - Solve QP: min 0.5*dw^T H dw + c^T dw, s.t. J2 * dw + g = 0
      where H = J1^T J1, c = J1^T r, r = F1(w_k), J1 = dF1/dw, J2 = dg/dw.

    Parameters:
    ----------
    w : casadi.MX
        Decision variable vector.
    F1 : casadi.MX
        Residual vector (to be squared).
    g : casadi.MX
        Equality constraint vector.
    w0 : numpy.ndarray
        Initial guess.
    max_iter : int, optional
        Maximum number of Gauss-Newton iterations.
    tol : float, optional
        Convergence tolerance on parameter update norm.
    qp_opts : dict, optional
        Options for QP solver.

    Returns:
    -------
    result : dict
        - 'x': numpy.ndarray, the final iterate.
        - 'iterations': int, number of iterations performed.
        - 'converged': bool, whether convergence was achieved.
    """
    # Create CASADI functions for F1, g, and their Jacobians
    F1_fun = ca.Function('F1_fun', [w], [F1])
    g_fun = ca.Function('g_fun', [w], [g])
    J1_fun = ca.Function('J1_fun', [w], [ca.jacobian(F1, w)])
    J2_fun = ca.Function('J2_fun', [w], [ca.jacobian(g, w)])

    xk = w0.copy()
    n = w.size1()
    converged = False

    for k in range(max_iter):
        # Evaluate residuals and Jacobians at current iterate
        r = F1_fun(xk).full().flatten()
        h = g_fun(xk).full().flatten()
        J1 = J1_fun(xk).full()
        J2 = J2_fun(xk).full()

        # Build Gauss-Newton Hessian and gradient
        H = J1.T @ J1
        c = J1.T @ r

        # Setup QP: min 0.5*dw^T H dw + c^T dw, subject to J2*dw + h = 0
        dw = ca.MX.sym('dw', n)
        qp = {
            'x': dw,
            'f': 0.5 * ca.mtimes(dw.T, ca.mtimes(H, dw)) + ca.dot(c, dw),
            'g': ca.mtimes(J2, dw) + h
        }
        qp_opts = qp_opts if qp_opts is not None else {}
        qp_solver = ca.qpsol('qp_solver', 'qpoases', qp, qp_opts)
        sol_qp = qp_solver(lbg=np.zeros(h.size), ubg=np.zeros(h.size))
        dw_opt = sol_qp['x'].full().flatten()

        # Update iterate
        xk = xk + dw_opt
        if np.linalg.norm(dw_opt) < tol:
            converged = True
            break

    return {
        'x': xk,
        'iterations': k+1,
        'converged': converged
    }
