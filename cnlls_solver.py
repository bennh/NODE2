import casadi as ca
import numpy as np

def solve_cnlls_ipopt(w: ca.MX, F1: ca.MX, g: ca.MX, F2: ca.MX, F3: ca.MX, w0: np.ndarray, opts: dict = None) -> dict:
    """
    Solve CNLLS problem explicitly adding F2 and F3 to constraints g(w).

    Parameters:
    ----------
    w : ca.MX
        Decision variables.
    F1 : ca.MX
        Residual to minimize.
    g : ca.MX
        Original continuity constraints.
    F2 : ca.MX
        Explicit additional equality constraints.
    F3 : ca.MX
        Explicit additional inequality constraints (F3 <= 0).
    w0 : numpy.ndarray
        Initial guess.
    opts : dict, optional
        Solver options.

    Returns:
    -------
    dict
        Optimization result.
    """
    # Explicitly add F2 and F3 constraints
    g_total = ca.vertcat(g, F2, F3)

    nlp = {
        'x': w,
        'f': 0.5 * ca.sumsqr(F1),
        'g': g_total
    }

    # Set bounds explicitly (equality: lb = ub = 0, inequality F3 <= 0: ub = 0, lb = -inf)
    lbg = np.zeros(g.size1() + F2.size1() + F3.size1())
    ubg = np.zeros(g.size1() + F2.size1() + F3.size1())

    # For inequality constraints (F3 <= 0), set lower bounds to -inf
    if F3.size1() > 0:
        lbg[-F3.size1():] = -np.inf

    solver_opts = opts if opts else {}
    solver = ca.nlpsol('solver', 'ipopt', nlp, solver_opts)

    sol = solver(x0=w0, lbg=lbg, ubg=ubg)

    return {
        'x': sol['x'].full().flatten(),
        'f': float(sol['f']),
        'g': sol['g'].full().flatten(),
        'ipopt': sol
    }

def solve_cnlls_gauss_newton(w: ca.MX,
                             F1: ca.MX,
                             g: ca.MX,
                             F2: ca.MX,
                             F3: ca.MX,
                             w0: np.ndarray,
                             max_iter: int = 20,
                             tol: float = 1e-6,
                             qp_opts: dict = None) -> dict:
    """
    Solve a constrained nonlinear least-squares problem explicitly including F2=0 and F3<=0 constraints.

    Parameters:
    ----------
    w : ca.MX
        Decision variable vector.
    F1 : ca.MX
        Residual vector.
    g : ca.MX
        Original continuity constraints.
    F2 : ca.MX
        Additional equality constraints.
    F3 : ca.MX
        Additional inequality constraints (F3 <= 0).
    w0 : numpy.ndarray
        Initial guess.
    max_iter : int, optional
        Maximum GN iterations.
    tol : float, optional
        Convergence tolerance.
    qp_opts : dict, optional
        Options for QP solver.

    Returns:
    -------
    result : dict
        Optimization result.
    """
    F1_fun = ca.Function('F1_fun', [w], [F1])
    g_fun = ca.Function('g_fun', [w], [g])
    F2_fun = ca.Function('F2_fun', [w], [F2])
    F3_fun = ca.Function('F3_fun', [w], [F3])

    J1_fun = ca.Function('J1_fun', [w], [ca.jacobian(F1, w)])
    Jg_fun = ca.Function('Jg_fun', [w], [ca.jacobian(g, w)])
    J2_fun = ca.Function('J2_fun', [w], [ca.jacobian(F2, w)])
    J3_fun = ca.Function('J3_fun', [w], [ca.jacobian(F3, w)])

    xk = w0.copy()
    n = w.size1()
    converged = False

    for k in range(max_iter):
        # Evaluate residuals and Jacobians at current iterate
        r = F1_fun(xk).full().flatten()
        h_g = g_fun(xk).full().flatten()
        h_F2 = F2_fun(xk).full().flatten()
        h_F3 = F3_fun(xk).full().flatten()

        J1 = J1_fun(xk).full()
        Jg = Jg_fun(xk).full()
        J2 = J2_fun(xk).full()
        J3 = J3_fun(xk).full()

        # Build Gauss-Newton Hessian and gradient
        H = J1.T @ J1
        c = J1.T @ r

        # Concatenate all constraints
        J_constr = np.vstack([Jg, J2, J3])
        h_constr = np.concatenate([h_g, h_F2, h_F3])

        # Bounds setup: equality constraints (g,F2) lb=ub=0, inequality (F3) lb=-inf, ub=0
        neq = h_g.size + h_F2.size
        nineq = h_F3.size

        lbg = np.concatenate([np.zeros(neq), -np.inf*np.ones(nineq)])
        ubg = np.zeros(neq + nineq)

        # Setup QP explicitly with all constraints
        dw = ca.MX.sym('dw', n)
        qp = {
            'x': dw,
            'f': 0.5 * ca.mtimes(dw.T, ca.mtimes(H, dw)) + ca.dot(c, dw),
            'g': ca.mtimes(J_constr, dw) + h_constr
        }

        qp_opts = qp_opts if qp_opts else {}
        qp_solver = ca.qpsol('qp_solver', 'qpoases', qp, qp_opts)
        sol_qp = qp_solver(lbg=lbg, ubg=ubg)

        dw_opt = sol_qp['x'].full().flatten()

        res_norm = np.linalg.norm(r)  
        alpha = 1.0
        while True:
            xk_new = xk + alpha * dw_opt
            r_new = F1_fun(xk_new).full().flatten()  
            if np.linalg.norm(r_new) <= res_norm or alpha < 1e-3:
                xk = xk_new
                break
            alpha *= 0.5

        if np.linalg.norm(alpha * dw_opt) < tol:
            converged = True
            break

    return {
        'x': xk,
        'iterations': k+1,
        'converged': converged
    }
