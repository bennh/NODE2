import casadi as ca
import numpy as np


def solve_cnlls_ipopt(w: ca.MX,
                      F1: ca.MX,
                      g: ca.MX,
                      w0: np.ndarray,
                      n_params: int = 11,
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

    lbw = -np.inf * np.ones(w.size1())
    # lbw[-n_params:] = 1e-8
    ubw = np.inf * np.ones(w.size1())

    solver_opts = opts if opts is not None else {}
    solver = ca.nlpsol('solver', 'ipopt', nlp, solver_opts)
    sol = solver(x0=w0, lbg=np.zeros(g.size1()), ubg=np.zeros(g.size1()), lbx=lbw, ubx=ubw)
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
                             n_params: int = 11,
                             max_iter: int = 30,
                             tol: float = 1e-8,
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
    g_fun = ca.Function('g_fun', [w], [g])
    J1_fun = ca.Function('J1_fun', [w], [ca.jacobian(F1, w)])
    J2_fun = ca.Function('J2_fun', [w], [ca.jacobian(g, w)])

    xk = w0.copy()
    n = w.size1()
    converged = False

    for k in range(max_iter):
        # Evaluate residuals and Jacobians
        r = F1_fun(xk).full().flatten()  # m1
        h = g_fun(xk).full().flatten()  # m2
        J1 = J1_fun(xk).full()  # m1 x n
        J2 = J2_fun(xk).full()  # m2 x n

        lbx = np.concatenate([
            -np.inf * np.ones(n - 11),
            np.zeros(11)
        ])
        ubx = np.full(n, np.inf)

        # Build QP matrices
        H = J1.T @ J1  # n x n

        # LM 阻尼
        mu = 1e-4  # 或问题特定适合的数值
        H = J1.T @ J1 + mu * np.eye(n)
        c = J1.T @ r  # n
        # Define QP variable dw
        dw = ca.MX.sym('dw', n)
        qp = {
            'x': dw,
            'f': 0.5 * ca.mtimes([dw.T, H, dw]) + ca.dot(c.T, dw),
            'g': J2 @ dw + h
        }
        qp_solver = ca.qpsol('qp_solver', 'qpoases', qp, qp_opts or {})
        sol_qp = qp_solver(lbg=np.zeros(h.size), ubg=np.zeros(h.size), lbx=lbx, ubx=ubx)
        dw_opt = sol_qp['x'].full().flatten()

        # 回退线搜索
        alpha = 1.0
        while alpha > 1e-6:
            trial_x = xk + alpha * dw_opt
            if np.linalg.norm(F1_fun(trial_x).full()) ** 2 < np.linalg.norm(F1_fun(xk).full()) ** 2:
                break
            alpha *= 0.5
        xk = xk + alpha * dw_opt

        # Update
        # xk = xk + dw_opt
        if np.linalg.norm(dw_opt) < tol:
            converged = True
            break

    return {
        'x': xk,
        'iterations': k + 1,
        'converged': converged
    }


def solve_cnlls_gauss_newton_logparam(
        w: ca.MX,
        F1: ca.MX,
        g: ca.MX,
        w0: np.ndarray,
        n_params: int = 11,
        max_iter: int = 30,
        tol: float = 1e-8,
        qp_opts: dict = None
) -> dict:
    """
    Gauss-Newton with log-parameter reparametrization.
    The last `n_params` entries of w are parameters p = exp(q).
    """

    # 维度
    n_total = w.size1()
    n_var = n_total - n_params  # 非参数分量数

    # 符号变量：s for states, q for log-parameters
    s = ca.MX.sym('s', n_var)
    q = ca.MX.sym('q', n_params)
    # 重构原始 w = [s; exp(q)]
    w_reparam = ca.vertcat(s, ca.exp(q))

    # 在 F1, g 中替换 w -> [s; exp(q)]
    F1_r = ca.substitute(F1, w, w_reparam)
    g_r = ca.substitute(g, w, w_reparam)

    # 构造对应的 CasADi 函数
    v = ca.vertcat(s, q)
    F1_fun = ca.Function('F1_fun', [v], [F1_r])
    g_fun = ca.Function('g_fun', [v], [g_r])
    J1_fun = ca.Function('J1_fun', [v], [ca.jacobian(F1_r, v)])
    J2_fun = ca.Function('J2_fun', [v], [ca.jacobian(g_r, v)])

    # 初始值：拆分 w0 → s0, p0 → q0 = log(p0)
    s0 = w0[:n_var]
    p0 = w0[n_var:]
    w0 = np.concatenate([s0, np.log(p0)])

    converged = False
    for k in range(max_iter):
        # 评估残差与雅可比
        r = F1_fun(w0).full().flatten()  # m1
        h = g_fun(w0).full().flatten()  # m2
        J1 = J1_fun(w0).full()  # m1 x n
        J2 = J2_fun(w0).full()  # m2 x n

        # 构造 QP
        # H = J1.T @ J1

        # LM 阻尼
        mu = 0.0001
        H = J1.T @ J1 + mu * np.eye(n_total)
        c = J1.T @ r

        dv = ca.MX.sym('dv', n_total)

        lam_reg = 1e-5
        reg = lam_reg / 2 * ca.sumsqr((w0[n_var:] + dv[n_var:]) - np.log(p0))

        qp = {
            'x': dv,
            'f': 0.5 * ca.mtimes([dv.T, H, dv]) + ca.dot(c.T, dv) + reg,
            'g': J2 @ dv + h
        }
        qp_solver = ca.qpsol('qp_solver', 'qpoases', qp, qp_opts or {})

        lbx = -np.inf * np.ones(n_total)
        ubx = np.inf * np.ones(n_total)

        sol = qp_solver(
            x0=np.zeros(n_total),
            lbg=np.zeros(h.size), ubg=np.zeros(h.size),
            lbx=lbx, ubx=ubx
        )
        dv_opt = sol['x'].full().flatten()

        # 回退线搜索
        alpha = 1.0
        f0 = np.linalg.norm(r) ** 2
        while alpha > 1e-6:
            v_trial = w0 + alpha * dv_opt
            f_trial = np.linalg.norm(F1_fun(v_trial).full().flatten()) ** 2
            if f_trial < f0:
                break
            alpha *= 0.5

        # 更新
        w0 = w0 + alpha * dv_opt
        if np.linalg.norm(dv_opt) < tol:
            converged = True
            break

    # 恢复 w = [s; p]，p = exp(q)
    s_opt = w0[:n_var]
    q_opt = w0[n_var:]
    p_opt = np.exp(q_opt)
    w_opt = np.concatenate([s_opt, p_opt])

    return {'x': w_opt, 'iterations': k + 1, 'converged': converged}
