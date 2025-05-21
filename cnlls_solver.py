import casadi as ca
import numpy as np

def solve_cnlls_ipopt(
        w: ca.MX,
        F1: ca.MX,
        g: ca.MX,
        w0: np.ndarray,
        n_params: int,
        opts: dict = None
) -> dict:
    """
    使用 IPOPT 求解带约束的非线性最小二乘问题：

        min 0.5||F1(w)||^2  s.t. g(w) = 0

    内部对参数部分进行对数重参数化，使其始终为正。

    参数:
    ----------
    w : casadi.MX
        优化变量向量。
    F1 : casadi.MX
        残差向量，用于二次和目标。
    g : casadi.MX
        等式约束向量（例如多重射击的连续性和初始约束）。
    w0 : numpy.ndarray
        初始猜测（包含状态和参数）。
    n_params : int
        参数部分的维度，位于 w 的末尾。
    opts : dict, optional
        IPOPT 参数。

    返回:
    -------
    result : dict
        - 'x': numpy.ndarray, 求解得到的 [s; p] 最终向量。
        - 'iterations': int, 迭代次数。
        - 'converged': bool, 是否满足收敛标准。
    """
    # w = [s; p]，将 p 转为对数形式 q
    n_var = w.size1() - n_params
    s = w[0:n_var]
    q = w[n_var:]
    w_reparam = ca.vertcat(s, ca.exp(q))

    # 替换 F1, g 中的 w 为 [s; exp(q)]
    F1_r = ca.substitute(F1, w, w_reparam)
    g_r = ca.substitute(g, w, w_reparam)

    # 构造 CasADi 函数
    v = ca.vertcat(s, q)
    F1_fun = ca.Function('F1_fun', [v], [F1_r])
    g_fun = ca.Function('g_fun', [v], [g_r])
    J1_fun = ca.Function('J1_fun', [v], [ca.jacobian(F1_r, v)])
    J2_fun = ca.Function('J2_fun', [v], [ca.jacobian(g_r, v)])

    # 初始化：将参数部分转为对数
    s0 = w0[:n_var]
    p0 = w0[n_var:]
    v0 = np.concatenate([s0, np.log(p0)])

    # 调用 IPOPT 求解器
    prob = {'x': v, 'f': 0.5 * ca.sumsqr(F1_r), 'g': g_r}
    solver = ca.nlpsol('solver', 'ipopt', prob, opts or {})
    sol = solver(x0=v0, lbg=np.zeros(g_r.size1()), ubg=np.zeros(g_r.size1()))
    v_opt = sol['x'].full().flatten()

    # 恢复 [s; p]
    s_opt = v_opt[:n_var]
    q_opt = v_opt[n_var:]
    p_opt = np.exp(q_opt)
    w_opt = np.concatenate([s_opt, p_opt])
    return {'x': w_opt, 'iterations': sol.stats()['iter_count'], 'converged': sol.stats()['return_status'] == 'Solve_Succeeded'}


def solve_cnlls_gauss_newton(
        w: ca.MX,
        F1: ca.MX,
        g: ca.MX,
        w0: np.ndarray,
        n_params: int,
        max_iter: int = 30,
        tol: float = 1e-8,
        qp_opts: dict = None
) -> dict:
    """
    使用广义 Gauss-Newton (GN) 迭代求解带约束的非线性最小二乘问题：
    
      min 0.5||F1(w)||^2  s.t. g(w)=0

    迭代步骤：
      - 在当前点线性化残差 F1 和约束 g
      - 构造并求解线性化的二次规划 (QP)
      - 回退线搜索 (damping) 保证目标降低

    参数:
    ----------
    w : casadi.MX
        决策变量向量。
    F1 : casadi.MX
        残差向量。
    g : casadi.MX
        等式约束向量。
    w0 : numpy.ndarray
        初始猜测。
    n_params : int
        参数维度。
    max_iter : int
        最大迭代次数。
    tol : float
        收敛阈值 (对 dw 的范数)。
    qp_opts : dict, optional
        QP 求解器参数。

    返回:
    -------
    result : dict
        - 'x': numpy.ndarray, 最终 [s; p] 向量。
        - 'iterations': int, 迭代次数。
        - 'converged': bool, 是否收敛。
    """
    # 创建 CasADi 函数
    F1_fun = ca.Function('F1_fun', [w], [F1])
    g_fun  = ca.Function('g_fun', [w], [g])
    J1_fun = ca.Function('J1_fun', [w], [ca.jacobian(F1, w)])
    J2_fun = ca.Function('J2_fun', [w], [ca.jacobian(g, w)])

    xk = w0.copy()
    n = w.size1()
    converged = False

    for k in range(max_iter):
        # 计算当前残差和雅可比
        r = F1_fun(xk).full().flatten()  # 目标残差
        h = g_fun(xk).full().flatten()   # 约束残差
        J1 = J1_fun(xk).full()           # 目标雅可比
        J2 = J2_fun(xk).full()           # 约束雅可比

        # 构造 QP：min 0.5*dw^T H dw + c^T dw, s.t. J2*dw + h = 0
        H = J1.T @ J1
        # 增加小阻尼 (Levenberg-Marquardt)
        mu = 1e-3
        H = H + mu * np.eye(n)

        c = J1.T @ r
        dw = ca.MX.sym('dw', n)
        qp = {'x': dw, 
              'f': 0.5 * ca.mtimes([dw.T, H, dw]) + ca.dot(c.T, dw), 
              'g': ca.mtimes(J2, dw) + h}
        qp_solver = ca.qpsol('qp_solver', 'qpoases', qp, qp_opts or {})

        lbx = -np.inf * np.ones(n)
        ubx =  np.inf * np.ones(n)
        # 参数部分可以设置非负下界 (视需要)
        sol_qp = qp_solver(x0=np.zeros(n), lbg=np.zeros(h.size), ubg=np.zeros(h.size), lbx=lbx, ubx=ubx)
        dw_opt = sol_qp['x'].full().flatten()

        # 回退线搜索以保证目标下降
        alpha = 1.0
        f0 = np.linalg.norm(r)**2
        while alpha > 1e-6:
            x_trial = xk + alpha * dw_opt
            f_trial = np.linalg.norm(F1_fun(x_trial).full().flatten())**2
            if f_trial < f0:
                break
            alpha *= 0.5
        xk = xk + alpha * dw_opt

        if np.linalg.norm(dw_opt) < tol:
            converged = True
            break

    return {'x': xk, 'iterations': k+1, 'converged': converged}
