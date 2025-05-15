import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple
from cnlls_solver import solve_cnlls_ipopt, solve_cnlls_gauss_newton


def setup_multiple_shooting(
        integrator: ca.Function,
        N: int,
        nx: int,
        np_p: int
) -> Tuple[ca.MX, List[ca.MX], ca.MX, List[ca.MX], ca.MX]:
    """
    Set up a direct multiple shooting discretization.

    Parameters:
    ----------
    integrator : casadi.Function
        The integrator created by create_integrator().
    N : int
        Number of shooting intervals.
    nx : int
        State dimension.
    np_p : int
        Parameter dimension.

    Returns:
    -------
    w       : MX
        The stacked optimization variable [s_0, ..., s_{N-1}, p].
    X_end   : List[MX]
        List of length N: the state at the end of each interval.
    g       : MX
        Continuity constraints of size (N-1)*nx.
    S_vars  : List[MX]
        The symbolic shooting initial states s_i.
    P_var   : MX
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
        # Continuity: x_end - s_{i+1} = 0
        if i < N - 1:
            g_constr.append(x_end - S_vars[i + 1])

    # Stack continuity constraints
    g = ca.vertcat(*g_constr) if g_constr else ca.MX.zeros(0)

    # Stack all decision variables [s_0; ...; s_{N-1}; p]
    w = ca.vertcat(*(S_vars + [P_var]))

    return w, X_end, g, S_vars, P_var


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


def simulate_lv_data(x0, p_true, t_grid):
    F = make_lv_integrator(t_grid[1] - t_grid[0])
    x = x0
    data = [x0]
    for _ in range(len(t_grid) - 1):
        x = F(x0=x, p=p_true)['xf'].full().flatten()
        data.append(x)
    return np.array(data)


if __name__ == '__main__':
    N = 50
    nx = 2
    np_p = 4
    T = 10.0
    t_grid = np.linspace(0, T, N + 1)
    dt = t_grid[1] - t_grid[0]

    # 真实参数
    p_true = np.array([2.0, 1.0, 1.5, 0.75])
    x0_true = np.array([2.0, 1.0])
    y_meas = simulate_lv_data(x0_true, p_true, t_grid)

    # 构造 integrator 和 multiple shooting 问题
    F = make_lv_integrator(dt)
    w, X_end, g, S_vars, P_var = setup_multiple_shooting(F, N, nx, np_p)

    # 目标函数 = 拟合误差平方和
    F1 = ca.vertcat(*[X_end[i] - y_meas[i + 1] for i in range(N)])

    # 创建 NLP 求解器
    # 初始值设置（x, p）
    w0 = np.concatenate([y_meas[:-1].reshape(-1), np.array([1.0, 1.0, 1.0, 0.5])])

    # sol_ipopt = solve_cnlls_ipopt(
    #     w=w,
    #     F1=F1,
    #     g=g,
    #     w0=w0
    # )
    #
    # w_opt_ipopt = sol_ipopt['x']
    # p_est_ipopt = w_opt_ipopt[-4:]  # 最后 4 个是参数
    # print("IPOPT estimated parameters:", p_est_ipopt)

    sol_gn = solve_cnlls_gauss_newton(
        w=w,
        F1=F1,
        g=g,
        w0=w0
    )

    w_opt_gn = sol_gn['x']
    p_est_gn = w_opt_gn[-4:]  # 最后 4 个是参数
    print("Gauss-Newton estimated parameters:", p_est_gn)

    # 用拟合参数重新模拟轨迹
    y_est_gn = simulate_lv_data(y_meas[0], p_est_gn, t_grid)

    # 单图展示 Prey 和 Predator 轨迹
    plt.figure(figsize=(8, 5))

    # 真实测量数据（圆点）
    plt.plot(t_grid, y_meas[:, 0], 'o', label='Measured Prey', alpha=0.6)
    plt.plot(t_grid, y_meas[:, 1], 's', label='Measured Predator', alpha=0.6)

    # 用拟合参数重新模拟的轨迹（实线）
    plt.plot(t_grid, y_est_gn[:, 0], '-', label='Estimated Prey', linewidth=2)
    plt.plot(t_grid, y_est_gn[:, 1], '--', label='Estimated Predator', linewidth=2)

    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Prey and Predator Dynamics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()