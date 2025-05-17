import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple
from cnlls_solver import solve_cnlls_ipopt, solve_cnlls_gauss_newton
from dynamics import make_pyridine_integrator, make_lv_integrator
from src.cnlls_solver import solve_cnlls_gauss_newton_logparam
from utils import add_noise, evaluate_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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


def simulate_lv_data(x0, p_true, t_grid):
    F = make_lv_integrator(t_grid[1] - t_grid[0])
    x = x0
    data = [x0]
    for _ in range(len(t_grid) - 1):
        x = F(x0=x, p=p_true)['xf'].full().flatten()
        data.append(x)
    return np.array(data)


def simulate_pyridine_data(x0, p, t_grid):
    F = make_pyridine_integrator(t_grid[1] - t_grid[0])
    x = x0
    result = [x0]
    for _ in range(len(t_grid) - 1):
        x = F(x0=x, p=p)['xf'].full().flatten()
        result.append(x)
    return np.array(result)


if __name__ == '__main__':
    N = 20
    nx = 7
    np_p = 11
    T = 6.0
    t_grid = np.linspace(0, T, N + 1)
    dt = t_grid[1] - t_grid[0]

    # True parameters
    p_true = np.array([1.81, 0.894, 29.4, 9.21, 0.058, 2.43, 0.0644, 5.55, 0.0201, 0.577, 2.15])
    x0_true = np.zeros(nx)
    x0_true[0] = 1.0  # Initial pyridine concentration
    y_meas = simulate_pyridine_data(x0_true, p_true, t_grid)
    y_meas_noised = add_noise(y_meas, noise_level=0.0)
    # Restore the initial value
    y_meas_noised[0, :] = y_meas[0, :]

    # Construct integrator and multiple shooting
    F = make_pyridine_integrator(dt)
    w, X_end, g, S_vars, P_var = setup_multiple_shooting(F, N, nx, np_p)

    # F1 = 拟合误差平方和
    F1 = ca.vertcat(*[X_end[i] - y_meas_noised[i + 1] for i in range(N)])

    # Create NLP solvers
    # Initial values（x, p）
    w0 = np.concatenate([y_meas_noised[:-1].reshape(-1), np.ones(np_p)])

    # sol_ipopt = solve_cnlls_ipopt(
    #     w=w,
    #     F1=F1,
    #     g=g,
    #     w0=w0
    # )
    #
    # w_opt_ipopt = sol_ipopt['x']
    # p_est_ipopt = w_opt_ipopt[-np_p:]  # 最后 np_p 个是参数
    # print("IPOPT estimated parameters:", p_est_ipopt)

    sol_gn = solve_cnlls_gauss_newton_logparam(
        w=w,
        F1=F1,
        g=g,
        w0=w0
    )

    w_opt_gn = sol_gn['x']
    p_est_gn = w_opt_gn[-np_p:]  # 最后 np_p 个是参数
    print("Gauss-Newton estimated parameters:", p_est_gn)

    # # --- 拟合轨迹 ---
    # y_est_ipopt = simulate_pyridine_data(y_meas_noised[0], p_est_ipopt, t_grid)
    y_est_gn = simulate_pyridine_data(y_meas_noised[0], p_est_gn, t_grid)

    # 按维度分别比较差异
    # evaluate_fit(y_meas_noised, y_est_ipopt, t_grid, label="IPOPT")
    evaluate_fit(y_meas_noised, y_est_gn, t_grid, label="Gauss-Newton")


    # --- 可视化对比 ---
    plt.figure(figsize=(8, 8))

    # 测量值
    plt.plot(t_grid, y_meas_noised[:, 0], 'o', label='Pyridine', alpha=0.5)
    plt.plot(t_grid, y_meas_noised[:, 5], 's', label='Ammonia', alpha=0.5)
    plt.plot(t_grid, y_meas_noised[:, 6], '^', label='Pentan', alpha=0.5)
    plt.plot(t_grid, y_meas_noised[:, 3], '.', label='N-Pentylpiperidin', alpha=0.5)
    plt.plot(t_grid, y_meas_noised[:, 1], 'D', label='Piperdin', alpha=0.5)

    # IPOPT 拟合轨迹
    # plt.plot(t_grid, y_est_ipopt[:, 0], '-', label='IPOPT Pyridine', linewidth=2)
    # plt.plot(t_grid, y_est_ipopt[:, 5], '-', label='IPOPT Ammonia', linewidth=2)
    # plt.plot(t_grid, y_est_ipopt[:, 6], '-', label='IPOPT Pentan', linewidth=2)
    # plt.plot(t_grid, y_est_ipopt[:, 3], '-', label='IPOPT N-Pentylpiperidin', linewidth=2)
    # plt.plot(t_grid, y_est_ipopt[:, 1], '-', label='IPOPT Piperdin', linewidth=2)

    # Gauss-Newton 拟合轨迹
    plt.plot(t_grid, y_est_gn[:, 0], '-', label='GN Pyridine', linewidth=2)
    plt.plot(t_grid, y_est_gn[:, 5], '-', label='GN Ammonia', linewidth=2)
    plt.plot(t_grid, y_est_gn[:, 6], '-', label='GN Pentan', linewidth=2)
    plt.plot(t_grid, y_est_gn[:, 3], '-', label='GN N-Pentylpiperidin', linewidth=2)
    plt.plot(t_grid, y_est_gn[:, 1], '-', label='GN Piperdin', linewidth=2)

    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Pyridine Dynamics: Measured vs Fitted')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
