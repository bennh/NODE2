import casadi as ca

def notorious_ode(y: ca.MX, p: ca.MX) -> ca.MX:
    """
    Bock-Kostina-Schlöder (2007) 中的 notorious test example 的右端函数。

    y: [y1, y2]，状态向量
    p: [p1, p2]，参数向量
    """
    y1, y2 = y[0], y[1]
    p1, p2 = p[0], p[1]

    dy1 = -p1 * y1 + p2 * y2
    dy2 = p1 * y1 - p2 * y2**2

    return ca.vertcat(dy1, dy2)

def make_notorious_integrator(dt: float = 0.2) -> ca.Function:
    """
    构造 notorious test 的 CasADi 积分器，默认使用显式 Runge-Kutta 方法。

    参数:
    ----------
    dt : float
        每个射击段的时间步长，默认 0.2。

    返回:
    -------
    integrator : casadi.Function
        可调用的积分器 integrator(x0=..., p=...) → {'xf': ...}
    """
    x = ca.MX.sym('x', 2)     # 状态变量 y = [y1, y2]
    p = ca.MX.sym('p', 2)     # 参数向量 p = [p1, p2]
    f = notorious_ode(x, p)

    dae = {'x': x, 'p': p, 'ode': f}
    opts = {'tf': dt}
    integrator = ca.integrator('notorious_integrator', 'rk', dae, opts)

    return integrator

def get_notorious_initial_state() -> ca.DM:
    """返回 notorious test 的初始状态。"""
    return ca.DM([1.0, 0.0])

def get_notorious_true_parameters() -> ca.DM:
    """返回 notorious test 的真实参数。"""
    return ca.DM([1.0, 0.5])
