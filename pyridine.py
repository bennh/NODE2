import casadi as ca

def pyridine_ode(y: ca.MX, p: ca.MX) -> ca.MX:
    """
    吡啶反应系统的 ODE 右端函数。

    参数:
    ----------
    y : MX
        状态向量 [Pyridin, Piperidin, Pentylamin, N-Pentylpiperidin, Dipentylamin, Ammonia, Pentan]。
    p : MX
        参数向量，共 11 个反应速率常数。

    返回:
    -------
    dy : MX
        状态变化率向量。
    """
    # 状态分量
    A, B, C, D, E, F, G = y[0], y[1], y[2], y[3], y[4], y[5], y[6]
    # 参数分量
    p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11 = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10]

    # 建立微分方程
    dy = ca.MX.zeros(7, 1)
    dy[0] = -p1*A + p9*B
    dy[1] = p1*A - p2*B - p3*B*C + p7*D - p9*B + p10*D*F
    dy[2] = p2*B - p3*B*C - 2*p4*C**2 - p6*C + p8*E + p10*D*F + 2*p11*E*F
    dy[3] = p3*B*C - p5*D - p7*D - p10*D*F
    dy[4] = p4*C**2 + p5*D - p8*E - p11*E*F
    dy[5] = p3*B*C + p4*C**2 + p6*C - p10*D*F - p11*E*F
    dy[6] = p6*C + p7*D + p8*E

    return dy

def make_pyridine_integrator(dt: float) -> ca.Function:
    """
    构造吡啶问题的数值积分器（使用 RK 步进）。

    参数:
    ----------
    dt : float
        每个射击段的时间步长。

    返回:
    -------
    integrator : casadi.Function
        可调用的积分器：res = integrator(x0=..., p=...), 返回 'xf'。
    """
    # 定义符号变量
    x = ca.MX.sym('x', 7)
    p = ca.MX.sym('p', 11)
    # 定义微分方程
    f = pyridine_ode(x, p)
    dae = {'x': x, 'p': p, 'ode': f}
    opts = {'tf': dt}
    integrator = ca.integrator('F', 'rk', dae, opts)
    return integrator
