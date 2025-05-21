import numpy as np
import casadi as ca
from typing import Callable, List, Tuple

def setup_multiple_shooting(
        integrator: ca.Function,
        N: int,
        nx: int,
        np_p: int,
        x0: np.ndarray = None,
        nonneg: bool = False
) -> Tuple[ca.MX, List[ca.MX], ca.MX, ca.MX, ca.MX]:
    """
    构建直接多重射击问题的离散化结构。

    参数:
    ----------
    integrator : casadi.Function
        由 create_integrator 或 make_pyridine_integrator 创建的 ODE 积分器。
    N : int
        射击分段数量。
    nx : int
        状态变量维度。
    np_p : int
        参数向量维度。
    x0 : np.ndarray, optional
        已知的初始状态（若提供，则作为约束 F2）。
    nonneg : bool, optional
        是否添加状态非负性约束（生成 F3 向量）。

    返回:
    -------
    w       : MX
        决策变量堆叠向量 [s_0, ..., s_{N-1}, p]。
    X_end   : List[MX]
        长度为 N 的列表，存储每个子区间积分后的终点状态。
    g       : MX
        连续性等式约束向量，大小=(N-1)*nx。
    F2      : MX
        初始条件约束向量 (s_0 - x0)，大小=nx；若 x0=None，则为空向量。
    F3      : MX
        非负性约束向量 (堆叠所有状态 s_i)，若 nonneg=False 则为空向量。
    S_vars  : List[MX]
        各射击段初始状态的符号变量列表 [s_0, s_1, ..., s_{N-1}]。
    P_var   : MX
        参数向量的符号变量 p。
    """
    # 定义射击段的初始状态符号变量 s_i
    S_vars = [ca.MX.sym(f's_{i}', nx) for i in range(N)]
    # 定义参数向量符号变量 p
    P_var = ca.MX.sym('p', np_p)

    X_end = []
    g_constr = []
    # 对每段进行积分并建立连续性约束： x_end(i) - s_{i+1} = 0
    for i in range(N):
        res = integrator(x0=S_vars[i], p=P_var)
        x_end = res['xf']
        X_end.append(x_end)
        if i < N - 1:
            g_constr.append(x_end - S_vars[i + 1])
    g = ca.vertcat(*g_constr) if g_constr else ca.MX.zeros(0)

    # 初始条件约束 F2: 如果给定 x0，则强制 s_0 = x0
    if x0 is not None:
        x0 = np.asarray(x0).flatten()
        if x0.size != nx:
            raise ValueError("初始状态维度与 nx 不匹配")
        F2 = S_vars[0] - ca.DM(x0)
    else:
        F2 = ca.MX.zeros(0)

    # 非负性约束 F3: 可选地将所有状态变量堆叠（供约束 s_i >= 0 使用）
    if nonneg:
        all_states = ca.vertcat(*S_vars)  # 供外部设定约束 s_i >= 0
        F3 = all_states
    else:
        F3 = ca.MX.zeros(0)

    # 决策变量向量 w = [s_0; s_1; ...; s_{N-1}; p]
    w = ca.vertcat(*(S_vars + [P_var]))

    return w, X_end, g, F2, F3, S_vars, P_var
