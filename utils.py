import numpy as np
import casadi as ca
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Tuple

def generate_measurements(
        integrator: ca.Function,
        x0: np.ndarray,
        p_true: np.ndarray,
        N: int,
        noise_level: float = 0.0,
        seed: int = 42
) -> np.ndarray:
    """
    生成合成测量数据。

    每个时间步用积分器从上一状态进行前向仿真，并加入高斯噪声。

    参数:
    ----------
    integrator : casadi.Function
        ODE 积分器。
    x0 : np.ndarray
        初始状态。
    p_true : np.ndarray
        真实参数向量。
    N : int
        数据点数（积分步数）。
    noise_level : float
        噪声标准差。
    seed : int
        随机种子。

    返回:
    -------
    measurements : np.ndarray
        形状 (N+1, nx) 的测量数据，包括初始状态在内。
    """
    np.random.seed(seed)
    nx = x0.size
    measurements = np.zeros((N+1, nx))
    measurements[0] = x0
    x = x0.copy()
    for i in range(1, N+1):
        res = integrator(x0=x, p=p_true)
        x = res['xf'].full().flatten()
        noisy = x + noise_level * np.random.randn(nx)
        measurements[i] = noisy
    return measurements

def add_noise(data: np.ndarray, noise_level: float = 0.01, eps: float = 0.0, seed: int = 42) -> np.ndarray:
    """
    给测量数据加入高斯噪声。

    参数:
    ----------
    data : np.ndarray
        原始干净数据 (形状如 (N+1, nx) )。
    noise_level : float
        噪声标准差。
    eps : float
        噪声后下限，避免负值。
    seed : int
        随机种子。

    返回:
    -------
    noisy_data : np.ndarray
        添加噪声后的数据。
    """
    np.random.seed(seed)
    noisy = data + noise_level * np.random.randn(*data.shape)
    return np.clip(noisy, a_min=eps, a_max=None)

def evaluate_fit(y_true: np.ndarray, y_pred: np.ndarray, t_grid: np.ndarray, label: str = "Fit"):
    """
    评估并输出预测轨迹与真实测量值之间的误差（RMSE 和 MAE）。
    
    参数:
    ----------
    y_true : np.ndarray
        测量数据 (形状 (N+1, nx))。
    y_pred : np.ndarray
        模型预测数据 (形状 (N+1, nx))。
    t_grid : np.ndarray
        时间点向量 (长度 N+1)。
    label : str
        标签前缀，用于输出提示 (如 "Gauss-Newton")。
    """
    species_names = ['Pyridin', 'Piperidin', 'Pentylamin', 'N-Pentylpiperidin',
                     'Dipentylamin', 'Ammonia', 'Pentan']
    print(f"=== {label} 拟合评估 ===")
    for i, species in enumerate(species_names):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae  = mean_absolute_error(   y_true[:, i], y_pred[:, i])
        print(f"{species}: RMSE = {rmse:.4f}, MAE = {mae:.4f}")
