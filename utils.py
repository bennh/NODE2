import numpy as np
import casadi as ca
from typing import Tuple

def generate_measurements(
    integrator: ca.Function,
    x0: np.ndarray,
    p_true: np.ndarray,
    N: int,
    noise_level: float = 0.01,
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic measurement data for parameter estimation.

    Parameters:
    ----------
    integrator : casadi.Function
        ODE integrator created using create_integrator.
    x0 : np.ndarray
        True initial state of the system.
    p_true : np.ndarray
        True parameter values.
    N : int
        Number of shooting intervals.
    noise_level : float, optional
        Standard deviation of Gaussian noise (default 0.01).
    seed : int, optional
        Random seed for reproducibility (default 42).

    Returns:
    -------
    measurements : np.ndarray
        Generated noisy measurements (N × nx).
    """
    np.random.seed(seed)
    measurements = np.zeros((N, x0.size))

    # Generate noisy data for each shooting node
    x_prev = x0
    for i in range(N):
        res = integrator(x0=x_prev, p=p_true)
        x_curr = res['xf'].full().flatten()
        noisy_data = x_curr + noise_level * np.random.randn(*x_curr.shape)
        measurements[i, :] = noisy_data
        x_prev = x_curr

    return measurements


def save_measurements(filename: str, data: np.ndarray):
    """
    Save measurement data to a .npy file.

    Parameters:
    ----------
    filename : str
        Path to the .npy file.
    data : np.ndarray
        Measurement data to save.
    """
    np.save(filename, data)


def load_measurements(filename: str) -> np.ndarray:
    """
    Load measurement data from a .npy file.

    Parameters:
    ----------
    filename : str
        Path to the .npy file.

    Returns:
    -------
    data : np.ndarray
        Loaded measurement data.
    """
    return np.load(filename)


def add_noise(data: np.ndarray, noise_level: float = 0.01, seed: int = 42) -> np.ndarray:
    """
    Add Gaussian noise to the measurement data.

    Parameters:
    ----------
    data : np.ndarray
        Clean measurement data.
    noise_level : float, optional
        Standard deviation of the noise (default 0.01).
    seed : int, optional
        Random seed for reproducibility (default 42).

    Returns:
    -------
    noisy_data : np.ndarray
        Noisy measurement data.
    """
    np.random.seed(seed)
    return data + noise_level * np.random.randn(*data.shape)
