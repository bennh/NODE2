import numpy as np
import casadi as ca
from typing import Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def generate_measurements(integrator: ca.Function, x0: np.ndarray, p_true: np.ndarray, N: int, noise_level: float = 0.01, seed: int = 42) -> np.ndarray:
    """
    Generate synthetic measurement data by simulating the system over N intervals with optional noise.

    Parameters:
    -----------
    integrator : casadi.Function
        CasADi integrator for the system, returning 'xf' state.
    x0 : numpy.ndarray
        Initial state.
    p_true : numpy.ndarray
        True parameter vector.
    N : int
        Number of intervals (simulations) to generate.
    noise_level : float
        Standard deviation of Gaussian noise to add.
    seed : int
        Random seed for reproducibility.

    Returns:
    -------
    measurements : numpy.ndarray
        Array of shape (N, nx) containing noisy measurements (no initial state).
    """
    np.random.seed(seed)
    measurements = np.zeros((N, x0.size))
    x_prev = x0
    for i in range(N):
        res = integrator(x0=x_prev, p=p_true)
        x_curr = res['xf'].full().flatten()
        noisy_data = x_curr + noise_level * np.random.randn(*x_curr.shape)
        measurements[i, :] = noisy_data
        x_prev = x_curr
    return measurements

def save_measurements(filename: str, data: np.ndarray):
    """Save measurement data to a .npy file."""
    np.save(filename, data)

def load_measurements(filename: str) -> np.ndarray:
    """Load measurement data from a .npy file."""
    return np.load(filename)

def add_noise(data: np.ndarray, noise_level: float = 0.01, eps: float = 0.0, seed: int = 42) -> np.ndarray:
    """
    Add Gaussian noise to the data and clip values below eps.
    """
    np.random.seed(seed)
    noisy = data + noise_level * np.random.randn(*data.shape)
    return np.clip(noisy, a_min=eps, a_max=None)

def evaluate_fit(y_true: np.ndarray, y_pred: np.ndarray, t_grid: np.ndarray, label: str = 'Fit'):
    """
    Compute and print error metrics between true and predicted trajectories.
    """
    species_names = ['Pyridin', 'Piperidin', 'Pentylamin', 'N-Pentylpiperidin',
                     'Dipentylamin', 'Ammonia', 'Pentan']
    for i, species in enumerate(species_names):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        print(f"[{label}] {species} Errors:")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  MAE  = {mae:.4f}")
