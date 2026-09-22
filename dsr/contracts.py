from dataclasses import dataclass
import numpy as np

@dataclass
class DetectionResult:
    best_trial_idx: int
    best_observed_sr: float
    noise_threshold_sr: float
    psr_confidence: float
    is_valid: bool
    n_periods: int
    n_trials: int

def validate_returns_matrix(returns: np.ndarray) -> tuple[int, int]:
    """Valida che la matrice dei rendimenti sia 2D e non vuota."""
    if not isinstance(returns, np.ndarray):
        returns = np.array(returns)
    if returns.ndim != 2:
        raise ValueError(f"Attesa matrice 2D (T_periods, N_trials), ricevuta dim {returns.ndim}")
    T, N = returns.shape
    if T < 10 or N < 1:
        raise ValueError(f"Dimensioni insufficienti: T={T}, N={N}")
    return T, N
