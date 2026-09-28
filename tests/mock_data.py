import numpy as np

def generate_synthetic_backtest_matrix(t_periods: int = 500, n_trials: int = 1000, seed: int = 42) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    # Rumore bianco giornaliero (es. volatilità del 1% al giorno)
    returns = rng.normal(loc=0.0001, scale=0.01, size=(t_periods, n_trials))
    true_edge_idx = 42
    
    # Inseriamo un piccolo edge reale nella colonna 42
    signal = rng.normal(loc=0.0008, scale=0.01, size=(t_periods,))
    returns[:, true_edge_idx] = signal
    return returns, true_edge_idx