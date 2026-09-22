import numpy as np
def generate_synthetic_backtest_matrix(t_periods: int=500, n_trials: int=1000, seed: int=42)->tuple[np.ndarray,int]:
    rng=np.random.default_rng(seed)
    returns=rng.normal(loc=0.0, scale=0.01, size=(t_periods, n_trials))
    true_edge_idx=42
    signal=rng.normal(loc=0.01, scale=0.001, size=(t_periods,))
    returns[:,42]=signal
    return(returns, true_edge_idx)
if __name__ == "__main__":
    r, idx = generate_synthetic_backtest_matrix()
    print("Done:", r.shape, idx)