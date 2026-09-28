import numpy as np
import scipy.stats as stats
def compute_expected_maximum_sharpe(sharpe_ratios:np.ndarray,skewness:np.ndarray | float, kurtosis:np.ndarray | float, num_trials:int)->float:
    if num_trials <= 1:
        return 0.0

    # Costante di Eulero-Mascheroni
    gamma = 0.57721566490153286060
    
    # Quantili della normale standard per il massimo di N variabili casuali
    arg1 = 1.0 - 1.0 / num_trials
    arg2 = 1.0 - 1.0 / (num_trials * np.e)
    
    z1 = stats.norm.ppf(arg1)
    z2 = stats.norm.ppf(arg2)
    
    # Deviazione standard campionaria degli Sharpe osservati tra i trial
    sigma_sr = np.std(sharpe_ratios, ddof=1) if len(sharpe_ratios) > 1 else 1.0
    
    # Valore atteso del massimo Sharpe del rumore
    expected_max_sharpe = sigma_sr * ((1.0 - gamma) * z1 + gamma * z2)
    
    return float(expected_max_sharpe)


def compute_deflated_sharpe_ratio(
    observed_sharpe: float,
    expected_max_sharpe: float,
    sample_size: int,
    skewness: float,
    kurtosis: float,
) -> float:
    if sample_size <= 2:
        return 0.0

    # Errore standard dello Sharpe ratio corretto per skewness e kurtosis
    variance_sr = (1.0 / (sample_size - 1.0)) * (
        1.0 - skewness * observed_sharpe + ((kurtosis - 1.0) / 4.0) * (observed_sharpe ** 2)
    )
    
    if variance_sr <= 0:
        standard_error = 1e-6
    else:
        standard_error = np.sqrt(variance_sr)

    # Statistica Z per la CDF
    z_stat = (observed_sharpe - expected_max_sharpe) / standard_error
    
    # Probabilità cumulativa (DSR confidence)
    dsr_prob = stats.norm.cdf(z_stat)
    
    return float(dsr_prob)
