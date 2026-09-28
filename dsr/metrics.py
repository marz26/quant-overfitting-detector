import numpy as np
def compute_sharpe_ratios(returns: np.ndarray, annualization_factor: float=np.sqrt(252))->np.ndarray:
    mean=np.mean(returns, axis=0)
    std=np.std(returns,axis=0,ddof=1)
    sharpe_ratios=np.where(std==0,0.0,mean/std*annualization_factor)
    return sharpe_ratios

def compute_higher_moments(returns: np.ndarray)->tuple[np.ndarray,np.ndarray]:
    mean=np.mean(returns,axis=0)
    std=np.std(returns,axis=0,ddof=1)
    z=np.where(std==0,0.0,(returns-mean)/std)
    skewness=np.mean(z**3,axis=0)
    kurtosis=np.mean(z**4,axis=0)
    return skewness,kurtosis
