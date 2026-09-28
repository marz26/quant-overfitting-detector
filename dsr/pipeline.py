import numpy as np
from dsr.contracts import validate_returns_matrix
from dsr.metrics import compute_higher_moments, compute_sharpe_ratios
from dsr.threshold import compute_deflated_sharpe_ratio, compute_expected_maximum_sharpe


def run_dsr_pipeline(returns: np.ndarray, confidence_threshold: float = 0.95) -> dict:
  """Orchestra l'intera analisi di overfitting con il Deflated Sharpe Ratio.

  Riceve la matrice T x N, calcola le metriche, individua la strategia migliore
  e restituisce il verdetto statistico.
  """
  # 1. Validazione della matrice in ingresso (Giorno 1 contracts)
  t_periods, n_trials = validate_returns_matrix(returns)

  # 2. Calcolo metriche per tutti i trial (Giorni 3-4 metrics)
  sharpe_ratios = compute_sharpe_ratios(returns)
  skewness_arr, kurtosis_arr = compute_higher_moments(returns)

  # 3. Individuazione della strategia con lo Sharpe più alto osservato
  best_idx = int(np.argmax(sharpe_ratios))
  best_sharpe = float(sharpe_ratios[best_idx])
  best_skew = float(skewness_arr[best_idx])
  best_kurt = float(kurtosis_arr[best_idx])

  # 4. Calcolo della soglia del rumore estremo (Giorni 5-6 threshold)
  expected_max_sr = compute_expected_maximum_sharpe(
      sharpe_ratios=sharpe_ratios,
      skewness=best_skew,
      kurtosis=best_kurt,
      num_trials=n_trials,
  )

  # 5. Calcolo del Deflated Sharpe Ratio (confidenza finale)
  dsr_confidence = compute_deflated_sharpe_ratio(
      observed_sharpe=best_sharpe,
      expected_max_sharpe=expected_max_sr,
      sample_size=t_periods,
      skewness=best_skew,
      kurtosis=best_kurt,
  )

  # 6. Verdetto finale basato sulla soglia di confidenza (es. 95%)
  is_valid = dsr_confidence >= confidence_threshold

  return {
      "best_trial_index": best_idx,
      "best_sharpe": best_sharpe,
      "expected_max_sharpe": expected_max_sr,
      "dsr_confidence": dsr_confidence,
      "is_valid": is_valid,
  }