import numpy as np
import pytest
from dsr.pipeline import run_dsr_pipeline
from tests.mock_data import generate_synthetic_backtest_matrix


def test_pipeline_pure_noise():
  # Genera una matrice di puro rumore bianco (senza segnale iniettato)
  rng = np.random.default_rng(123)
  returns = rng.normal(loc=0.0, scale=0.01, size=(500, 200))

  result = run_dsr_pipeline(returns, confidence_threshold=0.95)

  # Con il puro rumore, la confidenza DSR non deve superare la soglia del 95%
  assert result["dsr_confidence"] < 0.95
  assert result["is_valid"] is False


def test_pipeline_with_edge():
  # Usa il tuo generatore mock che inietta un segnale forte in una colonna
  returns, true_idx = generate_synthetic_backtest_matrix(
      t_periods=1000, n_trials=500, seed=42
  )

  result = run_dsr_pipeline(returns, confidence_threshold=0.95)

  # Con un edge forte, il DSR deve validare la strategia ottima
  assert result["dsr_confidence"] >= 0.95
  assert result["is_valid"] is True
  