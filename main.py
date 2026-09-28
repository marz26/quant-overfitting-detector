import matplotlib.pyplot as plt
import numpy as np
from dsr.metrics import compute_sharpe_ratios
from dsr.pipeline import run_dsr_pipeline
from tests.mock_data import generate_synthetic_backtest_matrix


def main():
  print("🚀 Avvio della pipeline Deflated Sharpe Ratio (DSR)...")

  # 1. Generazione dei dati sintetici (con segnale iniettato)
  t_periods, n_trials = 1000, 500
  returns, true_idx = generate_synthetic_backtest_matrix(
      t_periods=t_periods, n_trials=n_trials, seed=42
  )

  # 2. Esecuzione della pipeline DSR
  result = run_dsr_pipeline(returns, confidence_threshold=0.95)

  # 3. Stampa del Report a terminale
  print("\n" + "=" * 50)
  print("📊 REPORT DI ANALISI - QUANT OVERFITTING DETECTOR")
  print("=" * 50)
  print(f"• Numero di tentativi testati (N):  {n_trials}")
  print(f"• Periodi storici analizzati (T):    {t_periods}")
  print(
      f"• Indice trial migliore trovato:     {result['best_trial_index']}"
      f" (True edge era a {true_idx})"
  )
  print(f"• Miglior Sharpe osservato:          {result['best_sharpe']:.4f}")
  print(f"• Soglia attesa del rumore (E[max]): {result['expected_max_sharpe']:.4f}")
  print(f"• Confidenza DSR (PSR):              {result['dsr_confidence'] * 100:.2f}%")
  
  verdict = "✅ VALIDATO (Edge Reale)" if result['is_valid'] else "❌ BOCCIATO (Overfitted / Rumore)"
  print(f"• Verdetto Finale:                   {verdict}")
  print("=" * 50)

  # 4. Generazione del Grafico (Giorno 8: dsr_report.png)
  print("\n📈 Generazione del grafico di report (dsr_report.png)...")
  sharpe_ratios = compute_sharpe_ratios(returns)

  plt.figure(figsize=(10, 6))
  plt.hist(
      sharpe_ratios,
      bins=50,
      color="skyblue",
      edgecolor="black",
      alpha=0.7,
      label="Trial Sharpe Ratios (Distribuzione Rumore)",
  )
  plt.axvline(
      result["expected_max_sharpe"],
      color="red",
      linestyle="--",
      linewidth=2,
      label=f"Soglia Rumore Max: {result['expected_max_sharpe']:.2f}",
  )
  plt.axvline(
      result["best_sharpe"],
      color="green",
      linestyle="-",
      linewidth=2,
      label=f"Miglior Strategia: {result['best_sharpe']:.2f}",
  )

  plt.title(
      "Deflated Sharpe Ratio (DSR) - Overfitting Detection Report",
      fontsize=14,
      fontweight="bold",
  )
  plt.xlabel("Sharpe Ratio", fontsize=12)
  plt.ylabel("Frequenza", fontsize=12)
  plt.legend(loc="upper right", fontsize=10)
  plt.grid(True, linestyle=":", alpha=0.6)

  plt.tight_layout()
  plt.savefig("dsr_report.png", dpi=300)
  print("✨ Grafico salvato con successo come 'dsr_report.png'!")


if __name__ == "__main__":
  main()