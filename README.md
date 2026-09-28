# Quant Overfitting Detector

A professional-grade implementation of **Bailey & López de Prado’s Deflated Sharpe Ratio (DSR)** framework to detect and mitigate multiple-testing bias and overfitting in algorithmic trading strategies.

---

## 1. The Problem: Data Snooping & Multiple Testing
When backtesting $N$ trading strategies over $T$ periods, the probability that at least one strategy appears profitable purely by chance (false positive) approaches 100% as $N$ grows. Standard Sharpe ratios fail because they ignore multiple-testing bias and non-normal return distributions (skewness and kurtosis).

## 2. Mathematical Core
The DSR evaluates whether the best observed Sharpe ratio ($\widehat{SR}$) significantly exceeds the expected maximum Sharpe ratio under the null hypothesis of pure noise ($\mathbb{E}[\max SR]$), adjusted for higher moments:

$$DSR = Z\left( \frac{\widehat{SR} - \mathbb{E}[\max SR]}{\hat{\sigma}_{\text{SR}}} \right)$$

Where $\mathbb{E}[\max SR]$ is approximated using the Gumbel distribution and the Euler-Mascheroni constant ($\gamma$):

$$\mathbb{E}[\max SR] \approx \sigma_{\text{SR}} \left[ (1 - \gamma) Z^{-1}\left(1 - \frac{1}{N}\right) + \gamma Z^{-1}\left(1 - \frac{1}{N e}\right) \right]$$

## 3. Quickstart (5-Line Usage)

```python
import numpy as np
from dsr.pipeline import run_dsr_pipeline

# returns is a T x N numpy array (T periods, N trial strategies)
returns = np.random.normal(0, 0.01, size=(1000, 500)) 
result = run_dsr_pipeline(returns, confidence_threshold=0.95)

print(f"DSR Confidence: {result['dsr_confidence'] * 100:.2f}%")
print(f"Is Valid Strategy? {result['is_valid']}")