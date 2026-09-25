# JanusRisk v1.0 — Quantitative Risk & Portfolio Simulation Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Proprietary](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)]()
[![Framework: Quantitative Finance](https://img.shields.io/badge/Domain-Quantitative%20Finance-green.svg)]()

**JanusRisk v1.0** is an open-source quantitative risk management engine built in Python to model, simulate, and stress-test multi-asset investment portfolios under market uncertainty.

The framework integrates continuous-time stochastic calculus—specifically **Geometric Brownian Motion (GBM)** with **Itô’s Lemma** convexity adjustments—and **Multivariate Cholesky Factorization** to execute correlated Monte Carlo simulations. It delivers empirical and parametric estimations for downside tail-risk metrics ($\mathrm{VaR}_{95\%}$ and $\mathrm{CVaR}_{95\%}$) and risk-adjusted efficiency performance.

---

## Technical Architecture & Theoretical Framework

The engine is built around a modular architecture that enforces academic rigor and computational efficiency:

* 📊 **`data_engine.py`**: Fetches prices via `yfinance`, computes continuous log-returns, vector $\boldsymbol{\mu}$, and covariance matrix $\boldsymbol{\Sigma}$.
* 💼 **`portfolio_engine.py`**: Computes portfolio total capital, weights $\mathbf{w}$, expected return, and volatility.
* 🎲 **`monte_carlo.py`**: Generates correlated stochastic paths using Cholesky Factorization $\mathbf{L} \mathbf{L}^T = \boldsymbol{\Sigma}$.
* 📉 **`risk_metrics.py`**: Calculates empirical $\mathrm{VaR}_{95\%}$, $\mathrm{CVaR}_{95\%}$, Sharpe, and STARR ratios.
* 🖼️ **`visualizer.py`**: Renders simulation trajectory projections and loss distribution histograms via Matplotlib.

---

### Core Mathematical Foundations

1. **Stochastic Asset Dynamics (GBM & Itô's Lemma):**
   Asset trajectories are modeled via continuous Geometric Brownian Motion:
   
   $$dS_t = \mu S_t dt + \sigma S_t dW_t$$
   
   Applying Itô's Lemma to $f(S_t) = \ln S_t$ yields the exact discrete path generator with Jensen's convexity adjustment:
   
   $$S_{t+\Delta t} = S_t \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t} \, Z_t \right)$$

2. **Multivariate Correlation via Cholesky Decomposition:**
   To capture cross-asset dependencies, the system decomposes the positive semi-definite covariance matrix $\boldsymbol{\Sigma}$:
   
   $$\boldsymbol{\Sigma} = \mathbf{L} \mathbf{L}^T$$

3. **Tail-Risk Metrics ($\alpha = 0.95$ Confidence Level):**
   * **Value-at-Risk ($\mathrm{VaR}_{\alpha}$):** Empirical loss threshold at the $(1-\alpha)$-quantile of the simulated P&L distribution.
   * **Conditional Value-at-Risk ($\mathrm{CVaR}_{\alpha}$):** Expected loss given that the loss exceeds $\mathrm{VaR}_{\alpha}$:
   
     $$\mathrm{CVaR}_{\alpha} = \mathbb{E}\left[ L \mid L \ge \mathrm{VaR}_{\alpha} \right]$$

4. **Risk-Adjusted Efficiency:**
   Evaluates portfolio performance via the standard **Sharpe Ratio** ($\sigma_p$) and the tail-risk adjusted **STARR Ratio** ($\mathrm{CVaR}_{95\%}$).

---

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/tu-usuario/janusrisk.git](https://github.com/tu-usuario/janusrisk.git)
   cd janusrisk
