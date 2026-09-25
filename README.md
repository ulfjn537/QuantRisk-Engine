# JanusRisk v1.0 — Quantitative Risk & Portfolio Simulation Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Proprietary](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)]()
[![Framework: Quantitative Finance](https://img.shields.io/badge/Domain-Quantitative%20Finance-green.svg)]()

**JanusRisk v1.0** is an open-source quantitative risk management engine built in Python to model, simulate, and stress-test multi-asset investment portfolios under market uncertainty.

The framework integrates continuous-time stochastic calculus—specifically **Geometric Brownian Motion (GBM)** with **Itô’s Lemma** convexity adjustments—and **Multivariate Cholesky Factorization** to execute correlated Monte Carlo simulations. It delivers empirical and parametric estimations for downside tail-risk metrics ($\text{VaR}_{95\%}$ and $\text{CVaR}_{95\%}$) and risk-adjusted efficiency performance.

---

## Technical Architecture & Theoretical Framework

The engine is built around a modular architecture that enforces academic rigor and computational efficiency:
              ┌─────────────────────────────────┐
              │        data_engine.py           │
              │  (yfinance / Log-Returns / Σ, μ)│
              └────────────────┬────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │       portfolio_engine.py       │
              │ (Capital / Weights w / Volatility)│
              └────────────────┬────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │        monte_carlo.py           │
              │(Cholesky L / Correlated 3D Noise)│
              └────────────────┬────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │        risk_metrics.py          │
              │  (VaR_95 / CVaR_95 / Sharpe/STARR)│
              └────────────────┬────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │          visualizer.py          │
              │  (Matplotlib Projections & PDF) │
              └─────────────────────────────────┘

### Core Mathematical Foundations

1. **Stochastic Asset Dynamics (GBM & Itô's Lemma):**
   Asset trajectories are modeled via continuous Geometric Brownian Motion:
   $$dS_t = \mu S_t dt + \sigma S_t dW_t$$
   Applying Itô's Lemma to $f(S_t) = \ln S_t$ yields the exact discrete path generator with Jensen's convexity adjustment ($-\frac{1}{2}\sigma^2$):
   $$S_{t+\Delta t} = S_t \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t} \, Z_t \right)$$

2. **Multivariate Correlation via Cholesky Decomposition:**
   To capture cross-asset dependencies, the system decomposes the positive semi-definite covariance matrix $\boldsymbol{\Sigma}$:
   $$\boldsymbol{\Sigma} = \mathbf{L} \mathbf{L}^T$$
   *Note: A diagonal regularization term ($\mathbf{I} \cdot 10^{-8}$) is applied to guarantee numerical stability when encountering near-singular matrices.*

3. **Tail-Risk Metrics ($\alpha = 0.95$ Confidence Level):**
   - **Value-at-Risk ($\text{VaR}_{\alpha}$):** Empirical loss threshold at the $(1-\alpha)$-quantile of the simulated P&L distribution.
   - **Conditional Value-at-Risk ($\text{CVaR}_{\alpha}$):** Expected loss given that the loss exceeds $\text{VaR}_{\alpha}$ (Expected Shortfall):
     $$\text{CVaR}_{\alpha} = \mathbb{E}\left[ L \mid L \ge \text{VaR}_{\alpha} \right]$$

4. **Risk-Adjusted Efficiency:**
   Evaluates portfolio performance via the standard **Sharpe Ratio** ($\sigma_p$) and the tail-risk adjusted **STARR Ratio** ($\text{CVaR}_{95\%}$).

---

## Key Modules

- **`data_engine.py`**: Fetches adjusted closing prices, computes continuous log-returns, squeezes 1D vectors, and extracts mean return vectors $\boldsymbol{\mu}$ and covariance matrices $\boldsymbol{\Sigma}$.
- **`portfolio_engine.py`**: Manages the `Portfolio` class, calculates portfolio total value $V_{\text{total}} = \sum V_i$, normalized weights $\mathbf{w} = \mathbf{V}/V_{\text{total}}$, expected portfolio return $\mathbf{w}^T \boldsymbol{\mu}$, and variance $\sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}$.
- **`monte_carlo.py`**: Handles 3D tensor stochastic noise generation, applying lower triangular Cholesky matrices to project correlated multi-asset price paths over $M$ iterations.
- **`risk_metrics.py`**: Calculates empirical loss percentiles, dollar-denominated and percentage $\text{VaR}_{95\%}/\text{CVaR}_{95\%}$, loss probabilities, and Sharpe vs. STARR ratios.
- **`visualizer.py`**: Generates dual-panel graphics via Matplotlib showing time-series projections (up to 100 paths with global mean) alongside comparative loss-density histograms.
- **`cli.py`**: Dynamic interactive user interface to input real market tickers and capital allocations.

---

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/tu-usuario/janusrisk.git](https://github.com/tu-usuario/janusrisk.git)
   cd janusrisk
pip install -r requirements.txt
python cli.py
Methodological Limitations (Intellectual Honesty)JanusRisk v1.0 is a classic quantitative benchmark system. As documented in its technical specification:Lognormal Assumption: Assuming geometric Brownian motion excludes heavy tail behavior (fat tails) and jump processes.Sharpe/STARR Redundancy: Under Gaussian continuity, volatility $\sigma_p$ and $\text{CVaR}_{95\%}$ maintain a strict linear relationship, leading to identical portfolio rankings between Sharpe and STARR ratios. The STARR ratio serves as an architectural placeholder for future non-Gaussian extensions (e.g., Merton Jump-Diffusion).
Copyright (c) 2026 Francisco Luis Jiménez Navarro. All Rights Reserved.
This software, source code, and associated mathematical models are the proprietary intellectual property of Francisco Luis Jiménez Navarro. Unauthorized copying, redistribution, commercial exploitation, or publication of this code or documentation without prior written permission is strictly prohibited.
