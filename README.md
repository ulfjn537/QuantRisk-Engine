# JanusRisk v1.0 — Quantitative Risk & Portfolio Simulation Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Proprietary](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)]()
[![Domain: Quantitative Finance](https://img.shields.io/badge/Domain-Quantitative%20Finance-green.svg)]()

**JanusRisk v1.0** is an open-source quantitative risk management platform built in Python. It is designed to analyze multi-asset investment portfolios through Monte Carlo simulations and Geometric Brownian Motion (GBM). The system integrates a portfolio comparator and an Out-of-Sample backtesting module to empirically evaluate risk metrics under extreme market stress.

---

## System Architecture & Modules

The platform is built on a strictly decoupled Object-Oriented structure, ensuring computational efficiency through NumPy's vectorized operations:

* 📊 **`data_engine.py`**: Ingests adjusted closing prices via `yfinance`, extracting log-returns, mean return vectors ($\mu$), and the covariance matrix ($\Sigma$).
* 💼 **`portfolio_engine.py`**: Manages the portfolio's total capital, normalized weight vectors ($w$), expected returns, and global volatility.
* 🎲 **`monte_carlo.py`**: Generates 3D tensors of correlated stochastic price paths applying Cholesky decomposition ($L L^T = \Sigma$) without iterative loops.
* 📉 **`risk_metrics.py`**: Computes standard Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR) alongside risk-adjusted efficiency ratios (Sharpe and STARR).
* 🧪 **`backtesting.py`**: Executes Out-of-Sample stress tests, comparing In-Sample theoretical VaR against the real Maximum Drawdown (MDD) observed during crises.
* 🖼️ **`visualizer.py` & `cli.py`**: Command-line interface and Matplotlib projection panels for interactive portfolio comparison.

---

## Core Mathematical Framework

1. **Stochastic Asset Dynamics (GBM & Itô's Lemma):**
   Asset trajectories are modeled via continuous-time Geometric Brownian Motion:
   
   $$dS_t = \mu S_t dt + \sigma S_t dW_t$$
   
   Applying Itô's Lemma to the log-price yields the exact discrete path generator, which includes Jensen's convexity adjustment:
   
   $$S_{t+\Delta t} = S_t \exp\left( \left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t} \, Z_t \right)$$

2. **Multivariate Correlation via Cholesky Factorization:**
   To inject real market dependencies into the uncorrelated standard normal variables ($Z \sim \mathcal{N}(0, I)$), the covariance matrix is factorized:
   
   $$\Sigma = \mathbf{L} \mathbf{L}^T$$

3. **Risk Metrics (VaR & CVaR):**
   * **Value-at-Risk ($VaR_{\alpha}$):** The empirical loss threshold at the $(1-\alpha)$-quantile of the simulated P&L distribution.
   * **Conditional Value-at-Risk ($CVaR_{\alpha}$):** The expected shortfall, mathematically defined as the expected loss given that the loss exceeds the VaR threshold.

---

## Empirical Validation: COVID-19 Stress Test

The engine includes a rigorous Out-of-Sample validation against real black swan events. 
During the COVID-19 market crash (Feb-Apr 2020), the system demonstrated that a technology-weighted portfolio with a strict theoretical limit of **$VaR_{99\%} = 22.75\%$** suffered a real **Maximum Drawdown of 29.72%**. This empirical failure confirms that Gaussian projections calibrated during stable periods cannot contain the magnitude of losses during financial panic.

---

## Methodological Limitations (Intellectual Honesty)

As concluded in the technical memory, this model serves as a quantitative baseline but is bound by fundamental limitations regarding the true nature of financial risk:

* **The Gaussian Delusion & Asymmetry:** The assumption of log-normal returns ignores heavy tails (kurtosis). By enforcing a bell-curve geometry, the model fails to capture structural differences in risk elasticity between distinct asset classes.
* **Sharpe vs. STARR Redundancy:** Under the continuous Gaussian framework, volatility ($\sigma$) and CVaR maintain a strict linear relationship. Consequently, Sharpe and STARR ratios yield identical portfolio rankings, rendering the STARR ratio redundant unless non-Gaussian jump-diffusion processes are introduced.
* **The Lucas Critique & Parameter Instability:** The engine assumes historical parameters ($\mu$ and $\Sigma$) remain stationary. In reality, during market panics, drift collapses and cross-asset correlations approach 1, destroying the calculated diversification exactly when it is most needed.
* **Zero Friction (Infinite Liquidity):** The model ignores transaction costs and slippage (bid-ask spread widening) which drastically exacerbate real losses during liquidity crises.

---

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/tu-usuario/janusrisk.git](https://github.com/tu-usuario/janusrisk.git)
   cd janusrisk