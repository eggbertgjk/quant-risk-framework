# Quantitative Risk & Trading Frameworks

**Author:** Gregory J. Komansky
**Background:** 25 years institutional finance (Citi, JPMorgan, ClearBridge) | Creator of APTO ($112B traded)

---

## Overview

This repository contains implementations of quantitative frameworks for:
1. **DeFi Vault Risk Decomposition** - Atomic risk primitives with calibrated parameters
2. **BTC Regime-Switching Model** - 7-layer position sizing with validated alpha signals
3. **Risk Analytics Library** - VaR, CVaR, LPM, and risk-adjusted performance metrics

All frameworks emphasize mathematical rigor, empirical calibration, and practical applicability.

---

## 1. DeFi Vault Risk Framework

**File:** `vault_risk_framework.py`

**Published Research:** [SSRN Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6116826)

Decomposes any DeFi vault's risk into 4 atomic primitives and calculates Expected Annual Loss (EAL) in basis points.

### Four Atomic Primitives

| Primitive | Description | Incidents | Losses | Avg Severity |
|-----------|-------------|-----------|--------|--------------|
| **CONTRACT** | Smart contract bugs, reentrancy | 65% | 45% | $24M |
| **ORACLE** | Price feed manipulation | 14% | 4% | $10M |
| **GOVERNANCE** | Vote attacks, malicious proposals | 1% | 1% | $40M |
| **OPERATIONAL** | Key compromise, infrastructure | 20% | 50% | $87M |

### Key Finding

**OPERATIONAL exploits are 20% of incidents but 50% of losses.**

Average severity: $87M (OPERATIONAL) vs $24M (CONTRACT) = 3.6x difference.

Markets price by frequency. The data says price by severity.

### Usage

```python
from vault_risk_framework import VaultRiskProfile, calculate_eal

vault = VaultRiskProfile(
    name="Example Vault",
    protocol="ExampleProtocol",
    underlying="USDC",
    tvl_m=500,
    apy=5.5,
    age_months=24,
    weights={
        "CONTRACT": 0.55,
        "ORACLE": 0.25,
        "GOVERNANCE": 0.08,
        "OPERATIONAL": 0.12
    }
)

decomp = calculate_eal(vault)
print(f"Total EAL: {decomp.eal_total_bps} bps")
print(f"Risk-Adjusted Yield: {decomp.ray}")
```

### Calibration

- **449 exploits** analyzed (2016-2026)
- **$15.7B** total losses
- Validation: ρ = 0.748 (p = 0.003)

---

## 2. BTC Regime-Switching Volatility Model

**File:** `regime_model.py`

A 7-layer regime detection model for BTC position sizing, combining trend filtering, volatility regimes, and skewness signals.

### Model Architecture

| Layer | Component | Function |
|-------|-----------|----------|
| 1 | Trend Filter | EMA(50) ratio for trend direction |
| 2 | Regime Detection | Bull (1.15x) / Range (0.85x) / Bear (0.55x) |
| 3-4 | Volatility Factors | Short-term (14d) vs expanding long-term vol |
| 5 | Skew Adjustment | Short-term minus long-term skewness |
| 6 | Composite Factor | Log-linear combination of vol + skew |
| 7 | Position Sizing | Regime base × vol adjustment, clipped |

### Primary Alpha Signal

**Skew Differential (short-term minus long-term skewness)**

| Horizon | IC | t-stat |
|---------|-----|--------|
| 1-day | +0.14 | 6.4*** |
| 21-day | **+0.27** | 12.1*** |

### Validation

Monte Carlo robustness (1,000 simulations):
- P(Sharpe > 0.5) = **92.3%**
- P(Sharpe > 1.0) = 46.7%

### Usage

```python
from regime_model import run_regime_model, backtest_strategy
import pandas as pd

# prices = pd.Series of BTC daily closes
results = run_regime_model(prices)

# Backtest
bt = backtest_strategy(prices, results['position'])
```

---

## 3. Risk Analytics Library

**File:** `risk_analytics.py`

Comprehensive risk metrics for portfolio and trading applications.

### Value at Risk (VaR)

| Method | Function | Use Case |
|--------|----------|----------|
| Parametric | `var_parametric()` | Quick estimate, normal assumption |
| Historical | `var_historical()` | No distribution assumption |
| Cornish-Fisher | `var_cornish_fisher()` | Accounts for skew/kurtosis |
| Monte Carlo | `var_monte_carlo()` | Complex portfolios |

### Lower Partial Moments (LPM)

Focus on downside risk only:

```
LPM_n(τ) = E[max(τ - R, 0)^n]

n=0: Probability of underperformance
n=1: Expected shortfall below threshold
n=2: Semi-variance (downside variance)
n=3: Downside skewness component
```

### Risk-Adjusted Ratios

| Ratio | Formula | Advantage |
|-------|---------|-----------|
| Sharpe | (R - Rf) / σ | Standard benchmark |
| Sortino | (R - Rf) / σ_down | Ignores upside vol |
| Omega | E[gains] / E[losses] | Full distribution |
| Calmar | Ann Return / Max DD | Drawdown-focused |

### Usage

```python
from risk_analytics import generate_risk_report, print_risk_report

# returns = pd.Series of daily returns
report = generate_risk_report(returns, risk_free=0.0, confidence=0.95)
print_risk_report(report)
```

---

## Mathematical Foundation

### EAL (Expected Annual Loss)

```
EAL = Σ (λᵢ × Sᵢ × wᵢ) × age_discount × 10,000 bps

Where:
  λᵢ = Annual probability of exploit (primitive i)
  Sᵢ = Severity (% of TVL lost when exploit occurs)
  wᵢ = Weight (vault's exposure to primitive i)
```

### Risk-Adjusted Yield (RAY)

```
RAY = APY / (EAL / 100)

Higher RAY = more yield per unit of risk
```

### Cornish-Fisher VaR Expansion

```
z_cf = z + (z²-1)×S/6 + (z³-3z)×(K-3)/24 - (2z³-5z)×S²/36

Where:
  z = normal quantile
  S = skewness
  K = kurtosis
```

---

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
```

---

## Contact

**Gregory J. Komansky**
Email: gjkomansky@gmail.com
LinkedIn: [Profile](https://www.linkedin.com/in/gregory-john-komansky-936173273/)
SSRN: [Research](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6116826)

---

*These frameworks represent independent research applying institutional-grade quantitative methods to digital asset risk and trading.*
