"""
Risk Analytics Library
======================

Quantitative risk metrics for portfolio and trading applications.
Includes VaR, CVaR, LPM (Lower Partial Moments), and related measures.

Author: Gregory J. Komansky
Date: January 2026
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


# =============================================================================
# VALUE AT RISK (VaR)
# =============================================================================

def var_parametric(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1
) -> float:
    """
    Parametric VaR assuming normal distribution.

    VaR = μ - z × σ × √T

    Args:
        returns: Return series
        confidence: Confidence level (e.g., 0.95 for 95%)
        horizon_days: Time horizon in days

    Returns:
        VaR as positive number (loss)
    """
    mu = returns.mean() * horizon_days
    sigma = returns.std() * np.sqrt(horizon_days)
    z = stats.norm.ppf(1 - confidence)

    return -(mu + z * sigma)


def var_historical(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1
) -> float:
    """
    Historical VaR using empirical percentile.

    Args:
        returns: Return series
        confidence: Confidence level
        horizon_days: Time horizon (scales returns if > 1)

    Returns:
        VaR as positive number (loss)
    """
    if horizon_days > 1:
        # Scale returns by sqrt(T) approximation
        scaled_returns = returns * np.sqrt(horizon_days)
    else:
        scaled_returns = returns

    percentile = (1 - confidence) * 100
    return -np.percentile(scaled_returns.dropna(), percentile)


def var_cornish_fisher(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1
) -> float:
    """
    Cornish-Fisher VaR adjusted for skewness and kurtosis.

    Accounts for fat tails common in crypto returns.

    z_cf = z + (z²-1)×S/6 + (z³-3z)×(K-3)/24 - (2z³-5z)×S²/36

    Where:
        z = normal quantile
        S = skewness
        K = kurtosis
    """
    mu = returns.mean() * horizon_days
    sigma = returns.std() * np.sqrt(horizon_days)
    skew = returns.skew()
    kurt = returns.kurtosis()

    z = stats.norm.ppf(1 - confidence)

    # Cornish-Fisher expansion
    z_cf = (z +
            (z**2 - 1) * skew / 6 +
            (z**3 - 3*z) * (kurt - 3) / 24 -
            (2*z**3 - 5*z) * skew**2 / 36)

    return -(mu + z_cf * sigma)


def var_monte_carlo(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
    n_simulations: int = 10000
) -> float:
    """
    Monte Carlo VaR with bootstrapped returns.

    Args:
        returns: Return series
        confidence: Confidence level
        horizon_days: Time horizon
        n_simulations: Number of simulations

    Returns:
        VaR as positive number (loss)
    """
    returns_clean = returns.dropna().values

    # Bootstrap returns
    simulated_returns = np.random.choice(
        returns_clean,
        size=(n_simulations, horizon_days),
        replace=True
    )

    # Cumulative returns over horizon
    cum_returns = simulated_returns.sum(axis=1)

    percentile = (1 - confidence) * 100
    return -np.percentile(cum_returns, percentile)


# =============================================================================
# EXPECTED SHORTFALL (CVaR)
# =============================================================================

def cvar(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """
    Conditional VaR (Expected Shortfall).

    CVaR = E[Loss | Loss > VaR]

    The average loss in the worst (1-confidence)% of cases.

    Args:
        returns: Return series
        confidence: Confidence level

    Returns:
        CVaR as positive number (expected loss in tail)
    """
    var = var_historical(returns, confidence)
    tail_returns = returns[returns < -var]

    if len(tail_returns) == 0:
        return var

    return -tail_returns.mean()


# =============================================================================
# LOWER PARTIAL MOMENTS (LPM)
# =============================================================================

def lpm(
    returns: pd.Series,
    threshold: float = 0.0,
    order: int = 2
) -> float:
    """
    Lower Partial Moment of order n.

    LPM_n(τ) = E[max(τ - R, 0)^n]

    Orders:
        n=0: Probability of underperformance
        n=1: Expected shortfall below threshold
        n=2: Semi-variance (downside variance)
        n=3: Downside skewness component
        n=4: Downside kurtosis component

    Args:
        returns: Return series
        threshold: Target return (default 0)
        order: Moment order (default 2 for semi-variance)

    Returns:
        LPM value
    """
    shortfall = np.maximum(threshold - returns, 0)
    return (shortfall ** order).mean()


def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    threshold: float = 0.0,
    annualize: bool = True
) -> float:
    """
    Sortino Ratio using downside deviation.

    Sortino = (R - Rf) / Downside_Deviation

    Unlike Sharpe, only penalizes downside volatility.

    Args:
        returns: Return series
        risk_free: Risk-free rate (same frequency as returns)
        threshold: MAR for downside deviation
        annualize: Whether to annualize (assumes daily returns)

    Returns:
        Sortino ratio
    """
    excess_return = returns.mean() - risk_free

    # Downside deviation = sqrt(LPM_2)
    downside_dev = np.sqrt(lpm(returns, threshold, order=2))

    if downside_dev == 0:
        return np.inf if excess_return > 0 else 0

    ratio = excess_return / downside_dev

    if annualize:
        ratio *= np.sqrt(252)

    return ratio


def omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0
) -> float:
    """
    Omega Ratio: probability-weighted ratio of gains to losses.

    Omega = E[max(R - τ, 0)] / E[max(τ - R, 0)]
         = LPM_1 of gains / LPM_1 of losses

    Omega > 1 means expected gains exceed expected losses.

    Args:
        returns: Return series
        threshold: Target return

    Returns:
        Omega ratio
    """
    gains = np.maximum(returns - threshold, 0).mean()
    losses = np.maximum(threshold - returns, 0).mean()

    if losses == 0:
        return np.inf if gains > 0 else 1.0

    return gains / losses


# =============================================================================
# DRAWDOWN ANALYSIS
# =============================================================================

def calculate_drawdowns(returns: pd.Series) -> pd.DataFrame:
    """
    Calculate drawdown series and statistics.

    Args:
        returns: Return series

    Returns:
        DataFrame with cumulative returns and drawdowns
    """
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdown = (cum_returns - rolling_max) / rolling_max

    return pd.DataFrame({
        'cumulative': cum_returns,
        'peak': rolling_max,
        'drawdown': drawdown,
    })


def max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    dd = calculate_drawdowns(returns)
    return dd['drawdown'].min()


def calmar_ratio(
    returns: pd.Series,
    annualize: bool = True
) -> float:
    """
    Calmar Ratio: annualized return / max drawdown.

    Args:
        returns: Return series
        annualize: Whether to annualize returns

    Returns:
        Calmar ratio
    """
    if annualize:
        total_return = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        ann_return = (1 + total_return) ** (1 / n_years) - 1
    else:
        ann_return = returns.mean()

    mdd = abs(max_drawdown(returns))

    if mdd == 0:
        return np.inf if ann_return > 0 else 0

    return ann_return / mdd


# =============================================================================
# COMPREHENSIVE RISK REPORT
# =============================================================================

@dataclass
class RiskReport:
    """Comprehensive risk metrics report."""
    # Basic stats
    mean_return: float
    volatility: float
    skewness: float
    kurtosis: float

    # VaR metrics (95%)
    var_parametric: float
    var_historical: float
    var_cornish_fisher: float
    cvar: float

    # LPM metrics
    lpm_1: float  # Expected shortfall
    lpm_2: float  # Semi-variance
    downside_deviation: float

    # Ratios
    sharpe_ratio: float
    sortino_ratio: float
    omega_ratio: float
    calmar_ratio: float

    # Drawdown
    max_drawdown: float


def generate_risk_report(
    returns: pd.Series,
    risk_free: float = 0.0,
    confidence: float = 0.95
) -> RiskReport:
    """
    Generate comprehensive risk report.

    Args:
        returns: Return series
        risk_free: Risk-free rate (same frequency)
        confidence: VaR confidence level

    Returns:
        RiskReport dataclass
    """
    returns = returns.dropna()

    # Annualization factor
    ann_factor = np.sqrt(252)

    return RiskReport(
        # Basic stats
        mean_return=returns.mean() * 252,
        volatility=returns.std() * ann_factor,
        skewness=returns.skew(),
        kurtosis=returns.kurtosis(),

        # VaR (as positive numbers)
        var_parametric=var_parametric(returns, confidence),
        var_historical=var_historical(returns, confidence),
        var_cornish_fisher=var_cornish_fisher(returns, confidence),
        cvar=cvar(returns, confidence),

        # LPM
        lpm_1=lpm(returns, 0, 1),
        lpm_2=lpm(returns, 0, 2),
        downside_deviation=np.sqrt(lpm(returns, 0, 2)) * ann_factor,

        # Ratios
        sharpe_ratio=(returns.mean() - risk_free) / returns.std() * ann_factor,
        sortino_ratio=sortino_ratio(returns, risk_free),
        omega_ratio=omega_ratio(returns),
        calmar_ratio=calmar_ratio(returns),

        # Drawdown
        max_drawdown=max_drawdown(returns),
    )


def print_risk_report(report: RiskReport) -> None:
    """Pretty print a risk report."""
    print("=" * 60)
    print("RISK REPORT")
    print("=" * 60)

    print("\nReturn Statistics:")
    print(f"  Annualized Return:    {report.mean_return:>10.2%}")
    print(f"  Annualized Volatility:{report.volatility:>10.2%}")
    print(f"  Skewness:             {report.skewness:>10.2f}")
    print(f"  Excess Kurtosis:      {report.kurtosis:>10.2f}")

    print("\nValue at Risk (95%, 1-day):")
    print(f"  Parametric VaR:       {report.var_parametric:>10.2%}")
    print(f"  Historical VaR:       {report.var_historical:>10.2%}")
    print(f"  Cornish-Fisher VaR:   {report.var_cornish_fisher:>10.2%}")
    print(f"  CVaR (Exp Shortfall): {report.cvar:>10.2%}")

    print("\nDownside Risk (LPM):")
    print(f"  LPM(1) Exp Shortfall: {report.lpm_1:>10.4f}")
    print(f"  LPM(2) Semi-variance: {report.lpm_2:>10.6f}")
    print(f"  Downside Deviation:   {report.downside_deviation:>10.2%}")

    print("\nRisk-Adjusted Ratios:")
    print(f"  Sharpe Ratio:         {report.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:        {report.sortino_ratio:>10.2f}")
    print(f"  Omega Ratio:          {report.omega_ratio:>10.2f}")
    print(f"  Calmar Ratio:         {report.calmar_ratio:>10.2f}")

    print("\nDrawdown:")
    print(f"  Maximum Drawdown:     {report.max_drawdown:>10.2%}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Generate sample returns (crypto-like: high vol, fat tails)
    np.random.seed(42)
    n_days = 1000

    # Mix of normal and jump process for fat tails
    normal_returns = np.random.normal(0.0005, 0.025, n_days)
    jumps = np.random.choice([0, -0.10, 0.08], n_days, p=[0.98, 0.015, 0.005])
    returns = pd.Series(normal_returns + jumps)

    print("=" * 60)
    print("RISK ANALYTICS DEMONSTRATION")
    print("=" * 60)

    # Generate and print report
    report = generate_risk_report(returns)
    print_risk_report(report)

    # Compare VaR methods
    print("\n" + "=" * 60)
    print("VaR METHOD COMPARISON")
    print("=" * 60)
    print("""
Note: Cornish-Fisher VaR accounts for fat tails (kurtosis > 0).
For crypto assets with excess kurtosis, CF-VaR is typically higher
than parametric VaR, better capturing tail risk.

LPM-based measures (Sortino, downside deviation) focus only on
downside risk, ignoring upside volatility that benefits investors.
    """)

    # LPM demonstration
    print("=" * 60)
    print("LPM ORDER INTERPRETATION")
    print("=" * 60)
    for order in [0, 1, 2, 3]:
        lpm_val = lpm(returns, threshold=0, order=order)
        interpretations = {
            0: "Probability of loss",
            1: "Expected shortfall",
            2: "Semi-variance",
            3: "Downside skew component",
        }
        print(f"  LPM({order}): {lpm_val:.6f} - {interpretations[order]}")
