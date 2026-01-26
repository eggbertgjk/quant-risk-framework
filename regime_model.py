"""
Regime-Switching Volatility Model Template
==========================================

A template demonstrating regime-based position sizing methodology.
Combines trend filtering, volatility regimes, and higher-moment signals.

This is an EDUCATIONAL TEMPLATE showing architecture and methodology.
Actual parameters require calibration to specific assets and timeframes.

Author: Gregory J. Komansky
Date: January 2026
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class Regime(Enum):
    """Market regime classification."""
    BULL = "bull"
    RANGE = "range"
    BEAR = "bear"


@dataclass
class RegimeConfig:
    """
    Configuration for regime model.

    NOTE: These are PLACEHOLDER values for demonstration.
    Production parameters require:
    - Walk-forward optimization
    - Out-of-sample validation
    - Transaction cost analysis
    - Regime stability testing
    """
    # Trend filter (calibrate to asset volatility)
    ema_span: int = 50  # Typical range: 20-100

    # Regime position multipliers (calibrate to risk tolerance)
    bull_position: float = 1.0   # Placeholder
    range_position: float = 0.75  # Placeholder
    bear_position: float = 0.5   # Placeholder

    # Volatility windows (calibrate to mean-reversion speed)
    vol_short_window: int = 14    # Typical range: 7-21
    vol_long_min_periods: int = 90  # Typical range: 60-252

    # Higher moment parameters (calibrate to asset distribution)
    moment_short_window: int = 7   # Typical range: 5-14
    moment_long_window: int = 30   # Typical range: 21-63
    moment_multiplier: float = 0.5  # Placeholder - requires optimization

    # Position bounds (calibrate to risk limits)
    position_min: float = 0.25
    position_max: float = 1.5


@dataclass
class RegimeSignals:
    """Output signals from regime model."""
    regime: Regime
    trend_ratio: float
    vol_factor: float
    moment_signal: float
    final_position: float


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns from price series."""
    return np.log(prices / prices.shift(1))


# =============================================================================
# LAYER 1: TREND FILTER
# =============================================================================

def trend_filter(prices: pd.Series, config: RegimeConfig) -> pd.Series:
    """
    Trend filter using exponential moving average.

    Returns ratio of price to EMA:
    - ratio > 1: price above trend (bullish)
    - ratio < 1: price below trend (bearish)

    The EMA span should be calibrated to:
    - Asset volatility (higher vol = longer span)
    - Trading frequency (longer horizon = longer span)
    """
    ema = prices.ewm(span=config.ema_span, adjust=False).mean()
    return prices / ema


# =============================================================================
# LAYER 2: REGIME CLASSIFICATION
# =============================================================================

def classify_regime(
    trend_ratio: pd.Series,
    bull_threshold: float = 1.02,
    bear_threshold: float = 0.98
) -> pd.Series:
    """
    Classify market regime based on trend ratio.

    Thresholds should be calibrated to:
    - Minimize whipsaws (too tight = excess trading)
    - Capture regime shifts (too wide = slow reaction)

    Args:
        trend_ratio: Price / EMA ratio
        bull_threshold: Ratio above which = bull regime
        bear_threshold: Ratio below which = bear regime
    """
    def classify(ratio):
        if pd.isna(ratio):
            return None
        elif ratio > bull_threshold:
            return Regime.BULL
        elif ratio < bear_threshold:
            return Regime.BEAR
        else:
            return Regime.RANGE

    return trend_ratio.apply(classify)


# =============================================================================
# LAYER 3: VOLATILITY FACTOR
# =============================================================================

def volatility_factor(returns: pd.Series, config: RegimeConfig) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate volatility regime factor.

    Compares short-term realized vol to long-term baseline.

    vol_factor = long_term_vol / short_term_vol

    Interpretation:
    - vol_factor > 1: Current vol below average (calming)
    - vol_factor < 1: Current vol above average (stressed)

    Use expanding window for long-term to avoid lookahead bias.
    """
    vol_short = returns.rolling(config.vol_short_window).std()
    vol_long = returns.expanding(min_periods=config.vol_long_min_periods).std()

    vol_factor = vol_long / vol_short
    return vol_short, vol_factor


# =============================================================================
# LAYER 4: HIGHER MOMENT SIGNAL
# =============================================================================

def moment_signal(returns: pd.Series, config: RegimeConfig) -> pd.Series:
    """
    Higher moment signal comparing short-term to long-term distribution shape.

    METHODOLOGY (not specific implementation):
    - Compare recent return distribution to historical
    - Positive signal: recent distribution more favorable
    - Negative signal: recent distribution shows stress

    Possible approaches:
    - Skewness differential
    - Kurtosis changes
    - Tail ratio comparisons
    - Distribution distance metrics

    The specific moment and transformation requires calibration
    to the asset being traded.
    """
    # PLACEHOLDER: Simple skewness differential
    # Production implementation requires optimization
    moment_short = returns.rolling(config.moment_short_window).skew()
    moment_long = returns.rolling(config.moment_long_window).skew()

    return moment_short - moment_long


# =============================================================================
# LAYER 5: POSITION SIZING
# =============================================================================

def calculate_position(
    regime: Regime,
    vol_factor: float,
    moment_signal: float,
    config: RegimeConfig
) -> float:
    """
    Calculate final position size.

    Architecture:
    1. Start with regime-based position
    2. Adjust for volatility regime
    3. Adjust for moment signal
    4. Clip to position limits

    The combination function (multiplicative, additive, conditional)
    requires calibration.
    """
    # Base position by regime
    base_positions = {
        Regime.BULL: config.bull_position,
        Regime.RANGE: config.range_position,
        Regime.BEAR: config.bear_position,
    }
    base = base_positions.get(regime, config.range_position)

    # PLACEHOLDER: Simple multiplicative adjustment
    # Production requires optimized combination function
    if pd.isna(vol_factor) or pd.isna(moment_signal):
        return base

    # Normalize vol_factor (clip extremes)
    vol_adj = np.clip(vol_factor, 0.5, 1.5)

    # Moment adjustment
    moment_adj = 1.0 + np.clip(
        moment_signal * config.moment_multiplier,
        -0.3, 0.3
    )

    # Combine and clip
    position = base * vol_adj * moment_adj
    return np.clip(position, config.position_min, config.position_max)


# =============================================================================
# FULL MODEL PIPELINE
# =============================================================================

def run_regime_model(
    prices: pd.Series,
    config: Optional[RegimeConfig] = None
) -> pd.DataFrame:
    """
    Run regime model on price series.

    Args:
        prices: Price series (e.g., daily close)
        config: Model configuration

    Returns:
        DataFrame with signals and positions
    """
    if config is None:
        config = RegimeConfig()

    returns = calculate_returns(prices)

    # Layer 1: Trend
    trend_ratio = trend_filter(prices, config)

    # Layer 2: Regime
    regimes = classify_regime(trend_ratio)

    # Layer 3: Volatility
    vol_short, vol_fact = volatility_factor(returns, config)

    # Layer 4: Moment signal
    moment_sig = moment_signal(returns, config)

    # Combine into DataFrame
    df = pd.DataFrame({
        'price': prices,
        'returns': returns,
        'trend_ratio': trend_ratio,
        'regime': regimes,
        'vol_short': vol_short,
        'vol_factor': vol_fact,
        'moment_signal': moment_sig,
    })

    # Layer 5: Position sizing
    def get_position(row):
        if pd.isna(row['regime']) or pd.isna(row['vol_factor']):
            return np.nan
        return calculate_position(
            row['regime'],
            row['vol_factor'],
            row['moment_signal'],
            config
        )

    df['position'] = df.apply(get_position, axis=1)

    return df


# =============================================================================
# SIGNAL EVALUATION
# =============================================================================

def calculate_ic(
    signal: pd.Series,
    forward_returns: pd.Series,
    horizon: int = 21
) -> float:
    """
    Calculate Information Coefficient (IC).

    IC = Spearman correlation between signal and forward returns.

    Interpretation:
    - IC > 0.05: Potentially useful signal
    - IC > 0.10: Good signal
    - IC > 0.20: Excellent signal (rare)

    IMPORTANT: Always test statistical significance (t-stat > 2).
    """
    fwd = forward_returns.shift(-horizon)
    valid = ~(signal.isna() | fwd.isna())

    if valid.sum() < 30:
        return np.nan

    return signal[valid].corr(fwd[valid], method='spearman')


def walk_forward_backtest(
    prices: pd.Series,
    config: RegimeConfig,
    train_window: int = 252,
    test_window: int = 63
) -> pd.DataFrame:
    """
    Walk-forward backtest framework.

    CRITICAL for avoiding overfitting:
    1. Train on historical window
    2. Test on out-of-sample period
    3. Roll forward and repeat

    This template shows structure; production requires:
    - Parameter optimization in training window
    - Proper train/test separation
    - Transaction cost modeling
    """
    results = []

    for start in range(0, len(prices) - train_window - test_window, test_window):
        train_end = start + train_window
        test_end = train_end + test_window

        # In production: optimize config on training data
        # Here we use fixed config for demonstration

        test_prices = prices.iloc[train_end:test_end]
        test_results = run_regime_model(test_prices, config)

        results.append(test_results)

    return pd.concat(results) if results else pd.DataFrame()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_days = 500

    # Simulated price series
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = pd.Series(100 * np.exp(np.cumsum(returns)))
    prices.index = pd.date_range('2023-01-01', periods=n_days, freq='D')

    print("=" * 70)
    print("REGIME MODEL TEMPLATE - DEMONSTRATION")
    print("=" * 70)
    print("""
NOTE: This is an educational template showing methodology.
Parameters are PLACEHOLDERS requiring calibration for production use.

Calibration process:
1. Define target metric (Sharpe, Sortino, etc.)
2. Walk-forward optimization on training data
3. Out-of-sample validation
4. Monte Carlo robustness testing
5. Transaction cost sensitivity analysis
    """)

    # Run model with placeholder config
    config = RegimeConfig()
    results = run_regime_model(prices, config)

    print("\nSample Output (last 5 days):")
    cols = ['regime', 'vol_factor', 'moment_signal', 'position']
    print(results[cols].tail().to_string())

    # Regime distribution
    print("\nRegime Distribution:")
    regime_counts = results['regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(results) * 100
        print(f"  {regime.value:<6}: {count:>4} days ({pct:.1f}%)")

    # IC calculation example
    print("\nSignal IC (placeholder signal):")
    for horizon in [1, 5, 21]:
        ic = calculate_ic(results['moment_signal'], results['returns'], horizon)
        print(f"  {horizon}-day IC: {ic:.3f}")

    print("\n" + "=" * 70)
    print("CALIBRATION REQUIRED")
    print("=" * 70)
    print("""
To use this template in production:

1. PARAMETER OPTIMIZATION
   - EMA span: test range [20, 100]
   - Regime thresholds: test sensitivity
   - Position multipliers: align with risk budget
   - Moment windows: test multiple horizons

2. VALIDATION
   - Walk-forward testing (avoid lookahead)
   - Out-of-sample Sharpe ratio
   - Drawdown analysis
   - Turnover and transaction costs

3. ROBUSTNESS
   - Parameter stability across regimes
   - Performance in crisis periods
   - Monte Carlo with shuffled data
    """)


