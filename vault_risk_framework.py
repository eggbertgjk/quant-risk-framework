"""
DeFi Vault Risk Decomposition Framework
=======================================

Decomposes any DeFi vault's risk into 4 atomic primitives and calculates
Expected Annual Loss (EAL) in basis points.

Based on: "A Graph-Theoretic Framework for DeFi Vault Risk Decomposition"
SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6116826

Calibrated on 449 exploits ($15.7B losses), 2016-2026.

Author: Gregory J. Komansky
Date: January 2026
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional


# =============================================================================
# CALIBRATED PARAMETERS (from 449 exploits)
# =============================================================================

PRIMITIVE_PARAMS = {
    "CONTRACT": {
        "lambda": 0.0611,      # Annual probability of exploit
        "severity": 0.51,      # % of TVL lost when exploit occurs
        "incidents_pct": 0.65, # 65% of all incidents
        "losses_pct": 0.45,    # 45% of all losses
        "avg_loss_m": 24.2,    # Average loss per incident ($M)
    },
    "ORACLE": {
        "lambda": 0.0128,
        "severity": 0.51,
        "incidents_pct": 0.14,
        "losses_pct": 0.04,
        "avg_loss_m": 10.0,
    },
    "GOVERNANCE": {
        "lambda": 0.0006,
        "severity": 0.58,
        "incidents_pct": 0.01,
        "losses_pct": 0.01,
        "avg_loss_m": 40.0,
    },
    "OPERATIONAL": {
        "lambda": 0.0193,
        "severity": 0.60,
        "incidents_pct": 0.20,
        "losses_pct": 0.50,
        "avg_loss_m": 87.2,    # 3.6x more severe than CONTRACT
    },
}


def get_age_discount(age_months: int) -> float:
    """
    Calculate age discount factor based on protocol maturity.

    Older protocols have lower exploit rates (battle-tested).

    Args:
        age_months: Protocol age in months

    Returns:
        Discount factor (0.8 to 1.0)
    """
    if age_months < 6:
        return 1.00
    elif age_months < 12:
        return 0.95
    elif age_months < 24:
        return 0.90
    elif age_months < 36:
        return 0.85
    else:
        return 0.80


@dataclass
class VaultRiskProfile:
    """Risk profile for a DeFi vault."""
    name: str
    protocol: str
    underlying: str
    tvl_m: float
    apy: float
    age_months: int
    weights: Dict[str, float]  # Primitive weights (must sum to ~1.0)

    def __post_init__(self):
        # Validate weights sum to approximately 1.0
        total = sum(self.weights.values())
        if not 0.95 <= total <= 1.05:
            raise ValueError(f"Weights must sum to ~1.0, got {total}")


@dataclass
class RiskDecomposition:
    """Complete risk decomposition for a vault."""
    vault: VaultRiskProfile
    eal_by_primitive: Dict[str, float]  # EAL in bps per primitive
    eal_total_bps: float                # Total EAL in bps
    pct_by_primitive: Dict[str, float]  # % contribution per primitive
    ray: float                          # Risk-Adjusted Yield
    age_discount: float


def calculate_eal(vault: VaultRiskProfile) -> RiskDecomposition:
    """
    Calculate Expected Annual Loss (EAL) for a vault.

    EAL = Σ (λᵢ × Sᵢ × wᵢ) × age_discount × 10,000 bps

    Where:
        λᵢ = Annual probability of exploit (primitive i)
        Sᵢ = Severity (% of TVL lost when exploit occurs)
        wᵢ = Weight (vault's exposure to primitive i)

    Args:
        vault: VaultRiskProfile with weights and metadata

    Returns:
        RiskDecomposition with full breakdown
    """
    age_discount = get_age_discount(vault.age_months)

    eal_by_primitive = {}
    for primitive, weight in vault.weights.items():
        params = PRIMITIVE_PARAMS[primitive]
        # EAL = lambda × severity × weight × discount × 10000 (to get bps)
        eal = params["lambda"] * params["severity"] * weight * age_discount * 10000
        eal_by_primitive[primitive] = round(eal, 2)

    eal_total = sum(eal_by_primitive.values())

    # Calculate percentage contribution
    pct_by_primitive = {
        p: round(eal / eal_total * 100, 1) if eal_total > 0 else 0
        for p, eal in eal_by_primitive.items()
    }

    # Risk-Adjusted Yield: APY / (EAL as percentage)
    # Higher RAY = more yield per unit of risk
    ray = round(vault.apy / (eal_total / 100), 2) if eal_total > 0 else float('inf')

    return RiskDecomposition(
        vault=vault,
        eal_by_primitive=eal_by_primitive,
        eal_total_bps=round(eal_total, 2),
        pct_by_primitive=pct_by_primitive,
        ray=ray,
        age_discount=age_discount,
    )


def calculate_fair_premium(eal_bps: float, loading: float = 0.30) -> float:
    """
    Calculate actuarially fair insurance premium.

    Fair Premium = EAL × (1 + loading)

    Args:
        eal_bps: Expected Annual Loss in basis points
        loading: Loading factor for admin, capital, profit (default 30%)

    Returns:
        Fair premium in basis points
    """
    return round(eal_bps * (1 + loading), 2)


def compare_vaults(vaults: list[VaultRiskProfile]) -> pd.DataFrame:
    """
    Compare multiple vaults on risk-adjusted basis.

    Args:
        vaults: List of VaultRiskProfile objects

    Returns:
        DataFrame with comparison metrics, sorted by RAY (descending)
    """
    results = []
    for vault in vaults:
        decomp = calculate_eal(vault)
        results.append({
            "name": vault.name,
            "protocol": vault.protocol,
            "underlying": vault.underlying,
            "tvl_m": vault.tvl_m,
            "apy": vault.apy,
            "eal_bps": decomp.eal_total_bps,
            "ray": decomp.ray,
            "top_primitive": max(decomp.pct_by_primitive, key=decomp.pct_by_primitive.get),
            "top_pct": max(decomp.pct_by_primitive.values()),
        })

    df = pd.DataFrame(results)
    return df.sort_values("ray", ascending=False).reset_index(drop=True)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Define sample vaults
    vaults = [
        VaultRiskProfile(
            name="Lending Pool A USDC",
            protocol="LendingA",
            underlying="USDC",
            tvl_m=2100,
            apy=4.2,
            age_months=36,
            weights={"CONTRACT": 0.55, "ORACLE": 0.25, "GOVERNANCE": 0.08, "OPERATIONAL": 0.08},
        ),
        VaultRiskProfile(
            name="Optimizer USDC",
            protocol="Optimizer",
            underlying="USDC",
            tvl_m=150,
            apy=6.2,
            age_months=8,
            weights={"CONTRACT": 0.62, "ORACLE": 0.20, "GOVERNANCE": 0.05, "OPERATIONAL": 0.12},
        ),
        VaultRiskProfile(
            name="Staking Protocol",
            protocol="StakingX",
            underlying="ETH",
            tvl_m=25000,
            apy=4.5,
            age_months=40,
            weights={"CONTRACT": 0.50, "ORACLE": 0.15, "GOVERNANCE": 0.05, "OPERATIONAL": 0.30},
        ),
    ]

    print("=" * 80)
    print("VAULT RISK DECOMPOSITION FRAMEWORK")
    print("=" * 80)

    # Analyze each vault
    for vault in vaults:
        decomp = calculate_eal(vault)
        print(f"\n{vault.name}")
        print("-" * 40)
        print(f"APY: {vault.apy}%")
        print(f"Age: {vault.age_months} months (discount: {decomp.age_discount})")
        print(f"\nEAL by Primitive:")
        for p, eal in decomp.eal_by_primitive.items():
            pct = decomp.pct_by_primitive[p]
            print(f"  {p:<12} {eal:>6.1f} bps ({pct:>4.1f}%)")
        print(f"\nTotal EAL: {decomp.eal_total_bps} bps")
        print(f"RAY (Risk-Adjusted Yield): {decomp.ray}")
        print(f"Fair Premium (30% loading): {calculate_fair_premium(decomp.eal_total_bps)} bps")

    # Comparison table
    print("\n" + "=" * 80)
    print("VAULT COMPARISON (sorted by RAY)")
    print("=" * 80)
    comparison = compare_vaults(vaults)
    print(comparison.to_string(index=False))

    # Key insight
    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print("""
OPERATIONAL exploits are 20% of incidents but 50% of losses.
Average severity: $87M (OPERATIONAL) vs $24M (CONTRACT) = 3.6x difference.

Markets price by frequency. The data says price by severity.
That's the mispricing this framework identifies.
    """)
