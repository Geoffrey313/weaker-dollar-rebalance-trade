"""Two-country DCP rebalancing block (Phase 2, baseline reduced form).

This is the calibrated reduced form of the framework's mechanism: how much a bilateral real
depreciation improves the US-China trade balance, as a function of the two frictions
(dollar-price stickiness theta_dollar and the capital-controls wedge chi). It delivers
Conclusions 1 and 2 (does an engineered depreciation rebalance? and the switching threshold).
The full dynamic Ramsey GE solve (endogenous exchange-rate path, intertemporal margins) is the
next increment; this block isolates the impact mechanism transparently.

Derivation (log-linear, share-of-trade units). The semi-elasticity of the bilateral trade
balance to a real depreciation of the RMB is

    R(theta_dollar, chi) = Phi(chi) * (1 - theta_dollar) * (eta + eta_star - 1) * gamma

  * (eta + eta_star - 1): the Marshall-Lerner sum of import-demand elasticities. Expenditure
    switching needs relative prices to move.
  * (1 - theta_dollar): exchange-rate pass-through into border prices. Under dominant-currency
    pricing (dollar invoicing), theta_dollar -> 1 and pass-through -> 0, so a depreciation does
    NOT move the dollar border price and expenditure switching is shut down (H1).
  * Phi(chi) = 1/(1+chi) in (0,1]: capital-controls damping. A closed capital account (chi large)
    chokes the financing counterpart of the adjustment, so the equilibrium depreciation delivers
    less rebalancing.
  * gamma: openness scaling (import share).

Rebalancing is FEASIBLE if the depreciation needed to close the observed imbalance,
|de| = imbalance0 / R, stays within the bounded-cost maximum max_depreciation. Equivalently,
R must exceed R_min = imbalance0 / max_depreciation. The (theta_dollar, chi) locus where
R = R_min is the switching frontier (Conclusion 2).
"""
from __future__ import annotations

import numpy as np

from src.engine.calibration import Params, BASELINE


def capital_damping(p: Params = BASELINE) -> float:
    """Phi(chi) = 1/(1+chi): 1 at an open account (chi=0), -> 0 as the account closes."""
    return 1.0 / (1.0 + p.chi)


def rebalancing_power(p: Params = BASELINE) -> float:
    """R = d(NX/trade)/d(real depreciation): the exchange rate's rebalancing power."""
    return capital_damping(p) * (1.0 - p.theta_dollar) * (p.eta + p.eta_star - 1.0) * p.gamma


def required_depreciation(p: Params = BASELINE) -> float:
    """Real depreciation needed to close the imbalance (inf if the FX channel is dead)."""
    R = rebalancing_power(p)
    return np.inf if R <= 0 else p.imbalance0 / R


def is_feasible(p: Params = BASELINE) -> bool:
    """Can a bounded-cost depreciation close the bilateral imbalance? (Conclusion 1)."""
    return required_depreciation(p) <= p.max_depreciation


def r_min(p: Params = BASELINE) -> float:
    """Minimum rebalancing power for feasibility: R_min = imbalance0 / max_depreciation."""
    return p.imbalance0 / p.max_depreciation


def chi_threshold(theta_dollar: float, p: Params = BASELINE) -> float:
    """The capital-controls wedge chi at which R = R_min, given theta_dollar (Conclusion 2).

    Below this chi (more open) rebalancing is feasible; above it (more closed) it is blocked.
    Returns nan if no positive chi satisfies it (already infeasible even at an open account).
    """
    # R = (1/(1+chi))*(1-theta_dollar)*(eta+eta_star-1)*gamma = R_min
    num = (1.0 - theta_dollar) * (p.eta + p.eta_star - 1.0) * p.gamma
    if num <= 0:
        return float("nan")
    one_plus_chi = num / r_min(p)
    chi = one_plus_chi - 1.0
    return chi if chi >= 0 else float("nan")


def replace(p: Params, **kw) -> Params:
    """Return a copy of the calibration with some parameters overridden (for comparative statics)."""
    from dataclasses import replace as _r
    return _r(p, **kw)
