"""Two-country DCP rebalancing block (Phase 2, baseline reduced form).

This is the calibrated reduced form behind the paper's impact-bound feasibility test: how much a
bilateral real depreciation improves the US-China trade balance when the two frictions
(dollar-price stickiness theta_dollar and the capital-controls wedge chi) are held fixed. It
delivers the H2 diagnostic question: is an engineered depreciation large enough to rebalance
within the bounded-cost threshold? The dynamic GE model in src/engine/dsge.py supplies the H3
mechanism ranking; this block isolates the impact bound transparently.

Derivation (log-linear, OUTPUT units). Let X and M be bilateral exports and imports, s = M/(X+M)
the import share of bilateral trade and gamma = (X+M)/Y bilateral openness. A relative-price
change dp raises import value by (1-eta)*dp*M and export value by eta_star*dp*X, so the trade
balance relative to output moves by gamma*[(1-s)*eta_star + s*(eta-1)]*dp. In the impact bound a
real RMB appreciation de moves the relative price by (1-theta_dollar)*de and capital controls
let only Phi(chi) of the adjustment be financed:

    R(theta_dollar, chi) = Phi(chi) * (1 - theta_dollar) * gamma * [(1-s)*eta_star + s*(eta-1)]

  * [(1-s)*eta_star + s*(eta-1)]: the trade-weighted Marshall-Lerner term. It equals the long-run
    net-export response of the general-equilibrium model (manuscript, Proposition 2), and with
    balanced trade (s = 1/2) it is (eta + eta_star - 1)/2.
  * (1 - theta_dollar): exchange-rate pass-through into border prices, applied symmetrically.
  * Phi(chi) = 1/(1+k*chi) in (0,1]: capital-controls damping (k is a sensitivity parameter).
  * gamma: bilateral openness, which converts the trade balance into output units, so the
    initial imbalance imbalance0 is measured relative to output as well.

At the data-disciplined baseline (src.analysis.baseline) gamma * trade_term falls short of
R_min, so the bound is infeasible for every theta_dollar and chi (manuscript, Proposition 1(c)).
Because the bound applies (1 - theta_dollar) to both margins, it does not rank the frictions;
the general-equilibrium model in src/engine/dsge.py does.

Rebalancing is FEASIBLE if the depreciation needed to close the observed imbalance,
|de| = imbalance0 / R, stays within the bounded-cost maximum max_depreciation. Equivalently,
R must exceed R_min = imbalance0 / max_depreciation. The (theta_dollar, chi) locus where
R = R_min is the impact-bound switching frontier.
"""
from __future__ import annotations

import numpy as np

from src.engine.calibration import Params, BASELINE


def capital_damping(p: Params = BASELINE, damping_scale: float = 1.0) -> float:
    """Phi_k(chi) = 1/(1+k*chi): 1 at an open account, lower as the account closes.

    damping_scale controls how sharply the capital-controls wedge attenuates rebalancing.
    It is a sensitivity parameter, not a deep object calibrated from the data.
    """
    if damping_scale < 0:
        raise ValueError("damping_scale must be non-negative")
    return 1.0 / (1.0 + damping_scale * p.chi)


def trade_term(p: Params = BASELINE) -> float:
    """Trade-weighted Marshall-Lerner term (1-s)*eta_star + s*(eta-1)."""
    return (1.0 - p.import_share) * p.eta_star + p.import_share * (p.eta - 1.0)


def rebalancing_power(p: Params = BASELINE, damping_scale: float = 1.0) -> float:
    """R = d(NX/output)/d(real RMB appreciation): the exchange rate's rebalancing power."""
    return (
        capital_damping(p, damping_scale=damping_scale)
        * (1.0 - p.theta_dollar)
        * p.gamma
        * trade_term(p)
    )


def required_depreciation(p: Params = BASELINE, damping_scale: float = 1.0) -> float:
    """Real depreciation needed to close the imbalance (inf if the FX channel is dead)."""
    R = rebalancing_power(p, damping_scale=damping_scale)
    return np.inf if R <= 0 else p.imbalance0 / R


def is_feasible(p: Params = BASELINE, damping_scale: float = 1.0) -> bool:
    """Can a bounded-cost depreciation close the bilateral imbalance? (H2 impact bound)."""
    return required_depreciation(p, damping_scale=damping_scale) <= p.max_depreciation


def r_min(p: Params = BASELINE) -> float:
    """Minimum rebalancing power for feasibility: R_min = imbalance0 / max_depreciation."""
    return p.imbalance0 / p.max_depreciation


def chi_threshold(theta_dollar: float, p: Params = BASELINE, damping_scale: float = 1.0) -> float:
    """The capital-controls wedge chi at which R = R_min, given theta_dollar.

    Below this chi (more open) rebalancing is feasible; above it (more closed) it is blocked.
    Returns nan if no positive chi satisfies it (already infeasible even at an open account).
    """
    # R = (1/(1+k*chi))*(1-theta_dollar)*gamma*trade_term = R_min
    if damping_scale < 0:
        raise ValueError("damping_scale must be non-negative")
    num = (1.0 - theta_dollar) * p.gamma * trade_term(p)
    if num <= 0:
        return float("nan")
    one_plus_chi = num / r_min(p)
    if damping_scale == 0:
        return float("inf") if one_plus_chi >= 1.0 else float("nan")
    chi = (one_plus_chi - 1.0) / damping_scale
    return chi if chi >= 0 else float("nan")


def replace(p: Params, **kw) -> Params:
    """Return a copy of the calibration with some parameters overridden (for comparative statics)."""
    from dataclasses import replace as _r
    return _r(p, **kw)
