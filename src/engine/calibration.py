"""Calibration of the two-country NOEM/DCP model (Phase 2).

Deep preference/technology parameters and the two frictions that carry the paper's mechanism:
theta_dollar (degree of dollar-price stickiness / dollar invoicing, DCP) and chi (capital-
controls wedge). Values are standard in the international-macro literature; the dollar-invoicing
share for US-China trade is calibrated (not estimated) from Gopinath et al. (2020) / Boz et al.

These are the objects the audit flags as decisive (B2): the central result is calibration-
driven, so every downstream analysis reports sensitivity over theta_dollar and chi.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # --- deep parameters (literature-standard) ---
    beta: float = 0.99      # quarterly discount factor
    sigma: float = 2.0      # inverse EIS / risk aversion
    phi: float = 1.0        # inverse Frisch elasticity
    eta: float = 1.5        # elasticity of substitution Home vs Foreign goods (expenditure switching)
    eta_star: float = 1.5   # foreign counterpart (import demand elasticity)
    gamma: float = 0.20     # weight on Foreign goods (openness / import share)
    epsilon: float = 6.0    # elasticity across varieties (markup epsilon/(epsilon-1)=1.2)
    theta: float = 0.75     # Calvo price stickiness (baseline)

    # --- dynamic (DSGE) parameters ---
    phi_pi: float = 1.5     # Taylor rule inflation response (Taylor principle)
    phi_y: float = 0.5      # Taylor rule output-gap response (>=~0.5 needed for determinacy
                            # given the open-economy net-export coupling in the IS curve)
    portfolio_cost: float = 0.02  # Phi: portfolio-adjustment cost (pins stationary NFA)
    rho_tau: float = 0.90   # persistence of the tariff shock
    rho_z: float = 0.90     # persistence of the engineered-depreciation (UIP) shock
    import_share: float = 0.65  # US imports as a share of bilateral US-China trade (deficit:
                                # imports >> exports), so the DCP-muted import channel dominates

    @property
    def kappa(self) -> float:
        """Calvo slope of the NK Phillips curve, (1-theta)(1-beta*theta)/theta."""
        return (1 - self.theta) * (1 - self.beta * self.theta) / self.theta

    # --- frictions carrying the mechanism ---
    theta_dollar: float = 0.95  # dollar-price stickiness / dollar-invoicing share (Gopinath/Boz: US-China ~ dollar)
    chi: float = 1.0            # capital-controls wedge (>=0; 0 = open account, large = closed)

    # --- steady-state trade / imbalance ---
    imbalance0: float = 0.03      # observed bilateral imbalance, share of bilateral trade (~3%)
    max_depreciation: float = 0.25  # largest real depreciation whose output-gap cost is tolerable
                                    # ("bounded output-gap cost"); ~25% is already very large


BASELINE = Params()

# Notes on the two frictions (the friction "dials", audit B2):
#  - theta_dollar in [0,1]: 1 = fully sticky dollar prices (pure DCP, no exchange-rate pass-through
#    to border prices on impact); 0 = flexible / producer-currency-like (full pass-through).
#    Observed US-China invoicing is overwhelmingly in dollars -> theta_dollar high (~0.9-1.0).
#  - chi >= 0: capital-controls wedge. 0 = open capital account (exchange rate can equilibrate);
#    large = closed account (China), which chokes the financing side of any rebalancing.
