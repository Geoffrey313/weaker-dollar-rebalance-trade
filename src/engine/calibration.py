"""Literature calibration of the two-country NOEM/DCP model.

Deep preference, technology, and policy parameters and the two frictions that carry the paper's
mechanism: theta_dollar (dollar-price stickiness under dollar invoicing) and chi (the capital-
controls wedge). These are literature defaults. The paper's baseline (src.analysis.baseline)
replaces every parameter the project's data identify (eta, theta_dollar, rho_tau, import_share,
gamma, imbalance0) with its estimate from src.analysis.parameter_estimation; the rest stay at the
values below and are varied in the sensitivity analyses.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # --- deep parameters (literature-standard) ---
    beta: float = 0.99      # quarterly discount factor
    sigma: float = 2.0      # inverse EIS / risk aversion
    phi: float = 1.0        # inverse Frisch elasticity
    eta: float = 1.5        # US import-demand elasticity (literature default; estimated in the baseline)
    eta_star: float = 1.5   # foreign counterpart (import demand elasticity)
    gamma: float = 0.20     # bilateral openness (X+M)/Y in the paper's baseline (estimated from
                            # Census trade and BEA GDP); 0.20 is only the literature default
    epsilon: float = 6.0    # elasticity across varieties (markup epsilon/(epsilon-1)=1.2)
    theta: float = 0.75     # Calvo price stickiness (baseline)

    # --- dynamic (DSGE) parameters ---
    phi_pi: float = 1.5     # Taylor rule inflation response (Taylor principle)
    phi_y: float = 0.5      # Taylor rule output-gap response (>=~0.5 needed for determinacy
                            # given the open-economy net-export coupling in the IS curve)
    portfolio_cost: float = 0.02  # Phi: portfolio-adjustment cost (pins stationary NFA)
    rho_tau: float = 0.90   # persistence of the tariff shock (estimated in the baseline)
    rho_z: float = 0.90     # persistence of the engineered-depreciation (UIP) shock
    import_share: float = 0.65  # US imports as a share of bilateral US-China trade (literature
                                # default; the baseline uses the 2017 Census value, 0.80)

    @property
    def kappa(self) -> float:
        """Calvo slope of the NK Phillips curve, (1-theta)(1-beta*theta)/theta."""
        return (1 - self.theta) * (1 - self.beta * self.theta) / self.theta

    # --- frictions carrying the mechanism ---
    theta_dollar: float = 0.95  # dollar-price stickiness (literature default; the baseline uses the
                                # exchange-rate pass-through estimate)
    chi: float = 1.0            # capital-controls wedge (>=0; 0 = open account, large = closed)

    # --- steady-state trade / imbalance ---
    imbalance0: float = 0.03      # initial bilateral imbalance RELATIVE TO US OUTPUT (the units of
                                  # R in model.py); literature default, the baseline uses the 2017
                                  # Census and BEA value, 1.9% of output
    max_depreciation: float = 0.25  # largest real depreciation whose output-gap cost is tolerable
                                    # ("bounded output-gap cost"); ~25% is already very large


LITERATURE = Params()
# Literature defaults. The paper's baseline overrides the parameters the data identify
# (src.analysis.baseline.baseline()); BASELINE is kept as the engine's default argument.
BASELINE = LITERATURE

# Notes on the two frictions (the friction "dials", audit B2):
#  - theta_dollar in [0,1]: 1 = fully sticky dollar prices (pure DCP, no exchange-rate pass-through
#    to border prices on impact); 0 = flexible / producer-currency-like (full pass-through).
#    Estimated from exchange-rate pass-through into dollar import prices (0.885 at the baseline).
#  - chi >= 0: capital-controls wedge. 0 = open capital account (exchange rate can equilibrate);
#    large = closed account (China), which chokes the financing side of any rebalancing.
