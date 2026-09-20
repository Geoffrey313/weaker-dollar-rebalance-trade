"""Dynamic Mar-a-Lago counterfactual (Phase 2, GE): the weaker-dollar / output-gap trade-off.

Using the solved two-country DCP DSGE (src/engine/dsge), we engineer a weaker dollar (a UIP
shock that appreciates the RMB, e>0) and measure, over the transition, the bilateral net-export
improvement it buys and the output-gap cost it imposes, as the dollar-invoicing friction
theta_dollar (and the capital-controls wedge chi) vary.

Discounted sums for a unit weaker-dollar shock:
  NX_cum  = sum_t beta^t nx_t        (rebalancing delivered)
  GAP_cum = sum_t beta^t |y_t|       (output-gap cost incurred)
  efficiency = NX_cum / GAP_cum      (rebalancing bought per unit of output-gap cost)

Dynamic Conclusion 1: if efficiency falls as theta_dollar rises, the exchange-rate channel buys
less rebalancing per unit of activity cost under dominant-currency pricing — the FX lever is
structurally constrained. This is the GE version of the reduced-form threshold result, now with
an endogenous exchange-rate path and a genuine output-gap cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.calibration import BASELINE
from src.engine.dsge import irf
from src.engine.model import replace

HORIZON = 40


def _sums(df: pd.DataFrame, beta: float) -> dict:
    disc = beta ** np.arange(len(df))
    nx_cum = float(np.sum(disc * df["nx"].to_numpy()))
    gap_cum = float(np.sum(disc * np.abs(df["y"].to_numpy())))
    peak_fx = float(df["e"].abs().max())  # peak RMB appreciation (weaker-dollar shock)
    return {"peak_rmb_appreciation": peak_fx, "NX_cum": nx_cum, "GAP_cum": gap_cum,
            "efficiency": nx_cum / gap_cum if gap_cum > 0 else np.inf}


def theta_dollar_grid() -> pd.DataFrame:
    rows = []
    for td in [0.0, 0.50, 0.75, 0.90, 0.95, 0.99]:
        p = replace(BASELINE, theta_dollar=td)
        df = irf("z", periods=HORIZON, p=p)
        rows.append({"theta_dollar": td, **_sums(df, p.beta)})
    return pd.DataFrame(rows)


def chi_grid() -> pd.DataFrame:
    rows = []
    for chi in [0.0, 0.5, 1.0, 2.0, 4.0]:
        p = replace(BASELINE, chi=chi)
        df = irf("z", periods=HORIZON, p=p)
        rows.append({"chi": chi, **_sums(df, p.beta)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("Dynamic Mar-a-Lago counterfactual: weaker-dollar shock, GE transition\n")
    print("Rebalancing bought per unit output-gap cost, by dollar-invoicing (theta_dollar):")
    td = theta_dollar_grid()
    print(td.to_string(index=False))
    base_eff = td.loc[td["theta_dollar"] == 0.0, "efficiency"].iloc[0]
    obs_eff = td.loc[td["theta_dollar"] == 0.95, "efficiency"].iloc[0]
    print(f"\n  efficiency at theta_$=0.95 (observed) vs theta_$=0 (flexible): "
          f"{obs_eff:.3f} vs {base_eff:.3f}  (ratio {obs_eff/base_eff:.2f})")
    print("\nBy capital-controls wedge (chi):")
    print(chi_grid().to_string(index=False))
    print("\nReading: the sign of the theta_dollar gradient is the GE dynamic test of Conclusion 1.")
    print("A falling efficiency as theta_dollar rises means the weaker dollar buys less rebalancing")
    print("per unit of output-gap cost under dominant-currency pricing (FX channel constrained).")
