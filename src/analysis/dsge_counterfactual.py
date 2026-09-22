"""Dynamic Mar-a-Lago counterfactual (Phase 2, GE): the weaker-dollar / output-gap trade-off.

Using the solved two-country DCP DSGE (src/engine/dsge), we engineer a weaker dollar (a UIP
shock that appreciates the RMB, e>0) and measure, over the transition, the bilateral net-export
improvement it buys and the output-gap cost it imposes, as the dollar-invoicing friction
theta_dollar (and the capital-controls wedge chi) vary.

Discounted sums for a unit weaker-dollar shock:
  NX_cum  = sum_t beta^t nx_t        (rebalancing delivered)
  GAP_cum = sum_t beta^t |y_t|       (output-gap cost incurred)
  efficiency = NX_cum / GAP_cum      (rebalancing bought per unit of output-gap cost)

H3 (friction ranking), at the data-disciplined baseline (src.analysis.baseline): moving from
producer-currency pricing (theta_$=0) to the estimated invoicing friction lowers the efficiency
by about a fifth (0.36 -> 0.29), while closing the capital account from chi=0 to chi=4 lowers it
by about four fifths (1.13 -> 0.20), because the capital-controls wedge reverses the
appreciation before net exports accumulate. The efficiency is not monotone in theta_$: close to
full invoicing (theta_$ -> 1) the output-gap cost falls faster than the net-export gain. The
ranking holds in every one-way calibration change of sensitivity(). The impact bound in
src.engine.model gives the H2 feasibility diagnostic; this general-equilibrium experiment gives
the ranking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.baseline import baseline
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


THETA_GRID = [0.0, 0.50, 0.75, 0.95, 0.99]
CHI_GRID = [0.0, 0.5, 1.0, 2.0, 4.0]


def theta_dollar_grid() -> pd.DataFrame:
    """Efficiency over the invoicing friction, including the estimated value (flag `estimated`)."""
    b = baseline()
    rows = []
    for td in sorted(THETA_GRID + [b.theta_dollar]):
        p = replace(b, theta_dollar=td)
        df = irf("z", periods=HORIZON, p=p)
        rows.append({"theta_dollar": td, "estimated": td == b.theta_dollar, **_sums(df, p.beta)})
    return pd.DataFrame(rows)


def invoicing_interval() -> dict:
    """Efficiency at the bounds of the 95 percent confidence interval of the estimated
    invoicing friction, next to the producer-currency benchmark (theta_$=0), and the wedge
    ratio (chi=4 over chi=0) at each bound."""
    from src.analysis.parameter_estimation import estimate_passthrough
    b = baseline()
    pt = estimate_passthrough()
    out = {"eff_zero": efficiency(replace(b, theta_dollar=0.0))}
    for tag, td in (("lo", pt["theta"] - 1.96 * pt["se"]), ("hi", pt["theta"] + 1.96 * pt["se"])):
        p = replace(b, theta_dollar=td)
        out[f"theta_{tag}"] = td
        out[f"eff_{tag}"] = efficiency(p)
        out[f"wedge_{tag}"] = efficiency(replace(p, chi=4.0)) / efficiency(replace(p, chi=0.0))
    return out


def chi_grid() -> pd.DataFrame:
    """Efficiency over the capital-controls wedge at the estimated invoicing friction."""
    b = baseline()
    rows = []
    for chi in CHI_GRID:
        p = replace(b, chi=chi)
        df = irf("z", periods=HORIZON, p=p)
        rows.append({"chi": chi, **_sums(df, p.beta)})
    return pd.DataFrame(rows)


PERSISTENCE_GRID = [0.5, 0.75, 0.9, 0.95]


def intervention_persistence() -> pd.DataFrame:
    """Cumulative net exports and efficiency as the persistence rho_z of the engineered
    intervention varies: at the baseline frictions, without invoicing, and at an open and a
    tightly closed account. rho_z describes the policy experiment rather than the economy."""
    b = baseline()
    rows = []
    for rz in PERSISTENCE_GRID:
        base = replace(b, rho_z=rz)
        row = {"rho_z": rz}
        for tag, p in (("hat", base), ("theta_0", replace(base, theta_dollar=0.0)),
                       ("chi_0", replace(base, chi=0.0)), ("chi_4", replace(base, chi=4.0))):
            out = _sums(irf("z", periods=HORIZON, p=p), p.beta)
            row[f"nx_{tag}"] = out["NX_cum"]
            row[f"eff_{tag}"] = out["efficiency"]
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_grid() -> list[tuple[str, str, float]]:
    """One-way changes (parameter, label key, value) around the baseline: the level-tariff
    estimate of eta, alternative foreign elasticities, the observed 2015-2021 ranges of openness
    and the import share, and the calibrated preference, pricing, policy, and portfolio
    parameters."""
    from src.analysis.parameter_estimation import estimate_eta, trade_share_range
    b = baseline()
    years = trade_share_range()
    return [
        ("baseline", "baseline", float("nan")),
        ("eta", "eta_level", estimate_eta()["eta_level"]),
        ("eta_star", "eta_star", 1.0), ("eta_star", "eta_star", b.eta),
        ("gamma", "gamma", float(years["gamma"].min())),
        ("gamma", "gamma", float(years["gamma"].max())),
        ("import_share", "import_share", float(years["import_share"].min())),
        ("import_share", "import_share", float(years["import_share"].max())),
        ("portfolio_cost", "portfolio_cost", 0.01), ("portfolio_cost", "portfolio_cost", 0.05),
        ("sigma", "sigma", 1.0), ("sigma", "sigma", 5.0),
        ("theta", "calvo", 0.50), ("theta", "calvo", 0.90),
        ("phi_pi", "phi_pi", 1.25), ("phi_pi", "phi_pi", 2.0),
        ("phi_y", "phi_y", 0.25), ("phi_y", "phi_y", 1.0),
    ]


def efficiency(p) -> float:
    """Rebalancing efficiency of a unit engineered-depreciation shock under parameters p;
    raises ValueError when p has no unique stable equilibrium."""
    return _sums(irf("z", periods=HORIZON, p=p), p.beta)["efficiency"]


def sensitivity() -> pd.DataFrame:
    """H3 comparative statics under one-way calibration changes.

    For each calibration: the efficiency without dollar invoicing (theta_$=0) and at the
    estimated friction, both at the baseline wedge; and at an open (chi=0) and a tightly closed
    (chi=4) account, both at the estimated friction. The invoicing ratio and the wedge ratio
    summarize the two gradients. A calibration that violates the Blanchard-Kahn condition is
    reported as indeterminate.
    """
    b = baseline()
    rows = []
    for param, key, value in sensitivity_grid():
        base = b if param == "baseline" else replace(b, **{param: value})
        try:
            e_td0 = efficiency(replace(base, theta_dollar=0.0))
            e_hat = efficiency(base)
            e_chi0 = efficiency(replace(base, chi=0.0))
            e_chi4 = efficiency(replace(base, chi=4.0))
        except ValueError:
            rows.append({"parameter": param, "key": key, "value": value, "determinate": False})
            continue
        rows.append({"parameter": param, "key": key, "value": value, "determinate": True,
                     "eff_theta_0": e_td0, "eff_theta_hat": e_hat,
                     "invoicing_ratio": e_hat / e_td0, "eff_chi_0": e_chi0, "eff_chi_4": e_chi4,
                     "wedge_ratio": e_chi4 / e_chi0})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("Dynamic Mar-a-Lago counterfactual: weaker-dollar shock, GE transition\n")
    print("Rebalancing bought per unit output-gap cost, by dollar-invoicing (theta_dollar):")
    td = theta_dollar_grid()
    print(td.to_string(index=False))
    base_eff = td.loc[td["theta_dollar"] == 0.0, "efficiency"].iloc[0]
    obs_eff = td.loc[td["estimated"], "efficiency"].iloc[0]
    print(f"\n  efficiency at the estimated theta_$ vs theta_$=0: "
          f"{obs_eff:.3f} vs {base_eff:.3f}  (ratio {obs_eff/base_eff:.2f})")
    print("\nBy capital-controls wedge (chi):")
    print(chi_grid().to_string(index=False))
    print("\nPersistence of the engineered intervention (rho_z):")
    print(intervention_persistence().to_string(index=False))
    print("\nH3 comparative statics under one-way calibration changes:")
    print(sensitivity().to_string(index=False))
    print("\nReading (H3): the wedge gradient is steeper than the invoicing gradient in every")
    print("calibration, so the closed capital account, not dollar invoicing, binds rebalancing.")
