"""H2 impact-bound diagnostics from the DCP rebalancing block, with sensitivity.

The block asks whether an engineered depreciation closes the bilateral imbalance at bounded
activity cost when frictions are held fixed. It is a feasibility diagnostic, not the mechanism
ranking; the general-equilibrium model in src.engine.dsge supplies the H3 ranking between
dollar invoicing and capital controls.

Every scenario starts from the data-disciplined baseline (src.analysis.baseline): estimated
import-demand elasticity and dollar-invoicing friction, and the import share, bilateral openness,
and initial imbalance computed from Census trade and BEA output. Sensitivity ranges for the
trade objects are their observed values over 2015-2021; the range for the invoicing friction is
its 95 percent confidence interval.
"""
from __future__ import annotations

import pandas as pd

from src.analysis.baseline import baseline
from src.analysis.parameter_estimation import (estimate_eta, estimate_passthrough,
                                               trade_share_range)
from src.engine.model import (chi_threshold, is_feasible, rebalancing_power, replace,
                              required_depreciation)


def _row(label: str, p, damping_scale: float = 1.0) -> dict:
    return {"scenario": label, "theta_dollar": p.theta_dollar, "chi": p.chi, "eta": p.eta,
            "eta_star": p.eta_star, "R": rebalancing_power(p, damping_scale=damping_scale),
            "required_deprec": required_depreciation(p, damping_scale=damping_scale),
            "feasible": is_feasible(p, damping_scale=damping_scale)}


def conclusion1() -> pd.DataFrame:
    """Observed frictions against the frictionless and single-friction benchmarks."""
    b = baseline()
    return pd.DataFrame([
        _row("observed frictions", b),
        _row("no dollar-invoicing friction (theta_$=0)", replace(b, theta_dollar=0.0)),
        _row("open capital account (chi=0)", replace(b, chi=0.0)),
        _row("no friction (theta_$=0, chi=0)", replace(b, theta_dollar=0.0, chi=0.0)),
    ])


def threshold_frontier() -> pd.DataFrame:
    """Impact-bound capital-openness threshold chi* as dollar invoicing varies (nan = none)."""
    b = baseline()
    return pd.DataFrame([{"theta_dollar": td, "chi_threshold": chi_threshold(td, p=b)}
                         for td in [0.0, 0.3, 0.5, 0.7, 0.9, b.theta_dollar]])


def elasticity_sensitivity() -> pd.DataFrame:
    """Trade elasticities: the level-tariff estimate of eta and alternative foreign elasticities
    (eta_star is not identified by US import data)."""
    b = baseline()
    eta = estimate_eta()
    return pd.DataFrame([
        _row("baseline", b),
        _row("eta from the level tariff", replace(b, eta=eta["eta_level"])),
        _row("eta_star = 1", replace(b, eta_star=1.0)),
        _row("eta_star = eta", replace(b, eta_star=b.eta)),
        _row("eta_star = 5", replace(b, eta_star=5.0)),
    ])


def calibration_sensitivity() -> pd.DataFrame:
    """One-way changes: observed 2015-2021 ranges of the trade objects, the confidence interval
    of the invoicing friction, the tolerated appreciation, and the capital-controls wedge."""
    b = baseline()
    years = trade_share_range()
    pt = estimate_passthrough()
    lo, hi = pt["theta"] - 1.96 * pt["se"], min(pt["theta"] + 1.96 * pt["se"], 0.999)
    grids = {
        "gamma": [years["gamma"].min(), years["gamma"].max()],
        "imbalance0": [years["imbalance0"].min(), years["imbalance0"].max()],
        "import_share": [years["import_share"].min(), years["import_share"].max()],
        "theta_dollar": [lo, hi],
        "max_depreciation": [0.15, 0.50],
    }
    rows = []
    for param, values in grids.items():
        for value in values:
            rows.append({"parameter": param, "value": float(value),
                         **_row(param, replace(b, **{param: float(value)}))})
    return pd.DataFrame(rows)


def damping_sensitivity() -> pd.DataFrame:
    """Sensitivity to Phi_k(chi)=1/(1+k*chi), the unestimated capital-controls mapping."""
    b = baseline()
    rows = []
    for k in [0.0, 1.0, 4.0]:
        rows.append({"damping_scale_k": k, "phi_chi": 1.0 / (1.0 + k * b.chi),
                     **_row(f"damping {k}", b, damping_scale=k)})
    return pd.DataFrame(rows)


def sensitivity() -> pd.DataFrame:
    """Backward-compatible alias for the elasticity table."""
    return elasticity_sensitivity()


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    cols = ["scenario", "R", "required_deprec", "feasible"]
    print("H2 impact bound at the data-disciplined baseline\n")
    print(conclusion1()[cols].to_string(index=False))
    print("\nFeasibility frontier chi*(theta_$) (nan = infeasible for every chi >= 0)")
    print(threshold_frontier().to_string(index=False))
    print("\nTrade elasticities")
    print(elasticity_sensitivity()[cols].to_string(index=False))
    print("\nOne-way changes (observed ranges, confidence interval, tolerated appreciation)")
    print(calibration_sensitivity()[["parameter", "value", "R", "required_deprec", "feasible"]]
          .to_string(index=False))
    print("\nCapital-controls damping scale")
    print(damping_sensitivity()[["damping_scale_k", "R", "required_deprec", "feasible"]]
          .to_string(index=False))
