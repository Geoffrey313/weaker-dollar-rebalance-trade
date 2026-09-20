"""Conclusions 1 & 2 from the DCP rebalancing block, with sensitivity (audit B2).

C1: at observed dollar invoicing and a closed Chinese capital account, no bounded-cost
    depreciation closes the bilateral imbalance (the FX channel is structurally constrained).
C2: rebalancing becomes feasible only once the frictions fall below a switching frontier
    (RMB internationalisation / capital-account opening).
Sensitivity: the result is calibration-driven, so we trace it over the trade elasticity,
openness, initial imbalance, bounded-cost depreciation limit, dollar-invoicing share, and the
capital-controls damping function.
"""
from __future__ import annotations

import pandas as pd

from src.engine.calibration import BASELINE
from src.engine.model import (rebalancing_power, required_depreciation, is_feasible,
                              chi_threshold, replace)


def conclusion1() -> pd.DataFrame:
    """Baseline (observed frictions) vs an open/flexible benchmark."""
    scenarios = {
        "observed (theta_$=0.95, chi=1)": BASELINE,
        "open+flexible (theta_$=0, chi=0)": replace(BASELINE, theta_dollar=0.0, chi=0.0),
        "flexible prices only (theta_$=0, chi=1)": replace(BASELINE, theta_dollar=0.0),
        "open account only (theta_$=0.95, chi=0)": replace(BASELINE, chi=0.0),
    }
    rows = []
    for name, p in scenarios.items():
        req = required_depreciation(p)
        rows.append({"scenario": name, "R": rebalancing_power(p),
                     "required_deprec": req, "feasible": is_feasible(p)})
    return pd.DataFrame(rows)


def threshold_frontier() -> pd.DataFrame:
    """Conclusion 2: the capital-openness threshold chi* as dollar invoicing varies."""
    rows = []
    for td in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        rows.append({"theta_dollar": td, "chi_threshold": chi_threshold(td)})
    return pd.DataFrame(rows)


def elasticity_sensitivity() -> pd.DataFrame:
    """Feasibility of the observed regime under alternative trade elasticities."""
    rows = []
    for eta in [1.0, 1.5, 2.0, 3.0, 5.0]:
        p = replace(BASELINE, eta=eta, eta_star=eta)
        rows.append({"eta": eta, "R_observed": rebalancing_power(p),
                     "required_deprec": required_depreciation(p), "feasible": is_feasible(p)})
    return pd.DataFrame(rows)


def calibration_sensitivity() -> pd.DataFrame:
    """One-way sensitivities for the calibrated objects that can drive feasibility."""
    grids = {
        "gamma": [0.10, 0.15, 0.20, 0.25, 0.30],
        "imbalance0": [0.015, 0.020, 0.030, 0.040, 0.050],
        "max_depreciation": [0.15, 0.20, 0.25, 0.35, 0.50],
        "theta_dollar": [0.70, 0.80, 0.90, 0.95, 0.99],
        "chi": [0.0, 0.5, 1.0, 2.0, 4.0],
    }
    rows = []
    for param, values in grids.items():
        for value in values:
            p = replace(BASELINE, **{param: value})
            rows.append({
                "parameter": param,
                "value": value,
                "R_observed": rebalancing_power(p),
                "required_deprec": required_depreciation(p),
                "feasible": is_feasible(p),
            })
    return pd.DataFrame(rows)


def damping_sensitivity() -> pd.DataFrame:
    """Sensitivity to Phi_k(chi)=1/(1+k*chi), the unestimated capital-controls mapping."""
    rows = []
    for damping_scale in [0.0, 0.5, 1.0, 2.0, 4.0]:
        rows.append({
            "damping_scale_k": damping_scale,
            "phi_chi": 1.0 / (1.0 + damping_scale * BASELINE.chi),
            "R_observed": rebalancing_power(BASELINE, damping_scale=damping_scale),
            "required_deprec": required_depreciation(BASELINE, damping_scale=damping_scale),
            "feasible": is_feasible(BASELINE, damping_scale=damping_scale),
            "chi_threshold_theta_0_5": chi_threshold(0.5, damping_scale=damping_scale),
            "chi_threshold_theta_0_7": chi_threshold(0.7, damping_scale=damping_scale),
        })
    return pd.DataFrame(rows)


def sensitivity() -> pd.DataFrame:
    """Backward-compatible alias for the original elasticity table."""
    return elasticity_sensitivity()


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("PHASE 2 (baseline reduced form) — the exchange-rate rebalancing channel\n")
    print("Conclusion 1: does a bounded-cost depreciation close the bilateral imbalance?")
    print(conclusion1().to_string(index=False))
    print("\nConclusion 2: capital-openness threshold chi* by dollar-invoicing share")
    print("  (feasible only for chi below chi*; nan = infeasible even at an open account)")
    print(threshold_frontier().to_string(index=False))
    print("\nSensitivity (audit B2.1): observed regime under alternative trade elasticities")
    print(elasticity_sensitivity().to_string(index=False))
    print("\nSensitivity (audit B2.2): one-way calibration checks")
    print(calibration_sensitivity().to_string(index=False))
    print("\nSensitivity (audit B2.3): capital-controls damping function Phi_k(chi)")
    print(damping_sensitivity().to_string(index=False))
    print(f"\nReading: at observed dollar invoicing (theta_$~0.95) the required depreciation is")
    print("far beyond any bounded-cost level, for every plausible trade elasticity -> the FX")
    print("channel is constrained in this reduced-form calibration (C1). Rebalancing turns feasible")
    print("only once BOTH invoicing and capital-account frictions fall below the frontier (C2).")
    print("The DCP dominance statement is therefore a calibrated reduced-form result, not yet the")
    print("full dynamic Ramsey GE result.")
