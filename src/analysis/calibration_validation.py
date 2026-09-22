"""Validation of the structural conclusions under two baseline calibrations.

The paper's main structural baseline is data-disciplined: it starts from the literature
calibration and replaces the parameters identified by this episode. This module recomputes the
same H2 and H3 diagnostics under the pure literature calibration and under the data-disciplined
baseline. It is a robustness check of convergence, not a third estimator.
"""
from __future__ import annotations

import pandas as pd

from src.analysis.baseline import baseline
from src.analysis.dsge_counterfactual import efficiency
from src.engine.calibration import LITERATURE, Params
from src.engine.model import is_feasible, replace, required_depreciation


def _row(label: str, p: Params) -> dict:
    no_invoicing = efficiency(replace(p, theta_dollar=0.0))
    observed = efficiency(p)
    open_account = efficiency(replace(p, chi=0.0))
    closed_account = efficiency(replace(p, chi=4.0))
    no_friction = replace(p, theta_dollar=0.0, chi=0.0)
    invoicing_ratio = observed / no_invoicing
    wedge_ratio = closed_account / open_account
    return {
        "calibration": label,
        "required_observed": required_depreciation(p),
        "feasible_observed": is_feasible(p),
        "required_no_friction": required_depreciation(no_friction),
        "feasible_no_friction": is_feasible(no_friction),
        "invoicing_ratio": invoicing_ratio,
        "wedge_ratio": wedge_ratio,
        "ranking_holds": wedge_ratio < invoicing_ratio,
    }


def compare() -> pd.DataFrame:
    """H2 and H3 diagnostics under literature and data-disciplined calibrations."""
    return pd.DataFrame([
        _row("literature", LITERATURE),
        _row("data_disciplined", baseline()),
    ])


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(compare().to_string(index=False))
