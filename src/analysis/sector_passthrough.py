"""Product/sector pass-through panel (eq:sector_panel): the identifying reduced form.

On the HS4 x month panel of US imports from China (src/data/sector_imports), estimates the
response of the import trade flow to the effective applied tariff:

    z_{p,t} = alpha_p + delta_t + beta * tau_{p,t} + u_{p,t}

with product (HS4) and time (month) fixed effects and SE clustered by product (via the
validated two-way FE estimator in src/common/twfe). tau_{p,t} is the effective tariff
(duties / customs value).

Outcome: log import value (customs value, pre-duty). Census reports no usable quantity at the
HS4 level (UNIT_QY1 = '-'), so the direct quantity/unit-value response needs a finer HS6/HS10
pull; the border-PRICE pass-through is measured separately with the BLS NAICS index (which
needs the HS<->NAICS bridge). With border prices roughly flat across the waves (BLS evidence),
the value response here is informative about the quantity adjustment (audit A1/A2: the reduced
form identifies tariff incidence at the product level; the aggregate FX channel stays structural).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.twfe import twfe_cluster
from src.data.sector_imports import load_china_imports_hs4

OUTCOMES = {"log_value": "log import value (customs, pre-duty)"}


def build_panel() -> pd.DataFrame:
    """HS4 x month panel with a period key and a bounded effective tariff."""
    df = load_china_imports_hs4()
    df["period"] = df["year"] * 100 + df["month"]
    # An effective rate outside [0, 1] is a data artifact (tiny value / reporting); drop it.
    df = df[(df["effective_tariff"] >= 0) & (df["effective_tariff"] <= 1)].copy()
    return df


def run() -> pd.DataFrame:
    df = build_panel()
    rows = []
    for y, label in OUTCOMES.items():
        r = twfe_cluster(df[np.isfinite(df[y])], y, "effective_tariff", "hs4", "period")
        rows.append({"outcome": label, **r})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    res = run()
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("Sector pass-through, TWFE (HS4 + month FE, SE clustered by HS4)")
    print("  z ~ beta * effective_tariff   (tau = duties / customs value)\n")
    print(res[["outcome", "beta", "se", "p", "n", "n_fe1", "n_fe2", "n_cluster"]].to_string(index=False))
    print("\nInterpretation: beta<0 => higher effective tariff on a product is associated with a")
    print("lower value of US imports from China in that product (trade contraction). With BLS")
    print("border prices roughly flat across the waves, this value response reflects quantities.")
