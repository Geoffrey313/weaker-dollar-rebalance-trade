"""Dynamic sector event study (eq:sector_panel, event-time form).

Continuous-treatment event study of US import value from China on the tariff shock, at the
HS4 x quarter level:

    log V_{p,q} = alpha_p + delta_q + sum_{k != ref} beta_k * (Dtau_p x 1[quarter = k]) + u_{p,q}

with product (HS4) and calendar-quarter fixed effects and SE clustered by product (twfe_ols).
Dtau_p is the product's tariff-shock intensity, fixed pre-episode: mean effective tariff in
2019 minus mean in 2017. The omitted reference quarter is 2018Q2 (just before List 1, 2018-07).

beta_k for pre-episode quarters tests parallel trends (should be flat ~ 0); beta_k for post
quarters traces the dynamic trade contraction. This is the model-independent dynamic core of
the sector evidence; the aggregate FX channel stays structural (audit A1/A2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.twfe import twfe_ols
from src.data.sector_imports import load_china_imports_hs4

REF_Q = "2018Q2"          # omitted reference quarter (pre List-1)
PRE_YEAR, POST_YEAR = 2017, 2019  # windows defining the tariff-shock intensity Dtau_p


def build_quarter_panel() -> tuple[pd.DataFrame, list[str]]:
    """HS4 x quarter panel with log value, the fixed shock intensity Dtau_p, and event dummies."""
    m = load_china_imports_hs4()
    m = m[(m["effective_tariff"] >= 0) & (m["effective_tariff"] <= 1)]
    m["q"] = m["year"].astype(str) + "Q" + ((m["month"] - 1) // 3 + 1).astype(str)

    # Quarterly aggregation: sum value and duties, recompute the effective tariff.
    q = (m.groupby(["hs4", "q", "year"], as_index=False)
         .agg(value_usd=("value_usd", "sum"), duties_usd=("duties_usd", "sum")))
    q = q[q["value_usd"] > 0].copy()
    q["log_value"] = np.log(q["value_usd"])
    q["eff_tariff"] = q["duties_usd"] / q["value_usd"]

    # Dtau_p: mean effective tariff in POST_YEAR minus PRE_YEAR, per product (fixed).
    yr = (m.groupby(["hs4", "year"], as_index=False)
          .agg(v=("value_usd", "sum"), d=("duties_usd", "sum")))
    yr["t"] = yr["d"] / yr["v"]
    pre = yr[yr["year"] == PRE_YEAR][["hs4", "t"]].rename(columns={"t": "t_pre"})
    post = yr[yr["year"] == POST_YEAR][["hs4", "t"]].rename(columns={"t": "t_post"})
    dtau = pre.merge(post, on="hs4")
    dtau["dtau"] = dtau["t_post"] - dtau["t_pre"]
    q = q.merge(dtau[["hs4", "dtau"]], on="hs4")  # keeps products present in both years

    # Event dummies: Dtau_p x 1[quarter = k], one column per quarter except the reference.
    quarters = sorted(q["q"].unique(), key=lambda s: (int(s[:4]), int(s[-1])))
    terms = []
    for k in quarters:
        if k == REF_Q:
            continue
        col = f"e_{k}"
        q[col] = q["dtau"] * (q["q"] == k)
        terms.append(col)
    return q, terms


if __name__ == "__main__":
    panel, terms = build_quarter_panel()
    res = twfe_ols(panel, "log_value", terms, "hs4", "q", cluster="hs4")
    res.index = [t.replace("e_", "") for t in res.index]
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("Dynamic sector event study: log import value on Dtau_p x quarter | HS4 + quarter FE")
    print(f"  Dtau_p = eff. tariff {POST_YEAR} - {PRE_YEAR}; reference quarter = {REF_Q}; cluster HS4")
    print(f"  N={int(panel.shape[0])} product-quarters, HS4={panel['hs4'].nunique()}\n")
    print(res[["beta", "se", "p"]].to_string())
    pre = res.loc[[q for q in res.index if int(q[:4]) < 2018 or q in ("2018Q1",)]]
    post = res.loc[[q for q in res.index if int(q[:4]) > 2018 or q in ("2018Q3", "2018Q4")]]
    print(f"\nPre-episode coefficients (parallel-trends check): "
          f"{(pre['p'] < 0.05).sum()}/{len(pre)} significant at 5% (few expected).")
    print(f"Post-episode mean beta: {post['beta'].mean():.3f} "
          f"({(post['p'] < 0.05).sum()}/{len(post)} significant) -> dynamic trade contraction.")
