"""Staggered-robust sector event study (binarised, stacked DiD) — referee robustness.

The baseline event study treats 2018-2019 as a single event with a continuous treatment. This
module is the framework's promised secondary check: binarise the treatment and exploit the
staggered timing of the individual tariff waves with a design free of the "forbidden
comparisons" that bias two-way FE event studies (Sun-Abraham / Callaway-Sant'Anna concern).

Method — stacked difference-in-differences (Cengiz et al.):
  1. Assign each HS4 a treatment cohort = the first quarter (2018-2019) its effective tariff
     jumps more than THRESH above its 2015-2017 baseline; products that never jump are the
     never-treated controls.
  2. For each cohort c, build a clean sub-experiment: the cohort-c treated products PLUS all
     never-treated products, in an event window around c. Event time k is relative to c.
  3. Stack the sub-experiments and estimate
        log V = a_{p,stack} + d_{q,stack} + sum_{k != -1} beta_k (treated x 1[k]) + u
     with product-by-stack and quarter-by-stack fixed effects, clustered by product. Only
     never-treated units serve as controls within each stack, so no already-treated unit is
     used as a control for a later-treated one.
If the stacked beta_k match the baseline (flat pre-trends, negative post), the baseline is not
contaminated by staggered-timing bias.

Corrective weights (Wing, Freedman and Hollingsworth, 2024). The unweighted stacked regression
averages the sub-experiments with implicit variance weights, so it does not estimate a
well-defined aggregate effect when treated and control group sizes differ across stacks. The
weighted version gives each treated product weight 1 and each control product in stack c weight
    Q_c = (N^D_c / N^D) / (N^C_c / N^C),
where N^D_c and N^C_c count the treated and control products in stack c and N^D, N^C are their
sums over stacks. The weighted regression recovers the trimmed aggregate average treatment effect
on the treated, which weights each cohort by its share of treated products. Both versions are
reported; the weighted one is the reference estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.twfe import twfe_ols
from src.data.sector_imports import load_china_imports_hs4

THRESH = 0.05          # tariff jump (duties/value) that marks a product as treated
PRE, POST = 4, 6       # event window in quarters around the cohort's treatment
REF_K = -1


def _quarterly() -> pd.DataFrame:
    m = load_china_imports_hs4()
    m = m[(m["effective_tariff"] >= 0) & (m["effective_tariff"] <= 1)]
    m["qi"] = m["year"] * 4 + (m["month"] - 1) // 3           # quarter index
    q = (m.groupby(["hs4", "qi"], as_index=False)
         .agg(value=("value_usd", "sum"), duties=("duties_usd", "sum")))
    q = q[q["value"] > 0].copy()
    q["log_value"] = np.log(q["value"])
    q["eff"] = q["duties"] / q["value"]
    return q


def _cohorts(q: pd.DataFrame, thresh: float = THRESH) -> pd.Series:
    """First treated quarter index per HS4 (NaN = never treated)."""
    base = q[q["qi"] < 2018 * 4].groupby("hs4")["eff"].mean().rename("base")
    d = q.merge(base, on="hs4")
    treat = d[(d["qi"] >= 2018 * 4) & (d["eff"] - d["base"] > thresh)]
    return treat.groupby("hs4")["qi"].min()


def build_stacked(thresh: float = THRESH) -> pd.DataFrame:
    q = _quarterly()
    cohort = _cohorts(q, thresh)
    never = sorted(set(q["hs4"]) - set(cohort.index))
    cohort_qs = sorted(c for c in cohort.unique() if (cohort == c).sum() >= 15)  # non-trivial cohorts
    frames = []
    for c in cohort_qs:
        treated = cohort[cohort == c].index
        keep = q[q["hs4"].isin(list(treated) + never) & q["qi"].between(c - PRE, c + POST)].copy()
        keep["stack"] = int(c)
        keep["treated"] = keep["hs4"].isin(treated).astype(float)
        keep["k"] = keep["qi"] - c
        frames.append(keep)
    st = pd.concat(frames, ignore_index=True)
    st["hs4_stack"] = st["hs4"].astype(str) + "_" + st["stack"].astype(str)
    st["q_stack"] = st["qi"].astype(str) + "_" + st["stack"].astype(str)
    st["wfh_weight"] = corrective_weights(st)
    return st


def stack_counts(st: pd.DataFrame) -> pd.DataFrame:
    """Treated and control products per stack (N^D_c, N^C_c)."""
    units = st[["stack", "hs4", "treated"]].drop_duplicates(["stack", "hs4"])
    counts = (units.groupby("stack")["treated"]
              .agg(n_treated=lambda s: int((s == 1).sum()), n_control=lambda s: int((s == 0).sum())))
    return counts.reset_index()


def corrective_weights(st: pd.DataFrame) -> pd.Series:
    """Wing-Freedman-Hollingsworth weights: 1 for treated, Q_c for controls of stack c."""
    counts = stack_counts(st)
    share_d = counts["n_treated"] / counts["n_treated"].sum()
    share_c = counts["n_control"] / counts["n_control"].sum()
    q = pd.Series((share_d / share_c).to_numpy(), index=counts["stack"].to_numpy())
    return pd.Series(np.where(st["treated"].to_numpy() == 1, 1.0, q.reindex(st["stack"]).to_numpy()),
                     index=st.index)


def run(thresh: float = THRESH, weighted: bool = False) -> pd.DataFrame:
    """Stacked event study; `weighted=True` applies the corrective weights."""
    st = build_stacked(thresh)
    ks = [k for k in range(-PRE, POST + 1) if k != REF_K]
    terms = []
    for k in ks:
        col = f"k_{k}"
        st[col] = st["treated"] * (st["k"] == k)
        terms.append(col)
    res = twfe_ols(st, "log_value", terms, "hs4_stack", "q_stack", cluster="hs4",
                   weights="wfh_weight" if weighted else None)
    res.index = ks
    res.index.name = "event_time"
    return res


def post_mean(res: pd.DataFrame) -> float:
    """Equally weighted average of the post-treatment coefficients (k >= 0)."""
    return float(res.loc[[k for k in res.index if k >= 0], "beta"].mean())


def threshold_robustness(weighted: bool = False) -> pd.DataFrame:
    """Post-treatment average effect under alternative treatment thresholds (3/5/10 pp)."""
    rows = []
    for thr in (0.03, 0.05, 0.10):
        res = run(thr, weighted=weighted)
        post = res.loc[[k for k in res.index if k >= 0]]
        rows.append({"threshold_pp": int(thr * 100), "post_mean_beta": post["beta"].mean(),
                     "post_significant": f"{(post['p'] < 0.05).sum()}/{len(post)}"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    st = build_stacked()
    print("Staggered-robust event study (binarised, stacked DiD; product-by-stack + quarter-by-stack FE)")
    print(f"  cohorts (waves) x never-treated controls; clustered by HS4; N={len(st)} stacked obs")
    print(stack_counts(st).to_string(index=False))
    for weighted in (False, True):
        res = run(weighted=weighted)
        label = "WEIGHTED (Wing-Freedman-Hollingsworth)" if weighted else "UNWEIGHTED"
        print(f"\n{label}")
        print(res[["beta", "se", "p"]].to_string())
        pre = res.loc[[k for k in res.index if k < 0]]
        post = res.loc[[k for k in res.index if k >= 0]]
        print(f"Pre-trends: {(pre['p'] < 0.05).sum()}/{len(pre)} significant at 5%.")
        print(f"Post-treatment mean beta: {post['beta'].mean():.3f} "
              f"({(post['p'] < 0.05).sum()}/{len(post)} significant).")
        print("Treatment-threshold robustness (3/5/10 pp):")
        print(threshold_robustness(weighted=weighted).to_string(index=False))
