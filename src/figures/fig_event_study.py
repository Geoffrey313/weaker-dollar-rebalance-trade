"""Figure: event study of the tariff shock on the value of US imports from China.

Coefficients beta_k of the event-study regression (manuscript, equation eq:event) of the log
import value on the tariff-shock intensity interacted with calendar quarters, with product and
quarter fixed effects and 95 percent confidence bands clustered by product. The reference
quarter (2018Q2) is normalized to zero. It shows the flat pre-trends and the persistent
post-2018 contraction behind H1.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.sector_event_study import build_quarter_panel
from src.common.twfe import twfe_ols
from src.figures.style import (GREY_TINT, LANGS, NAVY, TEXT_WIDTH, apply_style, localize,
                               reference_line, save)

LABELS = {
    "en": {"x": "Quarter", "y": r"Effect on the log import value, $\beta_k$"},
    "fr": {"x": "Trimestre", "y": r"Effet sur le log de la valeur importée, $\beta_k$"},
}


def coefficients() -> pd.DataFrame:
    panel, terms = build_quarter_panel()
    res = twfe_ols(panel, "log_value", terms, "hs4", "q", cluster="hs4")
    res.index = [t.replace("e_", "") for t in res.index]
    quarters = sorted(set(panel["q"]), key=lambda s: (int(s[:4]), int(s[-1])))
    beta = res["beta"].reindex(quarters).fillna(0.0)      # reference quarter -> 0
    se = res["se"].reindex(quarters).fillna(0.0)          # no sampling error at the reference
    return pd.DataFrame({"q": quarters, "beta": beta.values, "se": se.values})


def make(lang: str) -> None:
    L = LABELS[lang]
    d = coefficients()
    x = np.arange(len(d))
    onset = int(np.flatnonzero(d["q"] == "2018Q3")[0])

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.9), layout="constrained")
    reference_line(ax, 0.0)
    reference_line(ax, onset - 0.5, axis="v")
    ax.fill_between(x, d["beta"] - 1.96 * d["se"], d["beta"] + 1.96 * d["se"],
                    color=GREY_TINT, lw=0, zorder=2)
    ax.plot(x, d["beta"], color=NAVY, marker="o", ms=3.2, zorder=3)
    ticks = [i for i, q in enumerate(d["q"]) if q.endswith("Q1")]
    ax.set_xticks(ticks, [d["q"].iloc[i][:4] for i in ticks])
    ax.set_xlim(-0.8, len(d) - 0.2)
    ax.set_xlabel(L["x"])
    ax.set_ylabel(L["y"])
    localize(ax, lang)
    save(fig, "fig_event_study", lang)


if __name__ == "__main__":
    apply_style()
    for lang in LANGS:
        make(lang)
    print("fig_event_study written for", ", ".join(LANGS))
