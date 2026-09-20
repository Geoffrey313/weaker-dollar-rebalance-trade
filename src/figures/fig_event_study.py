"""Figure 1 — sector event study: flat pre-trends, then a persistent post-2018 contraction.

log US import value from China on the tariff-shock intensity interacted with event quarters
(HS4 + quarter FE). The central empirical fact of the paper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.twfe import twfe_ols
from src.analysis.sector_event_study import build_quarter_panel, REF_Q
from src.figures.style import apply_style, save, BLUE, BLUE_TINT, MUTED, ORANGE, LANGS
import matplotlib.pyplot as plt

LABELS = {
    "en": {"title": "US imports from China contract with the tariff, after flat pre-trends",
           "x": "Quarter", "y": "Effect on log import value",
           "onset": "2018 tariff waves", "pre": "flat pre-trends", "ref": "reference"},
    "fr": {"title": "Les importations US de Chine se contractent avec le tarif, apres des pre-tendances plates",
           "x": "Trimestre", "y": "Effet sur la log-valeur importee",
           "onset": "vagues tarifaires 2018", "pre": "pre-tendances plates", "ref": "reference"},
}


def _coefficients() -> pd.DataFrame:
    panel, terms = build_quarter_panel()
    res = twfe_ols(panel, "log_value", terms, "hs4", "q", cluster="hs4")
    res.index = [t.replace("e_", "") for t in res.index]
    quarters = sorted(set(panel["q"]), key=lambda s: (int(s[:4]), int(s[-1])))
    beta = res["beta"].reindex(quarters).fillna(0.0)      # reference quarter -> 0
    se = res["se"].reindex(quarters)                      # NaN at the reference (no CI)
    return pd.DataFrame({"q": quarters, "beta": beta.values, "se": se.values})


def make(lang: str) -> None:
    L = LABELS[lang]
    d = _coefficients()
    x = np.arange(len(d))
    lo = d["beta"] - 1.96 * d["se"]; hi = d["beta"] + 1.96 * d["se"]
    onset = d.index[d["q"] == "2018Q3"][0] if (d["q"] == "2018Q3").any() else None

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax.fill_between(x, lo, hi, color=BLUE_TINT, alpha=0.9, linewidth=0)
    ax.plot(x, d["beta"], color=BLUE, lw=2.0, marker="o", ms=4)
    if onset is not None:
        ax.axvline(onset, color=ORANGE, lw=1.5, ls=":")
        ax.annotate(L["onset"], xy=(onset, ax.get_ylim()[1]), xytext=(onset + 0.3, ax.get_ylim()[1]),
                    color=ORANGE, fontsize=9, va="top")
    # sparse quarter ticks (annual)
    ticks = [i for i, q in enumerate(d["q"]) if q.endswith("Q1")]
    ax.set_xticks(ticks); ax.set_xticklabels([d["q"].iloc[i][:4] for i in ticks])
    ax.set_xlabel(L["x"]); ax.set_ylabel(L["y"]); ax.set_title(L["title"], loc="left")
    save(fig, "fig_event_study", lang)


if __name__ == "__main__":
    apply_style()
    for lang in LANGS:
        make(lang)
    print("fig_event_study written for", ", ".join(LANGS))
