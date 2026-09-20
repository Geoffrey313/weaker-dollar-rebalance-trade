"""Figure 2 — H1 decomposition: the tariff rises, the dollar border price stays flat.

Two stacked panels on a common time axis (no dual-axis): the effective tariff on US imports
from China climbs across the 2018-2019 waves, while the BLS dollar border-price index of those
imports is flat. No border-price offset -> the tariff passes to US importers; the adjustment is
in quantities/value (Figure 1), not in the price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.sector_imports import load_china_imports_hs4
from src.data.prices import load_china_import_prices
from src.figures.style import apply_style, save, BLUE, ORANGE, MUTED, LANGS
import matplotlib.pyplot as plt

LABELS = {
    "en": {"title": "The tariff rises, without a border-price offset",
           "x": "Quarter", "y_tar": "Effective tariff", "y_price": "Import price index (USD)",
           "onset": "2018 tariff waves"},
    "fr": {"title": "Le tarif monte, sans offset du prix frontiere",
           "x": "Trimestre", "y_tar": "Tarif effectif", "y_price": "Indice de prix a l'import (USD)",
           "onset": "vagues tarifaires 2018"},
}


def _series() -> pd.DataFrame:
    imp = load_china_imports_hs4()
    imp["qkey"] = imp["year"] * 10 + ((imp["month"] - 1) // 3 + 1)
    agg = imp.groupby("qkey", as_index=True).agg(d=("duties_usd", "sum"), v=("value_usd", "sum"))
    tar = (agg["d"] / agg["v"]).rename("tariff")
    pr = load_china_import_prices()
    pr = pr[pr["naics"] == "TOT"].copy()
    pr["qkey"] = pr["year"] * 10 + ((pr["month"] - 1) // 3 + 1)
    price = pr.groupby("qkey")["value"].mean().rename("price")
    df = pd.concat([tar, price], axis=1).dropna().reset_index()
    df = df[(df["qkey"] // 10).between(2015, 2021)]
    df["label"] = (df["qkey"] // 10).astype(str) + "Q" + (df["qkey"] % 10).astype(str)
    return df.reset_index(drop=True)


def make(lang: str) -> None:
    L = LABELS[lang]
    d = _series()
    x = np.arange(len(d))
    onset = d.index[d["label"] == "2018Q3"][0] if (d["label"] == "2018Q3").any() else None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    ax1.plot(x, d["tariff"], color=ORANGE, lw=2.0, marker="o", ms=3)
    ax1.set_ylabel(L["y_tar"])  # no internal title (paper-writing-rules 2.2; title in the LaTeX caption)
    ax2.plot(x, d["price"], color=BLUE, lw=2.0, marker="o", ms=3)
    ax2.set_ylabel(L["y_price"]); ax2.set_xlabel(L["x"])
    for ax in (ax1, ax2):
        if onset is not None:
            ax.axvline(onset, color=MUTED, lw=1.2, ls=":")
    if onset is not None:
        ax1.annotate(L["onset"], xy=(onset, ax1.get_ylim()[1]), xytext=(onset + 0.3, ax1.get_ylim()[1]),
                     color=MUTED, fontsize=9, va="top")
    ticks = [i for i, lab in enumerate(d["label"]) if lab.endswith("Q1")]
    ax2.set_xticks(ticks); ax2.set_xticklabels([d["label"].iloc[i][:4] for i in ticks])
    save(fig, "fig_h1_decomposition", lang)


if __name__ == "__main__":
    apply_style()
    for lang in LANGS:
        make(lang)
    print("fig_h1_decomposition written for", ", ".join(LANGS))
