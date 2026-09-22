"""Figure: the effective tariff and the dollar border price of US imports from China.

Two stacked panels on a common quarterly axis (no dual axis): the aggregate effective tariff
(calculated duties over customs value) climbs across the 2018-2019 waves, while the BLS import
price index of all Chinese-origin goods, in dollars and before duties, does not fall. It
illustrates the price side of H1: the tariff is not offset in the dollar border price.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.prices import china_import_price_total_quarterly
from src.data.sector_imports import aggregate_effective_tariff_quarterly
from src.figures.style import LANGS, NAVY, TEXT_WIDTH, apply_style, localize, reference_line, save

LABELS = {
    "en": {"x": "Quarter", "y_tar": r"Effective tariff, $\tau_t$",
           "y_price": "Import price index"},
    "fr": {"x": "Trimestre", "y_tar": r"Tarif effectif, $\tau_t$",
           "y_price": "Indice des prix à l’importation"},
}


def series() -> pd.DataFrame:
    df = pd.concat([aggregate_effective_tariff_quarterly(), china_import_price_total_quarterly()],
                   axis=1).dropna()
    df = df[(df.index.year >= 2015) & (df.index.year <= 2021)]
    df["label"] = [f"{q.year}Q{q.quarter}" for q in df.index]
    return df.reset_index(drop=True)


def make(lang: str) -> None:
    L = LABELS[lang]
    d = series()
    x = np.arange(len(d))
    onset = int(np.flatnonzero(d["label"] == "2018Q3")[0])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(TEXT_WIDTH, 3.9), sharex=True,
                                   layout="constrained")
    for ax, col, label in ((ax1, "tariff", L["y_tar"]), (ax2, "price", L["y_price"])):
        reference_line(ax, onset - 0.5, axis="v")
        ax.plot(x, d[col], color=NAVY, marker="o", ms=3.0)
        ax.set_ylabel(label)
        localize(ax, lang)
    ticks = [i for i, lab in enumerate(d["label"]) if lab.endswith("Q1")]
    ax2.set_xticks(ticks, [d["label"].iloc[i][:4] for i in ticks])
    ax2.set_xlabel(L["x"])
    fig.align_ylabels((ax1, ax2))
    save(fig, "fig_h1_decomposition", lang)


if __name__ == "__main__":
    apply_style()
    for lang in LANGS:
        make(lang)
    print("fig_h1_decomposition written for", ", ".join(LANGS))
