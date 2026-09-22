"""Figure: the appreciation required by the impact bound across parameter draws (H2).

Kernel density of the base-10 logarithm of the real appreciation of the renminbi that closes
the initial imbalance in the impact bound, b/R, over the Monte Carlo draws of
src.analysis.parameter_uncertainty: with the estimated parameters only (solid) and with all
parameters (dashed). The shaded band is the range of the tolerated appreciation drawn in the
Monte Carlo and the vertical line its baseline value. No draw of either layer falls inside it.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from src.analysis.parameter_uncertainty import CALIBRATED_RANGES, run
from src.analysis.baseline import baseline
from src.figures.style import (GREY, GREY_TINT, LANGS, NAVY, NAVY_TINT, TEXT_WIDTH, apply_style,
                               localize, number_labels, save)

LOG_GRID = np.linspace(1.0, 4.5, 701)       # 10 to about 30,000 percent
TICKS = [10, 100, 1_000, 10_000]
LABELS = {
    "en": {"x": "Required real appreciation of the renminbi (percent, logarithmic scale)",
           "y": "Density", "est": "Estimated parameters only", "full": "All parameters"},
    "fr": {"x": "Appréciation réelle requise du renminbi (pour cent, échelle logarithmique)",
           "y": "Densité", "est": "Paramètres estimés seuls", "full": "Tous les paramètres"},
}


def make(lang: str) -> None:
    L = LABELS[lang]
    _, lo, hi = CALIBRATED_RANGES["max_depreciation"]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.4), layout="constrained")
    ax.axvspan(np.log10(100 * lo), np.log10(100 * hi), color=GREY_TINT, lw=0, zorder=0)
    ax.axvline(np.log10(100 * baseline().max_depreciation), color=GREY, lw=0.8, zorder=1)
    for layer, ls, fill in (("estimation", "-", True), ("full", (0, (5, 2)), False)):
        values = np.log10(100 * run(layer)["required_deprec"].to_numpy())
        dens = gaussian_kde(values)(LOG_GRID)
        if fill:
            ax.fill_between(LOG_GRID, dens, color=NAVY_TINT, alpha=0.7, lw=0)
        ax.plot(LOG_GRID, dens, color=NAVY, ls=ls, label=L["est" if layer == "estimation" else "full"])
    ax.set_xlim(LOG_GRID[0], LOG_GRID[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel(L["x"])
    ax.set_ylabel(L["y"])
    ax.legend(loc="upper right")
    localize(ax, lang)
    ax.set_xticks(np.log10(TICKS), number_labels(TICKS, 0, lang, group=True))
    save(fig, "fig_uncertainty_bound", lang)


if __name__ == "__main__":
    apply_style()
    for lang in LANGS:
        make(lang)
    print("fig_uncertainty_bound written for", ", ".join(LANGS))
