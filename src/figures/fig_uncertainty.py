"""Figure: parameter uncertainty and the ranking of the two frictions (Monte Carlo).

Four panels from src.analysis.parameter_uncertainty, over the draws with a unique stable
equilibrium. Top: kernel densities of the percentage change in the rebalancing efficiency when
dollar invoicing (navy, solid) or the capital-controls wedge (brick, dashed) moves from zero to
its level in the draw; left with the estimated parameters only, right with all parameters.
Bottom left: the two effects draw by draw with all parameters; the ranking of H3 holds below
the diagonal, and the draws where it fails are marked. Bottom right: rank correlation of each
drawn parameter with the two effects and with their difference, sorted by the size of the last.
Effects beyond CLIP percent in absolute value are left out of the densities and pinned to the
edge of the scatter.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from src.analysis.parameter_uncertainty import drivers, run
from src.figures.latex import PARAMETER_SYMBOLS
from src.figures.style import (BRICK, BRICK_TINT, GREY_TINT, INK, LANGS, NAVY, NAVY_TINT,
                               TEXT_WIDTH, apply_style, localize, number_labels,
                               reference_line, save)

CLIP = 100.0
GRID = np.linspace(-CLIP, CLIP, 801)
SCATTER_X = (-40.0, 100.0)
SCATTER_Y = (-100.0, 40.0)
LABELS = {
    "en": {"inv": "Dollar invoicing", "wedge": "Capital controls",
           "x": r"Change in the efficiency $\Lambda$ (percent)", "dens": "Density",
           "sx": "Invoicing effect (percent)", "sy": "Capital-controls effect (percent)",
           "holds": "Ranking holds", "fails": "Ranking fails",
           "rho": "Rank correlation", "r_inv": "Invoicing\neffect",
           "r_wedge": "Capital-controls\neffect", "r_diff": "Difference"},
    "fr": {"inv": "Facturation en dollars", "wedge": "Contrôle des capitaux",
           "x": r"Variation de l’efficacité $\Lambda$ (pour cent)", "dens": "Densité",
           "sx": "Effet de la facturation (pour cent)",
           "sy": "Effet du contrôle des capitaux (pour cent)",
           "holds": "Classement vérifié", "fails": "Classement non vérifié",
           "rho": "Corrélation de rang", "r_inv": "Effet de la\nfacturation",
           "r_wedge": "Effet du contrôle\ndes capitaux", "r_diff": "Différence"},
}
SERIES = {"invoicing": (NAVY, NAVY_TINT, "-"), "wedge": (BRICK, BRICK_TINT, (0, (5, 2)))}


def usable(layer: str):
    df = run(layer)
    ok = df[df["determinate"] & df["well_defined"]].copy()
    for k in ("invoicing_effect", "wedge_effect"):
        ok[k] = 100 * ok[k]
    return ok


def _densities(ax, layer: str, L: dict) -> None:
    ok = usable(layer)
    for key, label in (("invoicing", L["inv"]), ("wedge", L["wedge"])):
        color, tint, ls = SERIES[key]
        values = ok[f"{key}_effect"].to_numpy()
        dens = gaussian_kde(values[np.abs(values) <= CLIP])(GRID)
        ax.fill_between(GRID, dens, color=tint, alpha=0.7, lw=0)
        ax.plot(GRID, dens, color=color, ls=ls, label=label)
        med = float(np.median(values))
        ax.plot([med, med], [0, np.interp(med, GRID, dens)], color=color, lw=0.8, ls=":")
    reference_line(ax, 0.0, axis="v")
    ax.set_xlim(-CLIP, CLIP)
    ax.set_ylim(0, None)
    ax.set_xlabel(L["x"])


def _scatter(ax, L: dict) -> None:
    ok = usable("full")
    x = np.clip(ok["invoicing_effect"], *SCATTER_X)
    y = np.clip(ok["wedge_effect"], *SCATTER_Y)
    holds = ok["ranking_holds"].to_numpy()
    ax.scatter(x[holds], y[holds], s=1.2, color=NAVY, alpha=0.18, lw=0, rasterized=True)
    ax.scatter(x[~holds], y[~holds], s=5, color=BRICK, alpha=0.9, lw=0, rasterized=True)
    lo, hi = max(SCATTER_X[0], SCATTER_Y[0]), min(SCATTER_X[1], SCATTER_Y[1])
    ax.plot([lo, hi], [lo, hi], color=INK, lw=0.7)
    reference_line(ax, 0.0)
    reference_line(ax, 0.0, axis="v")
    ax.set_xlim(*SCATTER_X)
    ax.set_ylim(*SCATTER_Y)
    ax.set_xlabel(L["sx"])
    ax.set_ylabel(L["sy"])
    handles = [Line2D([], [], ls="none", marker="o", ms=3.5, color=NAVY, alpha=0.6, label=L["holds"]),
               Line2D([], [], ls="none", marker="o", ms=3.5, color=BRICK, label=L["fails"])]
    ax.legend(handles=handles, loc="upper right", handletextpad=0.3)


def _correlations(ax, L: dict, lang: str) -> None:
    d = drivers()
    d = d.reindex(d["rho_difference"].abs().sort_values().index)
    y = np.arange(len(d))
    for yi in y:
        ax.axhline(yi, color=GREY_TINT, lw=0.5, zorder=0)
    reference_line(ax, 0.0, axis="v")
    ax.plot(d["rho_invoicing_effect"], y, ls="none", marker="o", ms=4, color=NAVY, label=L["r_inv"])
    ax.plot(d["rho_wedge_effect"], y, ls="none", marker="s", ms=3.8, color=BRICK, label=L["r_wedge"])
    ax.plot(d["rho_difference"], y, ls="none", marker="D", ms=4, mfc="white", mec=INK, mew=0.9,
            label=L["r_diff"])
    ax.set_yticks(y, [f"${PARAMETER_SYMBOLS[k]}$" for k in d["parameter"]])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.6, 1.0)
    ticks = [-0.5, 0.0, 0.5, 1.0]
    ax.set_xticks(ticks, number_labels(ticks, 1, lang))
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel(L["rho"])
    ax.legend(loc="lower right", handletextpad=0.4, labelspacing=0.6, frameon=True,
              fancybox=False, edgecolor=GREY_TINT, framealpha=1.0, borderpad=0.5)


def make(lang: str) -> None:
    L = LABELS[lang]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 5.4), layout="constrained")
    _densities(axes[0, 0], "estimation", L)
    _densities(axes[0, 1], "full", L)
    axes[0, 0].set_ylabel(L["dens"])
    axes[0, 0].legend(loc="upper right")
    _scatter(axes[1, 0], L)
    _correlations(axes[1, 1], L, lang)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        localize(ax, lang)
    save(fig, "fig_uncertainty", lang)


if __name__ == "__main__":
    apply_style()
    for lang in LANGS:
        make(lang)
    print("fig_uncertainty written for", ", ".join(LANGS))
