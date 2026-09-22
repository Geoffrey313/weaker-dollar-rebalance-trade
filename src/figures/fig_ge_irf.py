"""Figure: impulse responses to an engineered weaker dollar, open versus closed capital account.

Four panels on a common quarterly horizon, from the solved general-equilibrium model
(src/engine/dsge): the renminbi value e, bilateral net exports n, the net foreign asset position
d, and the United States output gap y, each for an open account (chi = 0, solid) and a tightly
closed one (chi = 4, dashed) at the estimated dollar-invoicing friction. It shows the mechanism
behind H3: with a closed account the premium on the accumulated foreign position rises faster,
the appreciation reverses sooner, and net exports turn negative.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from src.analysis.baseline import baseline
from src.analysis.dsge_counterfactual import HORIZON
from src.engine.dsge import irf
from src.engine.model import replace
from src.figures.style import (BRICK, LANGS, NAVY, TEXT_WIDTH, apply_style, localize,
                               reference_line, save)

CHIS = (0.0, 4.0)
LABELS = {
    "en": {"e": r"Renminbi value, $e_t$", "nx": r"Net exports, $n_t$",
           "d": r"Net foreign assets, $d_t$", "y": r"Output gap, $y_t$", "x": "Quarters",
           "open": r"Open account, $\chi=0$", "closed": r"Closed account, $\chi=4$"},
    "fr": {"e": r"Valeur du renminbi, $e_t$", "nx": r"Exportations nettes, $n_t$",
           "d": r"Actifs extérieurs nets, $d_t$", "y": r"Écart de production, $y_t$",
           "x": "Trimestres", "open": r"Compte ouvert, $\chi=0$",
           "closed": r"Compte fermé, $\chi=4$"},
}
STYLES = {0.0: {"color": NAVY, "ls": "-"}, 4.0: {"color": BRICK, "ls": (0, (5, 2))}}


def responses() -> dict[float, object]:
    b = baseline()
    return {chi: irf("z", periods=HORIZON, p=replace(b, chi=chi)) for chi in CHIS}


def make(lang: str) -> None:
    L = LABELS[lang]
    paths = responses()
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 4.2), sharex=True, layout="constrained")
    for ax, var in zip(axes.ravel(), ("e", "nx", "d", "y")):
        reference_line(ax, 0.0)
        for chi in CHIS:
            ax.plot(paths[chi].index, paths[chi][var], **STYLES[chi],
                    label=L["open"] if chi == 0 else L["closed"])
        ax.set_ylabel(L[var])
        ax.set_xlim(0, HORIZON - 1)
        localize(ax, lang)
    for ax in axes[1]:
        ax.set_xlabel(L["x"])
    axes[0, 0].legend(loc="upper right")
    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    save(fig, "fig_ge_irf", lang)


if __name__ == "__main__":
    apply_style()
    for lang in LANGS:
        make(lang)
    print("fig_ge_irf written for", ", ".join(LANGS))
