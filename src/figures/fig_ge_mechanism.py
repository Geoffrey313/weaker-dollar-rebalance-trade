"""Figure: rebalancing efficiency against the two frictions in the general-equilibrium model.

Two panels from the dynamic counterfactual (src.analysis.dsge_counterfactual) at the
data-disciplined baseline. Left: the efficiency (discounted net exports bought per unit of
discounted output-gap cost by an engineered weaker dollar) over the dollar-invoicing friction
theta, with the estimate marked and its 95 percent confidence interval shaded. Right: the
efficiency over the capital-controls wedge chi at the estimated friction. It illustrates H3: the
efficiency falls modestly with invoicing and steeply with the wedge.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from src.analysis.dsge_counterfactual import chi_grid, theta_dollar_grid
from src.analysis.parameter_estimation import estimate_passthrough
from src.figures.style import (BRICK, GREY_TINT, LANGS, NAVY, TEXT_WIDTH, apply_style, localize,
                               save)

LABELS = {
    "en": {"x_td": r"Dollar-invoicing friction, $\theta$",
           "x_chi": r"Capital-controls wedge, $\chi$",
           "y": r"Rebalancing efficiency, $\Lambda$"},
    "fr": {"x_td": r"Friction de facturation en dollars, $\theta$",
           "x_chi": r"Coin de contrôle des capitaux, $\chi$",
           "y": r"Efficacité de rééquilibrage, $\Lambda$"},
}


def make(lang: str) -> None:
    L = LABELS[lang]
    td, ch = theta_dollar_grid(), chi_grid()
    pt = estimate_passthrough()
    top = max(ch["efficiency"].max(), td["efficiency"].max()) * 1.08
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.7), sharey=True,
                                   layout="constrained")

    ax1.axvspan(pt["theta"] - 1.96 * pt["se"], pt["theta"] + 1.96 * pt["se"],
                color=GREY_TINT, lw=0, zorder=0)
    ax1.plot(td["theta_dollar"], td["efficiency"], color=NAVY, marker="o", ms=3.2, zorder=2)
    est = td[td["estimated"]].iloc[0]
    ax1.plot([est["theta_dollar"]], [est["efficiency"]], ls="none", marker="o", ms=6.5,
             mfc="white", mec=NAVY, mew=1.2, zorder=3)
    ax1.set_xlabel(L["x_td"])
    ax1.set_ylabel(L["y"])
    ax1.set_xlim(-0.03, 1.02)

    ax2.plot(ch["chi"], ch["efficiency"], color=BRICK, ls=(0, (5, 2)), marker="s", ms=3.2)
    ax2.set_xlabel(L["x_chi"])
    ax2.set_ylim(0, top)
    for ax in (ax1, ax2):
        localize(ax, lang)
    save(fig, "fig_ge_mechanism", lang)


if __name__ == "__main__":
    apply_style()
    for lang in LANGS:
        make(lang)
    print("fig_ge_mechanism written for", ", ".join(LANGS))
