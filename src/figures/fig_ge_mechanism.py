"""Figure 3 — GE mechanism reconciliation: DCP does not block rebalancing; capital controls do.

Two panels from the dynamic counterfactual: the FX rebalancing efficiency (net exports bought
per unit of output-gap cost by an engineered weaker dollar) is roughly flat in the dollar-
invoicing friction theta_dollar, but collapses in the capital-controls wedge chi. This is the
figure that reconciles the thesis: dominant-currency pricing shapes tariff incidence (H1),
while the closed capital account is what binds exchange-rate rebalancing.
"""
from __future__ import annotations

from src.analysis.dsge_counterfactual import theta_dollar_grid, chi_grid
from src.figures.style import apply_style, save, BLUE, ORANGE, MUTED, LANGS
import matplotlib.pyplot as plt

LABELS = {
    "en": {"title": "Dollar invoicing does not block rebalancing; capital controls do",
           "x_td": "Dollar-invoicing friction", "x_chi": "Capital-controls wedge",
           "y": "FX rebalancing efficiency", "flat": "roughly flat", "coll": "collapses",
           "obs": "observed"},
    "fr": {"title": "L'invoicing dollar ne bloque pas le reequilibrage ; les controles de capitaux si",
           "x_td": "Friction d'invoicing dollar", "x_chi": "Coin de controle des capitaux",
           "y": "Efficacite de reequilibrage FX", "flat": "quasi plate", "coll": "s'effondre",
           "obs": "observe"},
}


def make(lang: str) -> None:
    L = LABELS[lang]
    td = theta_dollar_grid(); ch = chi_grid()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.6))

    ax1.plot(td["theta_dollar"], td["efficiency"], color=BLUE, lw=2.0, marker="o", ms=4)
    ax1.set_xlabel(L["x_td"]); ax1.set_ylabel(L["y"])
    ax1.set_ylim(0, max(ch["efficiency"].max(), td["efficiency"].max()) * 1.1)
    ax1.annotate(L["flat"], xy=(0.5, td["efficiency"].mean()),
                 xytext=(0.15, td["efficiency"].mean() + 0.4), color=BLUE, fontsize=9)
    obs = td[td["theta_dollar"] == 0.95]
    if len(obs):
        ax1.scatter(obs["theta_dollar"], obs["efficiency"], color=ORANGE, zorder=5, s=36)
        ax1.annotate(L["obs"], xy=(0.95, obs["efficiency"].iloc[0]), xytext=(0.6, obs["efficiency"].iloc[0] - 0.6),
                     color=ORANGE, fontsize=9)

    ax2.plot(ch["chi"], ch["efficiency"], color=BLUE, lw=2.0, marker="o", ms=4)
    ax2.set_xlabel(L["x_chi"])
    ax2.set_ylim(0, max(ch["efficiency"].max(), td["efficiency"].max()) * 1.1)
    ax2.annotate(L["coll"], xy=(2.5, ch["efficiency"].iloc[-1]),
                 xytext=(1.6, ch["efficiency"].max() * 0.55), color=BLUE, fontsize=9)

    fig.tight_layout()  # no internal title (paper-writing-rules 2.2; title in the LaTeX caption)
    save(fig, "fig_ge_mechanism", lang)


if __name__ == "__main__":
    apply_style()
    for lang in LANGS:
        make(lang)
    print("fig_ge_mechanism written for", ", ".join(LANGS))
