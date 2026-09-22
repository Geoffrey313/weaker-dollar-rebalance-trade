"""Figure: the deepening of United States-China linkages, 2000 versus 2024.

Two panels on a common scale show the three bilateral channels between the United States and
China, in billions of current dollars: goods trade and services trade (annual flows) and the
direct-investment position (a year-end stock, historical-cost basis). Flows into the United
States, from China, extend to the left; flows out of the United States, to China, extend to the
right. The figure motivates the scale of the imbalance the paper studies: every channel expands
by roughly an order of magnitude between 2000 and 2024, and the goods deficit widens fastest.

Data: data/us_china_flows.csv (Census, Trade in Goods with China; BEA, international services and
direct-investment positions). China excludes Hong Kong and Macau.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from src.common.paths import DATA_DIR
from src.figures.style import (BRICK, LANGS, NAVY, TEXT_WIDTH, apply_style, number_labels,
                               reference_line, save)

TYPES = ["goods", "services", "investment"]
YEARS = [2000, 2024]
LABELS = {
    "en": {"goods": "Goods trade", "services": "Services trade",
           "investment": "Investment position", "out": "US to China", "in": "China to US"},
    "fr": {"goods": "Commerce de biens", "services": "Commerce de services",
           "investment": "Position d'investissement", "out": "US vers Chine", "in": "Chine vers US"},
}


def flows() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "us_china_flows.csv")


def _fmt(v: float, lang: str) -> str:
    nd = 0 if v >= 100 else (2 if v < 1 else 1)
    return number_labels([v], nd, lang)[0]


def make(lang: str) -> None:
    L = LABELS[lang]
    df = flows()
    xmax = float(df["value_bn"].max()) * 1.22
    ypos = {t: len(TYPES) - i for i, t in enumerate(TYPES)}

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.8), sharey=True, layout="constrained")
    for ax, year in zip(axes, YEARS):
        d = df[df["year"] == year].set_index(["flow_type", "direction"])["value_bn"]
        for t in TYPES:
            y = ypos[t]
            out, inn = float(d[(t, "us_to_china")]), float(d[(t, "china_to_us")])
            ax.barh(y, out, color=NAVY, height=0.6, zorder=3)
            ax.barh(y, -inn, color=BRICK, height=0.6, zorder=3)
            ax.text(out + xmax * 0.02, y, _fmt(out, lang), va="center", ha="left",
                    fontsize=6.5, color=NAVY)
            ax.text(-inn - xmax * 0.02, y, _fmt(inn, lang), va="center", ha="right",
                    fontsize=6.5, color=BRICK)
        reference_line(ax, 0.0, axis="v")
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(0.4, len(TYPES) + 1.0)
        ax.set_yticks(list(ypos.values()), [L[t] for t in TYPES])
        ax.set_xticks([])
        ax.tick_params(left=False)
        ax.annotate(str(year), (0, len(TYPES) + 0.7), ha="center", va="center", fontsize=10)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
    fig.legend(handles=[Patch(color=NAVY, label=L["out"]), Patch(color=BRICK, label=L["in"])],
               loc="outside lower center", ncol=2, fontsize=7.5)
    save(fig, "fig_flows", lang)


if __name__ == "__main__":
    apply_style()
    for lang in LANGS:
        make(lang)
    print("fig_flows written for", ", ".join(LANGS))
