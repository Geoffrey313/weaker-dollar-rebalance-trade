"""Shared figure style and save helper (Phase 6 figures).

Applies a compact, print-ready, colorblind-safe style following the dataviz method: a validated
categorical pair (blue #2a78d6, orange #eb6834 — CVD dE 24.7, contrast >= 3:1), thin marks,
recessive grid/axes, text in ink (never the series color), no dual-axis. Figures are generated
by code (never hand-edited) into the manuscript figure folders, in the manuscript language.
"""
from __future__ import annotations

import os
from pathlib import Path

from src.common.paths import PROJECT_ROOT

os.environ.setdefault("MPLCONFIGDIR", "/tmp/weaker-dollar-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette (dataviz reference, light surface).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BLUE_TINT = "#cde2fb"
ORANGE_TINT = "#f8d3c2"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3de"

LANGS = ("en", "fr")


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
        "lines.linewidth": 2.0, "figure.dpi": 150,
    })


def figures_dir(lang: str) -> Path:
    d = PROJECT_ROOT / "manuscript" / lang / "ssrn" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(fig, name: str, lang: str) -> Path:
    """Save a figure as PDF (for LaTeX) and PNG (for quick viewing) in the manuscript dir."""
    out = figures_dir(lang)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    png = out / f"{name}.png"
    fig.savefig(png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return png
