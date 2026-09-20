"""Single deterministic entry point for the paper's empirical reproduction.

Runs the full chain from the shipped transformed data to the paper's figures and tables:
data loading -> structural engine -> analysis (panels, event studies) -> figures (EN and FR).

The pipeline is deterministic: same inputs, same outputs. Each produced artifact is checked
against a stored SHA-256 fingerprint so a broken change is caught. No randomness is left
unseeded and no timestamps enter the published numbers.

This is a scaffold: the stages below are placeholders to be implemented phase by phase
(see the development plan). Running it now prints the intended pipeline without side effects.
"""
from __future__ import annotations


def stage_data() -> None:
    """Load the transformed input data shipped in data/ (no download here)."""
    # src.data.* : Compustat panels, CRSP returns, ICIO China-input exposure, tariffs, FX, prices.
    raise NotImplementedError("data stage not yet implemented")


def stage_engine() -> None:
    """Calibrate and solve the two-country NOEM/DCP model; run counterfactuals."""
    # src.engine.* : calibration, log-linear solution, Ramsey planner, threshold statics.
    raise NotImplementedError("engine stage not yet implemented")


def stage_analysis() -> None:
    """Estimate the firm panel and the product/sector panel; event studies."""
    # src.analysis.* : eq:twfe_firm, eq:eventstudy_firm, eq:sector_panel.
    raise NotImplementedError("analysis stage not yet implemented")


def stage_figures() -> None:
    """Generate all figures and tables, in EN and FR, into the manuscript figure folders."""
    # src.figures.* : one language parameter; no hand-edited figures.
    raise NotImplementedError("figures stage not yet implemented")


PIPELINE = [
    ("data", stage_data),
    ("engine", stage_engine),
    ("analysis", stage_analysis),
    ("figures", stage_figures),
]


def main() -> None:
    print("Reproduction pipeline (scaffold). Stages, in order:")
    for name, _ in PIPELINE:
        print(f"  - {name}")
    print("\nNot yet implemented. Stages are built phase by phase (see the development plan).")


if __name__ == "__main__":
    main()
