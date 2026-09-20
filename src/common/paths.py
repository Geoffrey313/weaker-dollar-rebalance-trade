"""Portable paths, resolved relative to the repository root — no machine paths in code.

Layout:
    data/               transformed, versioned inputs (public)
    data/raw/           downloaded raw tables (gitignored, e.g. OECD ICIO) — not published
"""
from __future__ import annotations

from pathlib import Path

# src/common/paths.py -> parents[2] == repository root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"  # gitignored (see .gitignore: data/raw/)


def ensure_raw() -> Path:
    """Create data/raw/ if needed and return it (holds downloaded ICIO/WIOD files)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR
