"""The paper's baseline calibration.

Literature values (src.engine.calibration.LITERATURE) overridden by every parameter the
project's data identify (src.analysis.parameter_estimation): the import-demand elasticity, the
dollar-invoicing friction, the tariff persistence, the import share, bilateral openness, and
the initial imbalance. The remaining parameters stay calibrated and are varied in the
sensitivity analyses. Every analysis of the structural model uses this baseline.
"""
from __future__ import annotations

from functools import lru_cache

from src.analysis.parameter_estimation import estimates
from src.engine.calibration import LITERATURE, Params
from src.engine.model import replace


@lru_cache(maxsize=None)
def baseline() -> Params:
    return replace(LITERATURE, **estimates())
