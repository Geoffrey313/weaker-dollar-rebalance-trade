"""ChinaExp: firm China-sales exposure from Compustat geographic segments (GEOSEG).

Reads the transformed GEOSEG panel (data/wrds/compustat_geoseg.parquet — licensed, gitignored,
available on request; acquired by data_download/pull_compustat_segments.py) and builds, per
firm-year, ChinaExp = China-segment sales / total geographic-segment sales.

This is the FALLBACK / robustness exposure: it measures where a firm SELLS (downstream China
demand / China risk), NOT tariff incidence and NOT upstream input sourcing. Segment names are
free-form, so a China segment is one whose name mentions China but not an "excluding China"
qualifier. Fixed at the pre-episode base year for use as a treatment.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.common.paths import PROJECT_ROOT
from src.common import config as C

GEOSEG_PATH = PROJECT_ROOT / "data" / "wrds" / "compustat_geoseg.parquet"

_EXCLUDE = re.compile(r"exclud|except|excl\.|excl |ex\.", re.IGNORECASE)
_CHINA = re.compile(r"china", re.IGNORECASE)


def _is_china_segment(name: str) -> bool:
    s = str(name)
    return bool(_CHINA.search(s)) and not _EXCLUDE.search(s)


def build_china_sales_exposure(path: str | Path = GEOSEG_PATH) -> pd.DataFrame:
    """Per firm-year ChinaExp = China-segment sales / total GEOSEG sales (sales>0)."""
    df = pd.read_parquet(path)
    df = df[df["sales"] > 0].copy()
    df["year"] = pd.to_datetime(df["datadate"]).dt.year
    df["is_china"] = df["snms"].map(_is_china_segment)
    grp = df.groupby(["gvkey", "year"])
    out = grp.apply(lambda g: pd.Series({
        "china_sales": g.loc[g["is_china"], "sales"].sum(),
        "total_geoseg_sales": g["sales"].sum(),
    })).reset_index()
    out["china_sales_exposure"] = out["china_sales"] / out["total_geoseg_sales"]
    return out


def base_year_exposure(path: str | Path = GEOSEG_PATH) -> pd.DataFrame:
    """ChinaExp fixed at the pre-episode base year (config.BASE_YEAR), one row per firm."""
    allyrs = build_china_sales_exposure(path)
    base = allyrs[allyrs["year"] == C.BASE_YEAR][["gvkey", "china_sales_exposure"]]
    return base.rename(columns={"china_sales_exposure": "china_sales_exposure_base"})


if __name__ == "__main__":
    allyrs = build_china_sales_exposure()
    base = allyrs[allyrs["year"] == C.BASE_YEAR]
    has = base[base["china_sales_exposure"] > 0]
    print(f"firm-years={len(allyrs)} | base {C.BASE_YEAR}: {len(base)} firms, "
          f"{len(has)} with a China sales segment (>0)")
    print(f"ChinaExp (base, >0) median={has['china_sales_exposure'].median():.3f} "
          f"mean={has['china_sales_exposure'].mean():.3f}")
