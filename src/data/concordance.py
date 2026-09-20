"""ISIC Rev.4 (ICIO industry) <-> NAICS concordance, to attach China-input exposure to firms.

Compustat firms carry a NAICS code; ICIO industries use ISIC Rev.4. The crosswalk maps a
firm's NAICS (up to 6 digits) to its ICIO industry by LONGEST-PREFIX match against the
`naics_prefix` column of data/xwalk_isic4_naics.csv, so a 6-digit firm code resolves to the
most specific rule available (e.g. 325412 -> 3254 -> pharma D21, else 325 -> chemicals D20).

The shipped crosswalk is a starter covering agriculture, mining and manufacturing (where the
tariffs bite). It must be validated/extended against the official UN ISIC Rev.4 <-> NAICS 2017
correspondence before publication.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common import config as C
from src.common.paths import DATA_DIR

XWALK_PATH = DATA_DIR / "xwalk_isic4_naics.csv"


def load_crosswalk(path: str | Path = XWALK_PATH) -> pd.DataFrame:
    """Load the ISIC4<->NAICS crosswalk; `naics_prefix` kept as string for prefix matching."""
    df = pd.read_csv(path, dtype={"naics_prefix": str})
    df["naics_prefix"] = df["naics_prefix"].str.strip()
    return df


def naics_to_icio(naics: str | int | float, crosswalk: pd.DataFrame | None = None) -> str | None:
    """Map one NAICS code to its ICIO ISIC4 industry by longest-prefix match; None if unmapped."""
    if crosswalk is None:
        crosswalk = load_crosswalk()
    if naics is None or (isinstance(naics, float) and pd.isna(naics)):
        return None
    code = str(naics).split(".")[0].strip()  # tolerate floats like 325412.0
    mapping = dict(zip(crosswalk["naics_prefix"], crosswalk["icio_industry"]))
    for length in range(len(code), 1, -1):
        hit = mapping.get(code[:length])
        if hit is not None:
            return hit
    return None


def _exposure_for_join(exposure: pd.DataFrame, share_col: str) -> pd.DataFrame:
    """Return one industry-level exposure row per industry for firm joins."""
    if share_col not in exposure.columns:
        raise KeyError(f"Exposure table is missing required column {share_col!r}.")
    if "year" in exposure.columns:
        exposure = exposure[exposure["year"] == C.BASE_YEAR].copy()
        if exposure.empty:
            raise ValueError(f"Exposure table has no rows for BASE_YEAR={C.BASE_YEAR}.")
    if exposure["industry"].duplicated().any():
        dupes = sorted(exposure.loc[exposure["industry"].duplicated(), "industry"].unique())
        raise ValueError(f"Exposure table has duplicate industries after filtering: {dupes[:10]}")
    cols = ["industry", share_col]
    for optional in ("year", "china_origin_codes"):
        if optional in exposure.columns:
            cols.append(optional)
    return exposure[cols]


def attach_exposure(firms: pd.DataFrame, exposure: pd.DataFrame,
                    naics_col: str = "naics", share_col: str = "china_input_share") -> pd.DataFrame:
    """Attach the industry-level China-input share to a firm table via NAICS -> ICIO industry.

    `firms` must have `naics_col`; `exposure` must have columns 'industry' and `share_col`.
    If `exposure` is a long year-industry table, it is filtered to config.BASE_YEAR before
    joining. Returns `firms` with added columns icio_industry and the exposure columns.
    """
    xwalk = load_crosswalk()
    out = firms.copy()
    out["icio_industry"] = out[naics_col].map(lambda n: naics_to_icio(n, xwalk))
    join_exposure = _exposure_for_join(exposure, share_col)
    out = out.merge(join_exposure,
                    left_on="icio_industry", right_on="industry", how="left")
    return out.drop(columns=["industry"])


if __name__ == "__main__":
    xw = load_crosswalk()
    print(f"crosswalk rows: {len(xw)}, ICIO industries covered: {xw['icio_industry'].nunique()}")
    checks = {"334111": "C26", "325412": "C21", "325199": "C20", "336110": "C29", "311111": "C10T12"}
    for naics, expected in checks.items():
        got = naics_to_icio(naics, xw)
        status = "OK" if got == expected else "MISMATCH"
        print(f"  {naics} -> {got} (expected {expected}) [{status}]")
        assert got == expected, f"{naics}: got {got}, expected {expected}"
    print("self-test OK")
