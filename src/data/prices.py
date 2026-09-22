"""BLS import price indexes by origin (China), by NAICS — the pass-through price outcome.

Reads the versioned public table data/import_price_china_bls.csv (pulled by the gitignored
data_download/pull_bls_prices.py from the keyless BLS API v1) and returns a tidy monthly
panel, attaching the ICIO industry (via the NAICS concordance) so prices merge with the
tariff and quantity dimensions of the product/sector panel.

These are dollar border-price indexes of Chinese-origin imports. Under H1, they should show
little tariff-driven rise (the tariff falls on US importers, not on the dollar border price).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.paths import DATA_DIR
from src.data.concordance import load_crosswalk, naics_to_icio

PRICES_PATH = DATA_DIR / "import_price_china_bls.csv"


def load_china_import_prices(path: str | Path = PRICES_PATH) -> pd.DataFrame:
    """Tidy monthly China import price panel with a datadate and mapped ICIO industry.

    `naics` == 'TOT' is the all-industries aggregate (icio_industry left NA). Detailed NAICS
    codes are mapped to their ICIO industry; 2-digit aggregates (31/32/33) stay unmapped.
    """
    df = pd.read_csv(path, dtype={"naics": "string"})
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1)).dt.date
    xw = load_crosswalk()
    df["icio_industry"] = df["naics"].map(
        lambda n: None if n == "TOT" else naics_to_icio(n, xw))
    return df.sort_values(["naics", "year", "month"]).reset_index(drop=True)


if __name__ == "__main__":
    p = load_china_import_prices()
    print(f"rows={len(p)} series={p['series_id'].nunique()} "
          f"years={p['year'].min()}-{p['year'].max()}")
    mapped = p[p["icio_industry"].notna()]
    print(f"NAICS series mapped to an ICIO industry: {mapped['series_id'].nunique()} "
          f"(industries: {sorted(mapped['icio_industry'].unique())})")


def china_import_price_total_quarterly(path: str | Path = PRICES_PATH) -> pd.Series:
    """All-industries import price index of Chinese-origin goods (NAICS 'TOT'), quarterly mean."""
    df = load_china_import_prices(path)
    df = df[df["naics"] == "TOT"]
    q = pd.PeriodIndex(pd.to_datetime(dict(year=df["year"], month=df["month"], day=1)), freq="Q")
    return df.groupby(q)["value"].mean().rename("price")
