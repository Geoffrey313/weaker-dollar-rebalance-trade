"""US imports from China by HS4 x month: effective tariff, value, and unit value.

Reads the versioned public table data/china_imports_hs4.parquet (pulled from the Census
International Trade API by the gitignored data_download/pull_census_imports.py) and derives
the product-level pass-through panel inputs:

  - effective_tariff = duties_usd / value_usd  (the product-level tariff tau_{p,t} that
    identifies the sector panel; the Bown series are only trade-weighted aggregates)
  - unit_value = value_usd / qty1  (a price proxy, comparable WITHIN an HS4 over time; the
    quantity unit varies across HS4, so it is not comparable across products in levels)
  - log value and log quantity as the traded-flow outcomes

Panel unit p = HS4 product, monthly. Used with product and time fixed effects (eq:sector_panel).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import DATA_DIR

IMPORTS_PATH = DATA_DIR / "china_imports_hs4.parquet"


def load_china_imports_hs4(path: str | Path = IMPORTS_PATH) -> pd.DataFrame:
    """Tidy HS4 x month China-imports panel with effective tariff, unit value and logs."""
    df = pd.read_parquet(path)
    df = df[df["value_usd"] > 0].copy()
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1)).dt.date
    df["effective_tariff"] = df["duties_usd"] / df["value_usd"]
    df["unit_value"] = np.where(df["qty1"] > 0, df["value_usd"] / df["qty1"], np.nan)
    df["log_value"] = np.log(df["value_usd"])
    df["log_qty"] = np.nan
    qty_pos = df["qty1"] > 0
    df.loc[qty_pos, "log_qty"] = np.log(df.loc[qty_pos, "qty1"])
    return df.sort_values(["hs4", "year", "month"]).reset_index(drop=True)


if __name__ == "__main__":
    df = load_china_imports_hs4()
    print(f"rows={len(df)} hs4={df['hs4'].nunique()} months={df['year'].min()}-{df['year'].max()}")
    # Effective tariff on Chinese imports should jump across the 2018-2019 waves.
    monthly_sums = df.groupby(["year", "month"], as_index=False)[["duties_usd", "value_usd"]].sum()
    monthly_sums["agg_effective_tariff"] = monthly_sums["duties_usd"] / monthly_sums["value_usd"]
    print("\nAggregate effective tariff on China imports (duties/value), by quarter:")
    q = monthly_sums
    q["ym"] = q["year"].astype(str) + "-" + q["month"].astype(str).str.zfill(2)
    for _, r in q[q["month"].isin([1, 4, 7, 10])].iterrows():
        print(f"  {r['ym']}: {r['agg_effective_tariff']:.4f}")


def aggregate_effective_tariff_quarterly(path: str | Path = IMPORTS_PATH) -> pd.Series:
    """Aggregate effective tariff on US imports from China by quarter: sum of duties over sum of
    customs value across all HS4 products (a trade-weighted realized rate)."""
    df = load_china_imports_hs4(path)
    q = pd.PeriodIndex(pd.to_datetime(dict(year=df["year"], month=df["month"], day=1)), freq="Q")
    agg = df.groupby(q)[["duties_usd", "value_usd"]].sum()
    return (agg["duties_usd"] / agg["value_usd"]).rename("tariff")
