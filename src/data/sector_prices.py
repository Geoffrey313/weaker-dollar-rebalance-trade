"""Border-price pass-through panel: BLS China import price index merged with the Census
NAICS effective tariff, on NAICS x month.

Merges data/import_price_china_bls.csv (BLS EIUCOCHN* price index by NAICS) with
data/china_imports_naics.parquet (Census China imports by NAICS -> effective tariff =
duties/value). Both are keyed by NAICS x month, so no HS<->NAICS bridge is needed.

The price index is the DOLLAR BORDER price of Chinese-origin imports (pre-duty). Under H1 it
should NOT respond to the tariff (the tariff is not offset in the border price; it passes to
the US importer). This panel is the price side of the H1 decomposition; the value/quantity
side is the sector event study.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import DATA_DIR

PRICES_PATH = DATA_DIR / "import_price_china_bls.csv"
NAICS_IMPORTS_PATH = DATA_DIR / "china_imports_naics.parquet"


def load_price_tariff_panel() -> pd.DataFrame:
    """NAICS x month panel with log border price and effective tariff, merged on NAICS."""
    price = pd.read_csv(PRICES_PATH, dtype={"naics": "string"})
    price = price.rename(columns={"value": "price_index"})

    imp = pd.read_parquet(NAICS_IMPORTS_PATH)
    imp["naics"] = imp["naics"].astype("string")
    imp = imp[imp["value_usd"] > 0].copy()
    imp["effective_tariff"] = imp["duties_usd"] / imp["value_usd"]

    m = price.merge(imp[["naics", "year", "month", "effective_tariff", "value_usd"]],
                    on=["naics", "year", "month"], how="inner")
    m = m[(m["effective_tariff"] >= 0) & (m["effective_tariff"] <= 1) & (m["price_index"] > 0)].copy()
    m["log_price"] = np.log(m["price_index"])
    m["period"] = m["year"] * 100 + m["month"]
    return m.sort_values(["naics", "period"]).reset_index(drop=True)


if __name__ == "__main__":
    p = load_price_tariff_panel()
    print(f"rows={len(p)} naics={p['naics'].nunique()} months={p['year'].min()}-{p['year'].max()}")
    print("matched NAICS:", sorted(p["naics"].unique()))
