"""CRSP monthly stock returns for US Compustat firms (linked via CCM) — the market outcome.

Reads the transformed panel pulled from WRDS (data/wrds/crsp_returns_monthly.parquet — licensed,
gitignored, available on request; acquired by data_download/pull_crsp_returns.py) and returns a
tidy monthly returns panel keyed by gvkey (and permno), with market capitalization.

Used for the event-window stock-return outcome (H4: market's anticipation of the tariff shock).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.paths import PROJECT_ROOT

RETURNS_PATH = PROJECT_ROOT / "data" / "wrds" / "crsp_returns_monthly.parquet"


def load_returns_monthly(path: str | Path = RETURNS_PATH) -> pd.DataFrame:
    """Tidy monthly CRSP returns with a date and market cap. `ret` is the holding-period return.

    CRSP `prc` is negative when it is a bid/ask average, so market cap uses |prc| * shrout
    (shrout is in thousands -> market cap in USD thousands).
    """
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df["mktcap"] = df["prc"].abs() * df["shrout"]
    return df.sort_values(["gvkey", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    df = load_returns_monthly()
    print(f"rows={len(df)} firms={df['gvkey'].nunique()} permnos={df['permno'].nunique()} "
          f"dates={df['date'].min()}..{df['date'].max()}")
    print(f"non-missing returns: {df['ret'].notna().mean():.1%}")
