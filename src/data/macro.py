"""Macro and bilateral-trade inputs of the parameter estimation (public data).

Reads the versioned tables written by the gitignored data_download/pull_macro_inputs.py:

  - us_china_trade_annual.csv: annual US goods exports to and general imports from China,
    Census Bureau, dollars; used for the import share, bilateral openness, and initial imbalance.
  - fx_cny_usd_monthly.csv: renminbi per dollar, monthly average (FRED EXCHUS); used for the
    exchange-rate pass-through that disciplines the dollar-invoicing friction.
  - us_gdp_annual.csv: US nominal GDP, billions of dollars (FRED GDPA, BEA).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.paths import DATA_DIR


def load_bilateral_trade() -> pd.DataFrame:
    """Annual US-China goods trade, billions of dollars, indexed by year."""
    df = pd.read_csv(DATA_DIR / "us_china_trade_annual.csv")
    df["exports_bn"] = df["exports_usd"] / 1e9
    df["imports_bn"] = df["imports_usd"] / 1e9
    return df.set_index("year")[["exports_bn", "imports_bn"]]


def load_gdp() -> pd.Series:
    """US nominal GDP, billions of dollars, indexed by year."""
    df = pd.read_csv(DATA_DIR / "us_gdp_annual.csv", parse_dates=["date"])
    return df.set_index(df["date"].dt.year)["gdp_usd_bn"].rename("gdp_bn")


def load_renminbi_value_quarterly() -> pd.Series:
    """Log value of the renminbi in dollars, e = -ln(CNY per USD), quarterly average."""
    df = pd.read_csv(DATA_DIR / "fx_cny_usd_monthly.csv", parse_dates=["date"])
    q = df.groupby(df["date"].dt.to_period("Q"))["cny_per_usd"].mean()
    return (-np.log(q)).rename("e")
