"""Compustat US firm-level annual panel: reader + margins + China-input exposure join.

Reads the transformed panel pulled from WRDS (data/wrds/compustat_funda.parquet — licensed,
gitignored, available on request; acquired by data_download/pull_compustat.py) and produces
the analysis-ready firm panel: gross margin, EBIT margin and COGS ratio per firm-year, with
the industry-level China-input exposure attached via NAICS (see concordance.py).

Outcomes here are the US-importer-side incidence variables (margins, COGS). The exposure is
the FEASIBLE-version treatment: China-INPUT exposure, not tariff incidence (firm x HS).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.paths import PROJECT_ROOT, DATA_DIR
from src.data.concordance import attach_exposure

FUNDA_PATH = PROJECT_ROOT / "data" / "wrds" / "compustat_funda.parquet"
EXPOSURE_PATH = DATA_DIR / "china_input_exposure.parquet"


def load_firm_panel(path: str | Path = FUNDA_PATH) -> pd.DataFrame:
    """Read the Compustat annual panel and add margin outcomes. Keeps firm-years with sale>0."""
    df = pd.read_parquet(path)
    df = df[df["sale"] > 0].copy()
    df["gross_margin"] = (df["sale"] - df["cogs"]) / df["sale"]
    df["ebit_margin"] = df["ebit"] / df["sale"]
    df["cogs_to_sale"] = df["cogs"] / df["sale"]
    df["naics"] = df["naics"].astype("string")
    return df


def build_firm_exposure_panel(firm_path: str | Path = FUNDA_PATH,
                              exposure_path: str | Path = EXPOSURE_PATH) -> pd.DataFrame:
    """Firm panel with China-input exposure attached by NAICS -> ICIO industry (base year)."""
    firms = load_firm_panel(firm_path)
    exposure = pd.read_parquet(exposure_path)
    return attach_exposure(firms, exposure)


if __name__ == "__main__":
    panel = build_firm_exposure_panel()
    n_rows = len(panel)
    n_firms = panel["gvkey"].nunique()
    mapped = panel["icio_industry"].notna()
    with_exp = panel["china_input_share"].notna()
    print(f"firm-years: {n_rows} | firms: {n_firms}")
    print(f"mapped to an ICIO goods industry: {mapped.sum()} rows "
          f"({mapped.mean():.1%}), {panel.loc[mapped, 'gvkey'].nunique()} firms")
    print(f"with China-input exposure: {with_exp.sum()} rows "
          f"({panel.loc[with_exp, 'gvkey'].nunique()} firms)")
    print("\nfirm-years by ICIO industry (goods firms), top 12:")
    print(panel.loc[mapped, "icio_industry"].value_counts().head(12).to_string())
    print("\nexposure x median gross margin (goods firms, 2017 base):")
    g = (panel[mapped & (panel["fyear"] == 2017)]
         .groupby("icio_industry")
         .agg(n=("gvkey", "nunique"),
              china_input_share=("china_input_share", "first"),
              med_gross_margin=("gross_margin", "median"))
         .sort_values("china_input_share", ascending=False))
    print(g.head(12).to_string())
