"""US-China tariff timeline from Chad Bown / PIIE (public, redistributable).

Parses the raw PIIE workbook (data/raw/bown_us_china_tariffs.xlsx, gitignored, fetched by
data_download/fetch_bown.py) into two versioned public tables:
  - data/tariffs_bown_timeline.csv : trade-weighted average tariff rates by dated action
    (US-on-China, China-on-US, and vs ROW), the event calendar for the 2018-2019 episode.
  - data/tariffs_bown_coverage.csv : share of trade subject to the tariffs, by date.

These are AGGREGATE (trade-weighted) series: the episode's event dates and intensity path.
Product-level (HS) tariff variation for the sector panel comes separately from Census
effective duties (duties / customs value by HS x China x month); see the data audit.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.paths import PROJECT_ROOT, DATA_DIR

RAW_XLSX = PROJECT_ROOT / "data" / "raw" / "bown_us_china_tariffs.xlsx"

TIMELINE_COLS = {
    "Tariff action": "tariff_action",
    "Date": "date",
    "Chinese tariffs on ROW exports": "cn_tariff_on_row",
    "Chinese tariffs on US exports": "cn_tariff_on_us",
    "US tariffs on Chinese exports": "us_tariff_on_cn",
    "US tariffs on ROW exports": "us_tariff_on_row",
}


def _find_header(sheet: pd.DataFrame, marker: str) -> int:
    """Row index whose cells contain `marker` (the real header row inside a titled sheet)."""
    mask = sheet.apply(lambda r: r.astype(str).str.contains(marker, regex=False).any(), axis=1)
    hits = sheet.index[mask]
    if len(hits) == 0:
        raise ValueError(f"header marker {marker!r} not found")
    return int(hits[0])


def parse_timeline(xlsx: str | Path = RAW_XLSX) -> pd.DataFrame:
    xl = pd.ExcelFile(xlsx)
    hdr = _find_header(xl.parse("Panel a", header=None), "Tariff action")
    tbl = xl.parse("Panel a", header=hdr).dropna(how="all")
    tbl = tbl[[c for c in TIMELINE_COLS if c in tbl.columns]].rename(columns=TIMELINE_COLS)
    tbl = tbl[tbl["date"].notna()].copy()
    tbl["date"] = pd.to_datetime(tbl["date"]).dt.date
    return tbl.reset_index(drop=True)


def parse_coverage(xlsx: str | Path = RAW_XLSX) -> pd.DataFrame:
    xl = pd.ExcelFile(xlsx)
    hdr = _find_header(xl.parse("Panel b", header=None), "Date")
    tbl = xl.parse("Panel b", header=hdr).dropna(how="all")
    tbl.columns = [str(c).strip() for c in tbl.columns]
    ren = {c: ("date" if c == "Date"
               else "cn_exports_subject_us_tariff" if "Chinese exports" in c
               else "us_exports_subject_cn_tariff" if "US exports" in c else c)
           for c in tbl.columns}
    tbl = tbl.rename(columns=ren)
    tbl = tbl[tbl["date"].notna()].copy()
    tbl["date"] = pd.to_datetime(tbl["date"]).dt.date
    return tbl.reset_index(drop=True)


def build_and_save(xlsx: str | Path = RAW_XLSX) -> tuple[Path, Path]:
    tl, cov = parse_timeline(xlsx), parse_coverage(xlsx)
    tl_path, cov_path = DATA_DIR / "tariffs_bown_timeline.csv", DATA_DIR / "tariffs_bown_coverage.csv"
    tl.to_csv(tl_path, index=False)
    cov.to_csv(cov_path, index=False)
    return tl_path, cov_path


if __name__ == "__main__":
    tl_path, cov_path = build_and_save()
    tl = pd.read_csv(tl_path)
    print(f"timeline: {len(tl)} dated actions -> {tl_path.name}")
    key = tl[["date", "tariff_action", "us_tariff_on_cn"]].copy()
    print("\nUS trade-weighted avg tariff on China, key jumps:")
    print(key[key["us_tariff_on_cn"].diff().abs().fillna(1) > 0.5].to_string(index=False))
