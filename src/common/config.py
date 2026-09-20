"""Centralized constants for the reproduction. One source of truth; imported, never duplicated.

Decisions recorded in docs/audit/02-io-exposure-scoping.md.
"""
from __future__ import annotations

# --- Study window ---------------------------------------------------------
TARIFF_ANCHOR = "2018-2019"          # clean identifying episode
EXTERNAL_TEST_YEAR = 2025            # used with care (near-universal tariffs)

# --- China-input exposure (ICIO) ------------------------------------------
BASE_YEAR = 2017                     # exposure fixed pre-episode
ROBUSTNESS_YEARS = (2015, 2016, 2017)  # averaged in robustness

HOME_COUNTRY = "USA"                 # importer side (US industries)
# Extended ICIO splits China (and Mexico) by firm type. Handle every code form
# a release might use; the sum over these = total China.
CHINA_CODES = ("CHN", "CN1", "CN2")
CHINA_CODE_PREFIXES = ("CHN", "CN")

ICIO_EDITION_PRIMARY = "2025-extended"   # 1995-2022, splits China processing/non-processing
ICIO_EDITION_FALLBACK = "2023"           # 1995-2020, documented fallback

# Column codes in an ICIO row that are NOT country-industries (final demand, totals).
FINAL_DEMAND_CODES = frozenset({
    "HFCE", "NPISH", "GGFC", "GFCF", "INVNT", "DPABR", "NONRES",
    "P3", "P33", "P5", "P51", "P52", "P53",
    "OUT", "OUTPUT", "TOTAL", "FD", "GO",
})
# Row codes that are value-added / tax rows, not country-industries.
VA_ROW_CODES = frozenset({
    "TLS", "TXS_INT_FNL", "TXS_IMP_FNL", "VALU", "VA", "PURCH", "OUT", "OUTPUT", "TOTAL",
})

# --- Label parsing --------------------------------------------------------
LABEL_SEP = "_"  # ICIO labels look like "USA_D01T02" -> (country, industry code)
