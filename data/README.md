# Data

This folder holds the **transformed input data** needed to run `reproduce.py`. Raw and
licensed data are never versioned here (see the policy below).

## Policy by source

| Source | Content | Redistributable? | How it ships |
|---|---|---|---|
| WRDS Compustat (NA + Global), segments | firm fundamentals, margins/COGS, geographic segments | **No (licensed)** | not versioned — **available on request**; schema documented below |
| WRDS CRSP | event-window stock returns | **No (licensed)** | not versioned — available on request |
| OECD ICIO 2025 *extended* (fallback 2023) | inter-country input-output tables → China-input exposure by industry | Yes (public) | derived exposure table versioned; raw ICIO downloaded by acquisition code (not shipped) |
| WIOD 2016 | input-output tables (historical cross-check) | Yes (CC BY 4.0) | derived table versioned |
| Bown/PIIE, USITC | 2018-2019 and 2025 tariff calendar | Yes (public) | versioned |
| BLS MXP | U.S. import price indices / unit values | Yes (public) | versioned |
| Census / BEA | trade flows / quantities | Yes (public) | versioned |
| FRED / BIS / IMF | RMB/USD, REER, macro | Yes (public) | versioned |

## Versioned transformed inputs

| File | Provenance | Contents |
|---|---|---|
| `china_input_exposure.parquet` | OECD ICIO 2023 regular fallback, `2017_SML.csv` from the 2016-2020 block served by `stats.oecd.org/wbos/fileview2.aspx`; raw zip/CSV kept in `data/raw/` and gitignored | Long year-industry table for 2017, 45 ICIO industries, direct China-input exposure (`china_input_share`), `china_origin_codes=CHN` |
| `xwalk_isic4_naics.csv` | Starter ICIO 2023 industry to NAICS prefix concordance | Agriculture, mining and manufacturing coverage for joining Compustat `naics` to ICIO industries |
| `tariffs_bown_timeline.csv` | Chad Bown / PIIE, `us-china-trade-war-tariffs.xlsx` (Panel a), raw kept in `data/raw/` (gitignored) | Trade-weighted average tariff rates by dated action (US-on-China, China-on-US, vs ROW), 2018-2025; the episode event calendar |
| `tariffs_bown_coverage.csv` | Chad Bown / PIIE (Panel b) | Share of trade subject to the tariffs, by date |
| `import_price_china_bls.csv` | BLS Import/Export Price Indexes, `EIUCOCHN*` series via `api.bls.gov` v1 | Monthly China-origin import price index by NAICS, 2015-2024 (pass-through outcome) |
| `china_imports_hs4.parquet` | US Census International Trade API (`intltrade/imports/hs`, China=5700), pulled by the gitignored `data_download/pull_census_imports.py` | US imports from China by HS4 x month, 2015-2021: value, quantity+unit, calculated duties → effective tariff `duties/value` (the product-level `tau_{p,t}`) |
| `china_imports_naics.parquet` | US Census (`intltrade/imports/naics`, NA3+NA4, China=5700), pulled by the gitignored `data_download/pull_census_naics.py` | US imports from China by NAICS x month, 2015-2021: value + calculated duties → effective tariff by NAICS, to merge with the BLS NAICS price index (no HS↔NAICS bridge needed) |
| `us_china_trade_annual.csv` | US Census International Trade API (`intltrade/exports/hs` `ALL_VAL_YR`, `intltrade/imports/hs` `GEN_VAL_YR`, China=5700), pulled by the gitignored `data_download/pull_macro_inputs.py` | Annual US goods exports to and general imports from China, dollars, 2015-2021: import share, bilateral openness, initial imbalance |
| `fx_cny_usd_monthly.csv` | FRED series `EXCHUS` (Federal Reserve H.10), same script | Renminbi per dollar, monthly average, 2014-2024: exchange-rate pass-through disciplining the dollar-invoicing friction |
| `us_gdp_annual.csv` | FRED series `GDPA` (BEA), same script | US nominal GDP, billions of dollars, annual, 2014-2024: scales bilateral trade and the imbalance to output |

## Data access notes (product/sector panel)

- **Prices**: BLS Import/Export Price Indexes (MXP) via `api.bls.gov` (v1 keyless) and `download.bls.gov`. The exchange rate and output come from FRED (`EXCHUS`, `GDPA`) through the gitignored `data_download/pull_macro_inputs.py`.
- **Quantities / values / effective duties by HS x China x month**: US Census International Trade API (`api.census.gov/.../intltrade/imports/hs`) — provides `GEN_VAL_MO`, `GEN_QY1_MO`, `CAL_DUT_MO`/`DUT_VAL_MO`. **Requires a free Census API key** (register at api.census.gov/data/key_signup.html); store it in `.env.local` as `CENSUS_API_KEY`. This unlocks the product-level tariff (duties/value) and quantity dimensions of the sector panel.

## Cleaning rules for analysis panels

- Compustat margin ratios are kept in raw form and also exposed as winsorized analysis
  columns suffixed `_w`.
- Winsorization rule: 1st and 99th percentiles, configured in `src/common/config.py`
  (`WINSOR_LOWER`, `WINSOR_UPPER`).
- Event studies should use the `_w` columns by default and report raw-column robustness
  only if needed.
- Census quantities can be zero or missing; `log_qty` is missing unless `qty1 > 0`.

## Layout (to be populated)

```
data/
  README.md              this file
  <transformed public tables>.parquet / .csv.gz
  # raw and licensed data live outside the repo (data/raw/, data/wrds/ are gitignored)
```

## WRDS schema (for the available-on-request data)

The licensed WRDS inputs are stored outside version control under `data/wrds/`. A third party
with WRDS access can reconstruct the transformed files with the following schemas.

| File | Source table | Unit and keys | Columns |
|---|---|---|---|
| `compustat_funda.parquet` | Compustat annual fundamentals, North America and Global | firm-year, keyed by `gvkey`, `datadate`, `fyear` | `gvkey`, `datadate`, `fyear`, `conm`, `naics`, `sic`, `loc`, `fic`, `sale`, `cogs`, `xsga`, `ebit`, `oibdp`, `gp`, `at`, `emp`, `ppent` |
| `compustat_fundq.parquet` | Compustat quarterly fundamentals, North America and Global | firm-quarter, keyed by `gvkey`, `datadate`, `fyearq`, `fqtr` | `gvkey`, `datadate`, `fyearq`, `fqtr`, `conm`, `naics`, `sic`, `loc`, `saleq`, `cogsq`, `xsgaq`, `oibdpq`, `niq`, `atq` |
| `compustat_geoseg.parquet` | Compustat geographic segments | firm-date-segment, keyed by `gvkey`, `datadate`, `snms` | `gvkey`, `datadate`, `snms`, `sales` |
| `crsp_returns_monthly.parquet` | CRSP monthly stock file linked to Compustat | firm-month/security-month, keyed by `gvkey`, `permno`, `date` | `gvkey`, `permno`, `date`, `ret`, `retx`, `prc`, `shrout`, `vol` |

The current transformed WRDS snapshot contains 57,843 annual Compustat rows, 230,384 quarterly
Compustat rows, 156,256 geographic segment rows, and 387,144 monthly CRSP rows. Raw WRDS extracts
remain unversioned under the repository data policy.
