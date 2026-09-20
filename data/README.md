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

## Layout (to be populated)

```
data/
  README.md              this file
  <transformed public tables>.parquet / .csv.gz
  # raw and licensed data live outside the repo (data/raw/, data/wrds/ are gitignored)
```

## WRDS schema (for the available-on-request data)

To be documented as the panels are built (tables, keys, columns, filters, sample window),
so a third party with WRDS access can reconstruct the licensed inputs.
