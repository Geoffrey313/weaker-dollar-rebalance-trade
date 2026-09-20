"""China-input exposure by US industry, from OECD ICIO inter-country input-output tables.

What it computes
----------------
For each US industry j, the DIRECT share of its intermediate inputs sourced from China:

    s_j = ( sum over China origin-industries i of  Z[China_i -> USA_j] )
          -----------------------------------------------------------------
          ( sum over ALL origin country-industries (c,i) of Z[(c,i) -> USA_j] )

This is an UPSTREAM "China-input exposure" measure (where US industries BUY), the object
relevant to tariff exposure on imported inputs. It is NOT tariff incidence (firm x HS) and
NOT the ADH "China shock" (downstream import competition). See docs/audit/02-io-exposure-scoping.md.

Inputs
------
Raw OECD ICIO year files (wide matrix, one CSV per year) live in data/raw/ (gitignored,
fetched by data_download/fetch_icio.py). Row/column labels are "CCC_III" (country_industry),
plus final-demand columns and value-added rows, which are excluded here.

Output
------
A long tidy table [year, industry, china_intermediate, total_intermediate,
china_input_share], versioned as data/china_input_exposure.parquet. The concordance
layer filters it to the base year before assigning exposure to firms.

Assumption to state in the paper: exposure is fixed at a pre-episode base year (config.BASE_YEAR),
so it is predetermined with respect to the 2018-2019 tariffs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common import config as C


def parse_label(label: str) -> tuple[str | None, str]:
    """'USA_D01T02' -> ('USA', 'D01T02'); a label without a separator -> (None, label)."""
    s = str(label)
    if C.LABEL_SEP in s:
        country, code = s.split(C.LABEL_SEP, 1)
        return country, code
    return None, s


def _is_country_industry_row(label: str) -> bool:
    country, code = parse_label(label)
    return country is not None and code not in C.VA_ROW_CODES and code not in C.FINAL_DEMAND_CODES


def _is_home_industry_col(label: str) -> bool:
    country, code = parse_label(label)
    return country == C.HOME_COUNTRY and code not in C.VA_ROW_CODES and code not in C.FINAL_DEMAND_CODES


def _is_china_country(country: str | None) -> bool:
    """True for China aggregate/split country codes used across ICIO releases."""
    if country is None:
        return False
    return country in C.CHINA_CODES or country.startswith(C.CHINA_CODE_PREFIXES)


def _is_china_row(label: str) -> bool:
    country, code = parse_label(label)
    return _is_china_country(country) and code not in C.VA_ROW_CODES and code not in C.FINAL_DEMAND_CODES


def china_country_codes_present(labels: pd.Index | list[str]) -> tuple[str, ...]:
    """Return China country codes observed in ICIO labels, for release-specific auditing."""
    found = {parse_label(label)[0] for label in labels}
    return tuple(sorted(country for country in found if _is_china_country(country)))


def load_icio_year(path: str | Path) -> pd.DataFrame:
    """Read one ICIO year file (CSV, optionally .gz/.zip) as a labelled wide matrix.

    The first column holds the row labels; all data cells are coerced to float.
    """
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df.apply(pd.to_numeric, errors="coerce")


def china_input_share(Z: pd.DataFrame) -> pd.DataFrame:
    """Direct China input share per US industry, from one ICIO year matrix `Z`.

    Returns a tidy frame indexed by ISIC Rev.4 industry code with columns
    china_intermediate, total_intermediate, china_input_share.
    """
    us_cols = [c for c in Z.columns if _is_home_industry_col(c)]
    china_rows = [r for r in Z.index if _is_china_row(r)]
    ci_rows = [r for r in Z.index if _is_country_industry_row(r)]
    if not us_cols:
        raise ValueError(f"No {C.HOME_COUNTRY} industry columns found — check label format / codes.")
    if not china_rows:
        raise ValueError(f"No China rows found (codes {C.CHINA_CODES}) — check the ICIO edition codes.")
    china_codes = china_country_codes_present(Z.index)

    china_int = Z.loc[china_rows, us_cols].sum(axis=0)
    total_int = Z.loc[ci_rows, us_cols].sum(axis=0)
    out = pd.DataFrame({
        "industry": [parse_label(c)[1] for c in us_cols],
        "china_intermediate": china_int.to_numpy(),
        "total_intermediate": total_int.to_numpy(),
    })
    # An industry appears once as a US column, but guard against duplicates by summing.
    out = out.groupby("industry", as_index=False).sum()
    out["china_input_share"] = out["china_intermediate"] / out["total_intermediate"]
    out["china_origin_codes"] = ",".join(china_codes)
    return out.sort_values("industry").reset_index(drop=True)


def build_exposure(year_paths: dict[int, str | Path]) -> pd.DataFrame:
    """Compute China-input exposure for several years -> long tidy frame [year, industry, ...]."""
    frames = []
    for year, path in sorted(year_paths.items()):
        share = china_input_share(load_icio_year(path))
        share.insert(0, "year", year)
        frames.append(share)
    return pd.concat(frames, ignore_index=True)


def base_and_robustness(long: pd.DataFrame) -> pd.DataFrame:
    """From the long frame, build the treatment table: base-year share + averaged robustness share."""
    base = (long[long["year"] == C.BASE_YEAR][["industry", "china_input_share"]]
            .rename(columns={"china_input_share": "china_input_share_base"}))
    rob = (long[long["year"].isin(C.ROBUSTNESS_YEARS)]
           .groupby("industry", as_index=False)["china_input_share"].mean()
           .rename(columns={"china_input_share": "china_input_share_avg"}))
    return base.merge(rob, on="industry", how="outer").sort_values("industry").reset_index(drop=True)


def build_and_save(year_paths: dict[int, str | Path], out_path: str | Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Compute the long year-industry exposure frame for `year_paths` and write it to parquet.

    Default output: data/china_input_exposure.parquet. It remains long-form even if only
    the base year is available; downstream firm joins filter to config.BASE_YEAR.
    """
    from src.common.paths import DATA_DIR
    long = build_exposure(year_paths)
    out = Path(out_path) if out_path is not None else DATA_DIR / "china_input_exposure.parquet"
    long.to_parquet(out, index=False)
    return long, out


if __name__ == "__main__":
    # Self-test on a tiny synthetic 2-country (USA, CHN) x 2-industry ICIO matrix,
    # so the computation is verifiable without the multi-GB real file.
    labels = ["USA_C10", "USA_C26", "CHN_C10", "CHN_C26", "CN1_C10", "USA_TLS"]
    fd_va = ["USA_HFCE", "TLS", "USA_VA"]
    demo = pd.DataFrame(
        [
            # cols: USA_C10 USA_C26 CHN_C10 CHN_C26 CN1_C10 USA_TLS | USA_HFCE TLS USA_VA
            [10, 5, 0, 0, 0, 999, 30, 0, 999],   # USA_C10 supplying
            [4, 20, 0, 0, 0, 999, 10, 0, 999],   # USA_C26 supplying
            [5, 15, 0, 0, 0, 999, 0, 0, 999],    # CHN_C10 supplying -> into USA cols
            [0, 40, 0, 0, 0, 999, 0, 0, 999],    # CHN_C26 supplying -> into USA cols
            [1, 0, 0, 0, 0, 999, 0, 0, 999],     # CN1 split-China row is included
            [100, 100, 0, 0, 0, 999, 0, 0, 999], # USA_TLS value-added row is excluded
        ],
        index=labels, columns=labels + fd_va,
    )
    res = china_input_share(demo)
    print(res.to_string(index=False))
    # Manual check: USA_C10 col intermediates = 10+4+5+0+1=20, China=5+0+1=6 -> 0.30
    #               USA_C26 col intermediates = 5+20+15+40+0=80, China=15+40+0=55 -> 0.6875
    assert abs(res.loc[res.industry == "C10", "china_input_share"].iloc[0] - 0.30) < 1e-9
    assert abs(res.loc[res.industry == "C26", "china_input_share"].iloc[0] - 0.6875) < 1e-9
    assert "TLS" not in set(res["industry"])
    assert "VA" not in set(res["industry"])
    assert res["china_origin_codes"].iloc[0] == "CHN,CN1"
    print("\nself-test OK")
