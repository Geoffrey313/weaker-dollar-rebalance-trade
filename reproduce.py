"""Single deterministic entry point for the paper's reproduction.

Runs the whole chain from the shipped transformed data to the paper's headline numbers and
figures, in one command:

    data (transformed inputs)  ->  engine (structural model)  ->  analysis (panels, event
    studies, counterfactual)  ->  figures and tables (EN and FR).

The tables and the in-text number macros are LaTeX fragments written into
manuscript/<lang>/ssrn/tables/, so no number in the manuscripts is typed by hand.

Determinism: every step is deterministic (no unseeded randomness, no timestamps in the numbers).
Headline numbers are rounded and hashed (SHA-256) into a manifest; a re-run compares against it
and reports MATCH or a per-key diff, so a change that moves a published number is caught. Figure
bytes carry a timestamp, so we fingerprint the numbers, not the PDF bytes.

Two manifests, because the data have two licences:
  - protocol/results_manifest_public.json  : PUBLIC-data results only (Census, BLS, PIIE, OECD
    ICIO). Versioned. This is what a third party reproduces WITHOUT any licensed data, so its
    hash is stable everywhere.
  - protocol/results_manifest_wrds.json     : the firm-level supplement from WRDS (Compustat,
    CRSP) — licensed, gitignored. Present only on a machine with data/wrds/. The public
    reproduction never depends on it and never breaks when it is absent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROTOCOL = Path(__file__).resolve().parent / "protocol"
MANIFEST_PUBLIC = PROTOCOL / "results_manifest_public.json"
MANIFEST_WRDS = PROTOCOL / "results_manifest_wrds.json"  # gitignored (licensed supplement)


def _r(x, nd: int = 4) -> float:
    return round(float(x), nd)


def stage_data() -> None:
    """Confirm the transformed public inputs are present; report whether licensed inputs exist."""
    from src.common.paths import DATA_DIR, PROJECT_ROOT
    public = ["china_input_exposure.parquet", "tariffs_bown_timeline.csv",
              "import_price_china_bls.csv", "china_imports_hs4.parquet",
              "china_imports_naics.parquet", "us_china_trade_annual.csv",
              "fx_cny_usd_monthly.csv", "us_gdp_annual.csv"]
    missing = [f for f in public if not (DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing transformed public inputs: {missing}")
    wrds = PROJECT_ROOT / "data" / "wrds"
    have_wrds = wrds.exists() and any(wrds.glob("*.parquet"))
    print(f"[data] public inputs present ({len(public)}); WRDS licensed inputs "
          f"{'present' if have_wrds else 'ABSENT (firm-level supplement skipped)'}")


def public_results() -> dict:
    """Headline numbers reproducible from PUBLIC data only (no WRDS)."""
    results: dict[str, object] = {}

    from src.analysis.sector_passthrough import run as sector_run
    results["sector_value_beta"] = _r(sector_run().loc[0, "beta"])

    from src.analysis.price_passthrough import run as price_run, run_wcb
    pr = price_run()
    results["border_price_beta"] = _r(pr["beta"])
    results["border_price_p"] = _r(pr["p"])
    results["border_price_wcb_p"] = _r(run_wcb()["p_wcb"], 3)  # small-G reliable inference

    # Parameters identified by the project's data (the structural baseline uses them).
    from src.analysis.parameter_estimation import estimates
    for key, value in estimates().items():
        results[f"estimated_{key}"] = _r(value)

    from src.analysis.rebalancing_threshold import conclusion1
    h2 = conclusion1().set_index("scenario")
    results["impact_bound_required_deprec_estimated"] = _r(h2.iloc[0]["required_deprec"], 3)
    results["impact_bound_required_deprec_no_friction"] = _r(h2.iloc[3]["required_deprec"], 3)
    results["impact_bound_feasible_estimated"] = bool(h2.iloc[0]["feasible"])

    from src.analysis.dsge_counterfactual import theta_dollar_grid, chi_grid
    td = theta_dollar_grid()
    ch = chi_grid()
    results["ge_efficiency_theta_dollar_0"] = _r(td.iloc[0]["efficiency"])
    results["ge_efficiency_theta_dollar_estimated"] = _r(td[td["estimated"]]["efficiency"].iloc[0])
    results["ge_efficiency_chi_0"] = _r(ch.iloc[0]["efficiency"])
    results["ge_efficiency_chi_4"] = _r(ch[ch["chi"] == 4.0]["efficiency"].iloc[0])

    from src.analysis.integration import data_implied_eta
    results["data_implied_eta"] = _r(data_implied_eta())

    # Staggered-robust (binarised stacked DiD) event-study post-treatment average, without and
    # with the Wing-Freedman-Hollingsworth corrective weights.
    from src.analysis.sector_event_study_staggered import run as staggered_run, post_mean
    sres = staggered_run()
    results["staggered_post_mean_beta"] = _r(sres.loc[[k for k in sres.index if k >= 0], "beta"].mean(), 3)
    results["staggered_post_mean_beta_weighted"] = _r(post_mean(staggered_run(weighted=True)), 3)

    # Identification tests of the event study: pre-announcement reference, split-sample intensity.
    from src.analysis.sector_event_study import identification_tests
    idt = identification_tests()
    for key in ("baseline", "reference_2017Q4", "split_sample"):
        results[f"event_post_mean_{key}"] = _r(idt[key]["post_mean"], 3)

    # Robustness of the H3 comparative statics to one-way calibration changes (GE model).
    from src.analysis.dsge_counterfactual import sensitivity as ge_sensitivity
    gs = ge_sensitivity()
    det = gs[gs["determinate"]]
    results["ge_sensitivity_determinate"] = f"{len(det)}/{len(gs)}"
    results["ge_sensitivity_invoicing_ratio_range"] = [_r(det["invoicing_ratio"].min(), 3),
                                                       _r(det["invoicing_ratio"].max(), 3)]
    results["ge_sensitivity_wedge_ratio_range"] = [_r(det["wedge_ratio"].min(), 3),
                                                   _r(det["wedge_ratio"].max(), 3)]

    # Literature defaults versus the data-disciplined baseline: the headline structural
    # conclusions should not depend on which calibration is used as the reference point.
    from src.analysis.calibration_validation import compare as calibration_compare
    cv = calibration_compare().set_index("calibration")
    for key in ("literature", "data_disciplined"):
        tag = "data" if key == "data_disciplined" else key
        results[f"calibration_{tag}_required_observed"] = _r(cv.loc[key, "required_observed"], 3)
        results[f"calibration_{tag}_required_no_friction"] = _r(cv.loc[key, "required_no_friction"], 3)
        results[f"calibration_{tag}_invoicing_ratio"] = _r(cv.loc[key, "invoicing_ratio"], 3)
        results[f"calibration_{tag}_wedge_ratio"] = _r(cv.loc[key, "wedge_ratio"], 3)
        results[f"calibration_{tag}_ranking_holds"] = bool(cv.loc[key, "ranking_holds"])

    # Monte Carlo propagation of parameter uncertainty (fixed seed, deterministic).
    from src.analysis.parameter_uncertainty import LAYERS, summary as mc_summary
    for layer in LAYERS:
        mc = mc_summary(layer)
        results[f"mc_{layer}_share_ranking"] = _r(mc["share_ranking"], 4)
        results[f"mc_{layer}_share_determinate"] = _r(mc["n_determinate"] / mc["n"], 4)
        results[f"mc_{layer}_invoicing_effect_median"] = _r(mc["invoicing"]["q50"], 3)
        results[f"mc_{layer}_wedge_effect_median"] = _r(mc["wedge"]["q50"], 3)
        results[f"mc_{layer}_share_feasible"] = _r(mc["share_feasible"], 4)
    return results


def wrds_results() -> dict | None:
    """Firm-level supplement from licensed WRDS data; None when the data are absent."""
    try:
        from src.analysis.firm_incidence import run_annual
        fa = run_annual()
        gm = fa[fa["outcome"] == "gross margin"].iloc[0]
        return {"firm_gross_margin_beta_firm_time": _r(gm["beta_firm_time"], 3),
                "firm_gross_margin_beta_sectorXtime": _r(gm["beta_firm_sectorXtime"], 3)}
    except FileNotFoundError:
        return None


def stage_figures() -> int:
    """Regenerate the master figures in EN and FR (deterministic; no hand-editing)."""
    from src.figures.style import apply_style, LANGS
    from src.figures import (fig_event_study, fig_flows, fig_ge_irf, fig_ge_mechanism,
                             fig_h1_decomposition, fig_uncertainty, fig_uncertainty_bound)
    apply_style()
    n = 0
    for mod in (fig_flows, fig_event_study, fig_h1_decomposition, fig_ge_mechanism, fig_ge_irf,
                fig_uncertainty, fig_uncertainty_bound):
        for lang in LANGS:
            mod.make(lang)
            n += 1
    return n


def stage_tables() -> int:
    """Regenerate the results tables and in-text number macros in EN and FR."""
    from src.figures.tables import make_all
    return make_all()


def fingerprint(results: dict, manifest: Path, label: str) -> None:
    """Hash the rounded results and compare against `manifest` (write it on first run)."""
    digest = hashlib.sha256(json.dumps(results, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    PROTOCOL.mkdir(exist_ok=True)
    if manifest.exists():
        stored = json.loads(manifest.read_text())
        if stored.get("sha256") == digest:
            print(f"[{label}] MATCH ({digest[:12]}...) — reproduces exactly")
        else:
            print(f"[{label}] MISMATCH — stored {stored.get('sha256','')[:12]}... vs now {digest[:12]}...")
            for k in sorted(set(results) | set(stored.get("results", {}))):
                a = stored.get("results", {}).get(k)
                b = results.get(k)
                if a != b:
                    print(f"    {k}: stored {a!r} -> now {b!r}")
    else:
        manifest.write_text(json.dumps({"sha256": digest, "results": results}, indent=2, ensure_ascii=False))
        print(f"[{label}] wrote {manifest.name} ({digest[:12]}...)")


def main() -> None:
    print("Reproducing: data -> engine/analysis -> figures\n")
    stage_data()
    pub = public_results()
    wr = wrds_results()
    print(f"[figures] regenerated {stage_figures()} figures (EN+FR)")
    print(f"[tables] regenerated {stage_tables()} table fragments (EN+FR)\n")

    print("Headline results (public):")
    for k, v in pub.items():
        print(f"  {k}: {v}")
    fingerprint(pub, MANIFEST_PUBLIC, "public manifest")

    if wr is None:
        print("\n[WRDS supplement] SKIP — licensed firm-level data absent "
              "(public reproduction is complete and independent of it)")
    else:
        print("\nFirm-level supplement (WRDS, licensed):")
        for k, v in wr.items():
            print(f"  {k}: {v}")
        fingerprint(wr, MANIFEST_WRDS, "WRDS supplement")


if __name__ == "__main__":
    main()
