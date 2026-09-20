"""Single deterministic entry point for the paper's reproduction.

Runs the whole chain from the shipped transformed data to the paper's headline numbers and
figures, in one command:

    data (transformed inputs)  ->  engine (structural model)  ->  analysis (panels, event
    studies, counterfactual)  ->  figures (EN and FR).

Determinism: every step is deterministic (no unseeded randomness, no timestamps in the numbers).
The headline results are rounded and hashed (SHA-256) into protocol/results_manifest.json; a
re-run compares against it and reports MATCH or a diff, so a change that moves a published number
is caught. Figures are regenerated (their bytes carry a timestamp, so we fingerprint the numbers,
not the PDF bytes).

Licensed data: the firm-level panels come from WRDS (Compustat, CRSP), which is not
redistributable and lives in data/wrds/ (gitignored, available on request). Those stages are
skipped with a clear note when the files are absent, so the public-data reproduction still runs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROTOCOL = Path(__file__).resolve().parent / "protocol"
MANIFEST = PROTOCOL / "results_manifest.json"


def _r(x, nd: int = 4) -> float:
    return round(float(x), nd)


def stage_data() -> list[str]:
    """Confirm the transformed public inputs are present; report which licensed inputs exist."""
    from src.common.paths import DATA_DIR, PROJECT_ROOT
    public = ["china_input_exposure.parquet", "tariffs_bown_timeline.csv",
              "import_price_china_bls.csv", "china_imports_hs4.parquet",
              "china_imports_naics.parquet"]
    missing = [f for f in public if not (DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing transformed public inputs: {missing}")
    wrds = PROJECT_ROOT / "data" / "wrds"
    have_wrds = wrds.exists() and any(wrds.glob("*.parquet"))
    print(f"[data] public inputs present ({len(public)}); WRDS licensed inputs "
          f"{'present' if have_wrds else 'ABSENT (firm-level stages skipped)'}")
    return public


def stage_engine_analysis() -> dict:
    """Compute the headline numbers from the structural model and the empirical panels."""
    results: dict[str, object] = {}

    # Sector pass-through (static): value response to the effective tariff.
    from src.analysis.sector_passthrough import run as sector_run
    results["sector_value_beta"] = _r(sector_run().loc[0, "beta"])

    # H1 border-price pass-through (public NAICS panel).
    from src.analysis.price_passthrough import run as price_run
    pr = price_run()
    results["border_price_beta"] = _r(pr["beta"])
    results["border_price_p"] = _r(pr["p"])

    # Reduced-form and GE structural results.
    from src.analysis.rebalancing_threshold import conclusion1
    c1 = conclusion1()
    obs = c1[c1["scenario"].str.startswith("observed")].iloc[0]
    results["reduced_form_required_deprec_observed"] = _r(obs["required_deprec"], 2)
    results["reduced_form_feasible_observed"] = bool(obs["feasible"])

    from src.analysis.dsge_counterfactual import theta_dollar_grid, chi_grid
    td = theta_dollar_grid(); ch = chi_grid()
    results["ge_efficiency_theta_dollar_0"] = _r(td.iloc[0]["efficiency"])
    results["ge_efficiency_theta_dollar_095"] = _r(
        td[td["theta_dollar"] == 0.95]["efficiency"].iloc[0])
    results["ge_efficiency_chi_0"] = _r(ch.iloc[0]["efficiency"])
    results["ge_efficiency_chi_4"] = _r(ch[ch["chi"] == 4.0]["efficiency"].iloc[0])

    # Phase 4 integration: data-implied trade elasticity.
    from src.analysis.integration import data_implied_eta
    results["data_implied_eta"] = _r(data_implied_eta())

    # Firm-level layer (licensed WRDS data; skipped if absent).
    try:
        from src.analysis.firm_incidence import run_annual
        fa = run_annual()
        gm = fa[fa["outcome"] == "gross margin"].iloc[0]
        results["firm_gross_margin_beta_firm_time"] = _r(gm["beta_firm_time"], 3)
        results["firm_gross_margin_beta_sectorXtime"] = _r(gm["beta_firm_sectorXtime"], 3)
    except FileNotFoundError:
        results["firm_layer"] = "skipped (WRDS data available on request)"
    return results


def stage_figures() -> list[str]:
    """Regenerate the master figures in EN and FR (deterministic; no hand-editing)."""
    from src.figures.style import apply_style, LANGS
    from src.figures import fig_event_study, fig_h1_decomposition, fig_ge_mechanism
    apply_style()
    made = []
    for mod, name in [(fig_event_study, "fig_event_study"),
                      (fig_h1_decomposition, "fig_h1_decomposition"),
                      (fig_ge_mechanism, "fig_ge_mechanism")]:
        for lang in LANGS:
            mod.make(lang)
            made.append(f"{name}.{lang}")
    return made


def fingerprint(results: dict) -> None:
    """Hash the rounded headline numbers and compare against the stored manifest."""
    payload = json.dumps(results, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    PROTOCOL.mkdir(exist_ok=True)
    if MANIFEST.exists():
        stored = json.loads(MANIFEST.read_text())
        if stored.get("sha256") == digest:
            print(f"[fingerprint] MATCH ({digest[:12]}...) — results reproduce exactly")
        else:
            print(f"[fingerprint] MISMATCH — stored {stored.get('sha256','')[:12]}... "
                  f"vs now {digest[:12]}...")
            for k in sorted(set(results) | set(stored.get("results", {}))):
                a = stored.get("results", {}).get(k); b = results.get(k)
                if a != b:
                    print(f"    {k}: stored {a!r} -> now {b!r}")
    else:
        MANIFEST.write_text(json.dumps({"sha256": digest, "results": results},
                                       indent=2, ensure_ascii=False))
        print(f"[fingerprint] wrote {MANIFEST.name} ({digest[:12]}...)")


def main() -> None:
    print("Reproducing: data -> engine/analysis -> figures\n")
    stage_data()
    results = stage_engine_analysis()
    made = stage_figures()
    print(f"[figures] regenerated {len(made)} figures (EN+FR)")
    print("\nHeadline results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print()
    fingerprint(results)


if __name__ == "__main__":
    main()
