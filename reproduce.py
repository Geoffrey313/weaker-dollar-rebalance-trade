"""Single deterministic entry point for the paper's reproduction.

Runs the whole chain from the shipped transformed data to the paper's headline numbers and
figures, in one command:

    data (transformed inputs)  ->  engine (structural model)  ->  analysis (panels, event
    studies, counterfactual)  ->  figures (EN and FR).

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
              "china_imports_naics.parquet"]
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

    from src.analysis.price_passthrough import run as price_run
    pr = price_run()
    results["border_price_beta"] = _r(pr["beta"])
    results["border_price_p"] = _r(pr["p"])

    from src.analysis.rebalancing_threshold import conclusion1
    obs = conclusion1().pipe(lambda d: d[d["scenario"].str.startswith("observed")]).iloc[0]
    results["reduced_form_required_deprec_observed"] = _r(obs["required_deprec"], 2)
    results["reduced_form_feasible_observed"] = bool(obs["feasible"])

    from src.analysis.dsge_counterfactual import theta_dollar_grid, chi_grid
    td = theta_dollar_grid(); ch = chi_grid()
    results["ge_efficiency_theta_dollar_0"] = _r(td.iloc[0]["efficiency"])
    results["ge_efficiency_theta_dollar_095"] = _r(td[td["theta_dollar"] == 0.95]["efficiency"].iloc[0])
    results["ge_efficiency_chi_0"] = _r(ch.iloc[0]["efficiency"])
    results["ge_efficiency_chi_4"] = _r(ch[ch["chi"] == 4.0]["efficiency"].iloc[0])

    from src.analysis.integration import data_implied_eta
    results["data_implied_eta"] = _r(data_implied_eta())
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
    from src.figures import fig_event_study, fig_h1_decomposition, fig_ge_mechanism
    apply_style()
    n = 0
    for mod in (fig_event_study, fig_h1_decomposition, fig_ge_mechanism):
        for lang in LANGS:
            mod.make(lang); n += 1
    return n


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
                a = stored.get("results", {}).get(k); b = results.get(k)
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
    print(f"[figures] regenerated {stage_figures()} figures (EN+FR)\n")

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
