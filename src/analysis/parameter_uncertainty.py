"""Monte Carlo propagation of parameter uncertainty into the H2 and H3 results.

Two layers of draws from the data-disciplined baseline (src.analysis.baseline):

  estimation  only the estimated parameters vary, drawn from their sampling distributions:
              eta ~ N(eta_hat, se) and theta ~ N(theta_hat, se), truncated to [0, THETA_MAX].
              The quantiles of the results are then confidence intervals.
  full        in addition, the trade ratios vary uniformly over their 2015-2021 range and every
              calibrated parameter over the documented range of CALIBRATED_RANGES. The
              quantiles then describe the sensitivity of the results to the calibration, not
              sampling uncertainty.

For each draw, the H3 comparison moves each friction from zero to its level in the draw while
holding the other at its level:

    invoicing effect = Lambda(theta, chi) / Lambda(0, chi) - 1
    wedge effect     = Lambda(theta, chi) / Lambda(theta, 0) - 1

and the ranking of H3 holds when the wedge effect is below the invoicing effect. Draws without
a unique stable equilibrium are dropped and counted. The H2 impact bound is evaluated on the
same draws. The persistence rho_z of the engineered intervention is a feature of the policy
experiment and stays at its baseline value (its role is reported separately in
src.analysis.dsge_counterfactual.intervention_persistence).

The draws are summarized by shares, quantiles, and rank correlations, not by t-statistics: the
number of draws is a choice, and the spread of the calibrated parameters reflects the chosen
ranges rather than sampling variation.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.analysis.baseline import baseline
from src.analysis.dsge_counterfactual import efficiency
from src.analysis.parameter_estimation import estimate_eta, estimate_passthrough, trade_share_range
from src.engine.model import is_feasible, replace, required_depreciation

N_DRAWS = 10_000
SEED = 20260921
THETA_MAX = 0.99
QUANTILES = (0.05, 0.50, 0.95)
LAYERS = ("estimation", "full")

# Calibrated parameters: (distribution, lower bound, upper bound). Ranges bracket the values
# used in the literature cited in the calibration table; the capital-controls wedge and the
# portfolio cost are log-uniform because only their order of magnitude is known.
CALIBRATED_RANGES = {
    "eta_star": ("uniform", 1.0, 5.0),
    "chi": ("log-uniform", 0.25, 4.0),
    "portfolio_cost": ("log-uniform", 0.005, 0.10),
    "sigma": ("uniform", 1.0, 5.0),
    "theta": ("uniform", 0.50, 0.90),
    "phi_pi": ("uniform", 1.25, 2.0),
    "phi_y": ("uniform", 0.25, 1.0),
    "max_depreciation": ("uniform", 0.15, 0.50),
    "damping_scale": ("uniform", 0.0, 4.0),
}
TRADE_RATIOS = ("gamma", "import_share", "imbalance0")


def _uniform(rng: np.random.Generator, kind: str, lo: float, hi: float, n: int) -> np.ndarray:
    if kind == "log-uniform":
        return np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    return rng.uniform(lo, hi, n)


def parameter_ranges() -> pd.DataFrame:
    """Distribution of every parameter drawn in the full layer."""
    b, eta, pt = baseline(), estimate_eta(), estimate_passthrough()
    years = trade_share_range()
    rows = [
        {"parameter": "eta", "distribution": "normal", "lo": eta["eta"], "hi": eta["se"]},
        {"parameter": "theta_dollar", "distribution": "normal", "lo": pt["theta"], "hi": pt["se"]},
    ]
    rows += [{"parameter": k, "distribution": "uniform", "lo": float(years[k].min()),
              "hi": float(years[k].max())} for k in TRADE_RATIOS]
    rows += [{"parameter": k, "distribution": d, "lo": lo, "hi": hi}
             for k, (d, lo, hi) in CALIBRATED_RANGES.items()]
    out = pd.DataFrame(rows)
    out["baseline"] = [1.0 if k == "damping_scale" else getattr(b, k) for k in out["parameter"]]
    return out


def draw_parameters(layer: str, n: int = N_DRAWS, seed: int = SEED) -> pd.DataFrame:
    """Parameter draws for one layer; parameters that do not vary stay at the baseline."""
    if layer not in LAYERS:
        raise ValueError(f"layer must be one of {LAYERS}")
    rng = np.random.default_rng(seed)
    b, eta, pt = baseline(), estimate_eta(), estimate_passthrough()
    draws = {
        "eta": rng.normal(eta["eta"], eta["se"], n),
        "theta_dollar": np.clip(rng.normal(pt["theta"], pt["se"], n), 0.0, THETA_MAX),
    }
    ranges = parameter_ranges().set_index("parameter")
    for k in (*TRADE_RATIOS, *CALIBRATED_RANGES):
        if layer == "full":
            kind = "log-uniform" if ranges.loc[k, "distribution"] == "log-uniform" else "uniform"
            draws[k] = _uniform(rng, kind, ranges.loc[k, "lo"], ranges.loc[k, "hi"], n)
        else:
            draws[k] = np.full(n, 1.0 if k == "damping_scale" else getattr(b, k))
    return pd.DataFrame(draws)


def _evaluate(row: pd.Series) -> dict:
    """H2 and H3 objects for one parameter draw."""
    params = {k: float(v) for k, v in row.items() if k != "damping_scale"}
    p = replace(baseline(), **params)
    k = float(row["damping_scale"])
    out = {"required_deprec": required_depreciation(p, damping_scale=k),
           "feasible": is_feasible(p, damping_scale=k)}
    try:
        lam = efficiency(p)
        lam_pcp = efficiency(replace(p, theta_dollar=0.0))
        lam_open = efficiency(replace(p, chi=0.0))
    except ValueError:
        return {**out, "determinate": False}
    well_defined = lam_pcp > 0 and lam_open > 0
    out.update(determinate=True, well_defined=well_defined, eff=lam, eff_no_invoicing=lam_pcp,
               eff_open=lam_open,
               invoicing_effect=lam / lam_pcp - 1 if well_defined else np.nan,
               wedge_effect=lam / lam_open - 1 if well_defined else np.nan)
    return out


@lru_cache(maxsize=None)
def run(layer: str, n: int = N_DRAWS, seed: int = SEED) -> pd.DataFrame:
    """Draws and their results, one row per draw."""
    draws = draw_parameters(layer, n, seed)
    results = pd.DataFrame([_evaluate(row) for _, row in draws.iterrows()])
    out = pd.concat([draws, results], axis=1)
    out["determinate"] = out["determinate"].ne(False)
    out["well_defined"] = out["well_defined"].eq(True)
    out["difference"] = out["wedge_effect"] - out["invoicing_effect"]
    out["ranking_holds"] = out["difference"] < 0
    return out


def _quantiles(values: pd.Series) -> dict:
    return {f"q{int(100 * a):02d}": float(values.quantile(a)) for a in QUANTILES}


def summary(layer: str) -> dict:
    """Shares, quantiles, and counts that the paper reports for one layer."""
    df = run(layer)
    ok = df[df["determinate"] & df["well_defined"]]
    return {
        "n": len(df), "n_determinate": int(df["determinate"].sum()), "n_used": len(ok),
        "share_ranking": float(ok["ranking_holds"].mean()),
        "share_larger_magnitude": float((ok["wedge_effect"].abs() > ok["invoicing_effect"].abs()).mean()),
        "share_invoicing_negative": float((ok["invoicing_effect"] < 0).mean()),
        "share_wedge_negative": float((ok["wedge_effect"] < 0).mean()),
        "invoicing": _quantiles(ok["invoicing_effect"]), "wedge": _quantiles(ok["wedge_effect"]),
        "difference": _quantiles(ok["difference"]),
        "required_deprec": _quantiles(df["required_deprec"]),
        "share_feasible": float(df["feasible"].mean()),
        "share_beyond_100": float(((ok["invoicing_effect"].abs() > 1) | (ok["wedge_effect"].abs() > 1)).mean()),
    }


def drivers() -> pd.DataFrame:
    """Spearman rank correlation of each drawn parameter with the two effects and their
    difference, in the full layer."""
    df = run("full")
    ok = df[df["determinate"] & df["well_defined"]]
    rows = []
    for k in parameter_ranges()["parameter"]:
        if k in ("max_depreciation", "damping_scale", "imbalance0"):
            continue  # enter the impact bound only
        rows.append({"parameter": k, **{
            f"rho_{target}": float(spearmanr(ok[k], ok[target]).statistic)
            for target in ("invoicing_effect", "wedge_effect", "difference")}})
    return pd.DataFrame(rows)


def ranking_failures() -> dict:
    """Where the ranking fails in the full layer: the median wedge and portfolio cost of the
    failing draws, and the share of draws with the ranking inside two sub-regions."""
    df = run("full")
    ok = df[df["determinate"] & df["well_defined"]]
    fail = ok[~ok["ranking_holds"]]
    return {"n_fail": len(fail), "chi_median": float(fail["chi"].median()),
            "psi_median": float(fail["portfolio_cost"].median()),
            "share_ranking_chi_high": float(ok.loc[ok["chi"] >= 0.5, "ranking_holds"].mean()),
            "share_ranking_psi_low": float(ok.loc[ok["portfolio_cost"] <= 0.05, "ranking_holds"].mean())}


def indeterminacy() -> pd.DataFrame:
    """Mean of each drawn parameter in the determinate and indeterminate draws (full layer)."""
    df = run("full")
    cols = list(parameter_ranges()["parameter"])
    return df.groupby("determinate")[cols].mean().T


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.width", 160)
    print(parameter_ranges().to_string(index=False))
    for layer in LAYERS:
        print(f"\n{layer} layer")
        for key, val in summary(layer).items():
            print(f"  {key}: {val}")
    print("\nRank correlations with the effects (full layer)")
    print(drivers().to_string(index=False))
    print("\nWhere the ranking fails (full layer)", ranking_failures())
    print("\nParameter means by determinacy (full layer)")
    print(indeterminacy().to_string())
