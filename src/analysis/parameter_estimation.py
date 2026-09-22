"""Estimation of the structural parameters that the project's data identify.

The structural model has three tiers of parameters.

  1. Estimated from the project's panels:
       eta      import-demand elasticity, from the product panel regressed on the log gross
                tariff (Lemma 1: the coefficient is -eta when the border price does not offset);
       theta    dollar-invoicing friction, from the exchange-rate pass-through into the dollar
                import price of Chinese goods: the model implies a cumulative pass-through of
                1 - theta^h after h quarters, so theta = (1 - CPT_h)^(1/h);
       rho_tau  persistence of the tariff, AR(1) of the quarterly aggregate effective tariff.
  2. Computed from trade and output data for the pre-episode base year (2017):
       s        import share of bilateral trade, M / (X + M);
       gamma    bilateral openness, (X + M) / Y;
       b        initial bilateral imbalance relative to output, (M - X) / Y.
  3. Calibrated from the literature, with sensitivity: eta_star (the foreign elasticity is not
     identified by US import data), chi, psi, sigma, xi (Calvo), phi_pi, phi_y, beta, rho_z,
     the tolerated appreciation, and the damping scale.

Outputs feed the baseline calibration (src.analysis.baseline) and the parameter table.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data.macro import load_bilateral_trade, load_gdp, load_renminbi_value_quarterly
from src.data.prices import china_import_price_total_quarterly
from src.data.sector_imports import aggregate_effective_tariff_quarterly

BASE_YEAR = 2017                 # pre-episode year for trade shares and the imbalance
ERPT_LAGS = 4                    # quarters of distributed lags in the pass-through regression
ERPT_HORIZON = 4                 # horizon at which the model is matched to the data
SAMPLE = ("2015Q1", "2021Q4")    # the project's sample
HAC_LAGS = 2


@lru_cache(maxsize=None)
def trade_shares(year: int = BASE_YEAR) -> dict:
    """Import share, bilateral openness, and imbalance for `year` (Census trade, BEA GDP)."""
    trade = load_bilateral_trade().loc[year]
    gdp = float(load_gdp().loc[year])
    x, m = float(trade["exports_bn"]), float(trade["imports_bn"])
    return {"year": year, "exports_bn": x, "imports_bn": m, "gdp_bn": gdp,
            "import_share": m / (x + m), "gamma": (x + m) / gdp, "imbalance0": (m - x) / gdp}


def trade_share_range() -> pd.DataFrame:
    """The same three objects for every sample year, for the sensitivity ranges."""
    years = load_bilateral_trade().index
    return pd.DataFrame([trade_shares(int(y)) for y in years]).set_index("year")


@lru_cache(maxsize=None)
def estimate_eta() -> dict:
    """Import-demand elasticity from the log-gross-tariff regression of the product panel."""
    from src.analysis.sector_passthrough import run_tariff_transform
    tr = run_tariff_transform()
    log_row = tr[tr["tariff_regressor"].eq("log(1+tau)")].iloc[0]
    lvl_row = tr[tr["tariff_regressor"].eq("tau = duties / customs value")].iloc[0]
    return {"eta": -float(log_row["beta"]), "se": float(log_row["se"]),
            "eta_level": -float(lvl_row["beta"]), "se_level": float(lvl_row["se"])}


def quarterly_inputs() -> pd.DataFrame:
    """Quarterly log import price p, log renminbi value e, and aggregate effective tariff tau,
    over the sample."""
    df = pd.concat([np.log(china_import_price_total_quarterly()).rename("p"),
                    load_renminbi_value_quarterly(),
                    aggregate_effective_tariff_quarterly().rename("tau")], axis=1)
    lo, hi = (pd.Period(q, freq="Q") for q in SAMPLE)
    return df[(df.index >= lo) & (df.index <= hi)].dropna()


@lru_cache(maxsize=None)
def estimate_passthrough() -> dict:
    """Exchange-rate pass-through into the dollar import price, and the implied theta.

    dp_t = a + sum_{k=0}^{K} b_k de_{t-k} + c dtau_t + u_t, quarterly first differences, HAC SE.
    The cumulative pass-through after h quarters is CPT_h = sum_{k<=h} b_k. In the model, a
    permanent change in e passes into the dollar import price as 1 - theta^h after h quarters,
    so theta = (1 - CPT_H)^(1/H) at the matching horizon H, with a delta-method standard error.
    The path also reports the theta implied at every shorter horizon, as a check of the mapping.
    """
    d = quarterly_inputs().diff()
    lags = pd.concat({f"de{k}": d["e"].shift(k) for k in range(ERPT_LAGS + 1)}, axis=1)
    X = pd.concat([lags, d["tau"].rename("dtau")], axis=1)
    data = pd.concat([d["p"], X], axis=1).dropna()
    fit = sm.OLS(data["p"], sm.add_constant(data.drop(columns="p"))).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    names = [f"de{k}" for k in range(ERPT_LAGS + 1)]
    V = fit.cov_params().loc[names, names].to_numpy()
    path = []
    for h in range(ERPT_LAGS + 1):
        w = np.array([1.0 if k <= h else 0.0 for k in range(ERPT_LAGS + 1)])
        cpt_h = float(w @ fit.params[names].to_numpy())
        path.append({"horizon": h, "cpt": cpt_h, "se": float(np.sqrt(w @ V @ w)),
                     "theta": (1.0 - cpt_h) ** (1.0 / h) if h > 0 else float("nan")})
    path = pd.DataFrame(path)
    cpt = float(path.loc[ERPT_HORIZON, "cpt"])
    cpt_se = float(path.loc[ERPT_HORIZON, "se"])
    theta = (1.0 - cpt) ** (1.0 / ERPT_HORIZON)
    theta_se = cpt_se * (1.0 / ERPT_HORIZON) * (1.0 - cpt) ** (1.0 / ERPT_HORIZON - 1.0)
    return {"theta": theta, "se": theta_se, "cpt": cpt, "cpt_se": cpt_se, "path": path,
            "n": int(fit.nobs), "tariff_coef": float(fit.params["dtau"]),
            "tariff_se": float(fit.bse["dtau"])}


@lru_cache(maxsize=None)
def estimate_rho_tau() -> dict:
    """AR(1) persistence of the quarterly aggregate effective tariff, with a constant."""
    t = quarterly_inputs()["tau"]
    fit = sm.OLS(t.iloc[1:].to_numpy(), sm.add_constant(t.iloc[:-1].to_numpy())).fit(
        cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return {"rho_tau": float(fit.params[1]), "se": float(fit.bse[1]), "n": int(fit.nobs)}


def estimates() -> dict:
    """Every data-disciplined value, keyed by the calibration field it overrides."""
    shares = trade_shares()
    return {"eta": estimate_eta()["eta"], "theta_dollar": estimate_passthrough()["theta"],
            "rho_tau": estimate_rho_tau()["rho_tau"], "import_share": shares["import_share"],
            "gamma": shares["gamma"], "imbalance0": shares["imbalance0"]}


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    e, pt, r, s = estimate_eta(), estimate_passthrough(), estimate_rho_tau(), trade_shares()
    print(f"eta = {e['eta']:.3f} (se {e['se']:.3f}); level-tariff version {e['eta_level']:.3f}")
    print(f"pass-through: CPT_{ERPT_HORIZON} = {pt['cpt']:.3f} (se {pt['cpt_se']:.3f}), n = {pt['n']}")
    print(pt["path"].to_string(index=False))
    print(f"theta = {pt['theta']:.3f} (se {pt['se']:.3f})")
    print(f"rho_tau = {r['rho_tau']:.3f} (se {r['se']:.3f}), n = {r['n']}")
    print(f"{BASE_YEAR}: s = {s['import_share']:.3f}, gamma = {s['gamma']:.4f}, "
          f"b = {s['imbalance0']:.4f}")
    print(trade_share_range()[["import_share", "gamma", "imbalance0"]].to_string())
