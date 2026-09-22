"""Self-contained two-way fixed-effects estimator with one continuous regressor and
cluster-robust standard errors. Used for the sector and firm panels (eq:sector_panel,
eq:twfe_firm) so the reproduction does not depend on a heavy FE library.

Method: iterative alternating-projection demeaning of the outcome and the regressor by both
fixed-effect factors (exact two-way within transformation for unbalanced panels), then OLS of
the demeaned outcome on the demeaned regressor. Standard errors are clustered on `cluster`,
with a reghdfe-style small-sample adjustment. Validated against a dummy-variable OLS.

`twfe_ols` also accepts observation weights (weighted least squares). The within
transformation then uses weighted group means, which is the Frisch-Waugh-Lovell projection
for WLS, and the cluster-robust variance uses the weighted scores. This serves the corrective
weights of the stacked difference-in-differences (Wing, Freedman and Hollingsworth, 2024).
Validated against a dummy-variable WLS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _demean_col(x: np.ndarray, codes: np.ndarray, k: int,
                w: np.ndarray | None = None) -> np.ndarray:
    """Subtract the (weighted) group mean of x within each group (groups labelled 0..k-1)."""
    if w is None:
        sums = np.bincount(codes, weights=x, minlength=k)
        counts = np.bincount(codes, minlength=k)
        means = sums / np.maximum(counts, 1)
        return x - means[codes]
    sums = np.bincount(codes, weights=w * x, minlength=k)
    mass = np.bincount(codes, weights=w, minlength=k)
    means = np.divide(sums, mass, out=np.zeros(k), where=mass > 0)
    return x - means[codes]


def demean_2way(mat: np.ndarray, c1: np.ndarray, c2: np.ndarray,
                k1: int, k2: int, iters: int = 200, tol: float = 1e-10,
                w: np.ndarray | None = None) -> np.ndarray:
    """Two-way (weighted) within transformation by alternating projections until convergence."""
    X = mat.astype(float).copy()
    for _ in range(iters):
        X0 = X.copy()
        for j in range(X.shape[1]):
            X[:, j] = _demean_col(X[:, j], c1, k1, w)
            X[:, j] = _demean_col(X[:, j], c2, k2, w)
        if np.max(np.abs(X - X0)) < tol:
            break
    return X


def twfe_cluster(df: pd.DataFrame, y: str, x: str, fe1: str, fe2: str,
                 cluster: str | None = None) -> dict:
    """Two-way FE regression y ~ beta*x | fe1 + fe2, SE clustered on `cluster` (default fe1)."""
    cluster = cluster or fe1
    d = df.dropna(subset=[y, x, fe1, fe2, cluster])
    c1, k1 = pd.factorize(d[fe1])
    c2, k2 = pd.factorize(d[fe2])
    k1, k2 = len(k1), len(k2)
    M = demean_2way(np.column_stack([d[y].to_numpy(float), d[x].to_numpy(float)]), c1, c2, k1, k2)
    yt, xt = M[:, 0], M[:, 1]
    sxx = xt @ xt
    beta = (xt @ yt) / sxx
    e = yt - beta * xt

    clab, _ = pd.factorize(d[cluster])
    G = clab.max() + 1
    meat = 0.0
    for g in range(G):
        sg = xt[clab == g] @ e[clab == g]
        meat += sg * sg
    N = len(yt)
    k_params = (k1 + k2 - 1) + 1  # absorbed FE + the regressor
    adj = (G / (G - 1)) * ((N - 1) / (N - k_params))
    se = np.sqrt(adj * meat) / sxx
    t = beta / se
    p = 2 * stats.t.sf(abs(t), G - 1)
    return {"beta": beta, "se": se, "p": p, "n": N, "n_fe1": k1, "n_fe2": k2, "n_cluster": G}


def twfe_ols(df: pd.DataFrame, y: str, xs: list[str], fe1: str, fe2: str,
             cluster: str | None = None, weights: str | None = None,
             return_vcov: bool = False):
    """Two-way FE regression y ~ b*xs | fe1 + fe2 with MULTIPLE regressors, cluster-robust SE.

    Returns a frame indexed by regressor with beta, se, t, p (used for event-study leads/lags).
    With `weights` (a column of positive observation weights) the regression is weighted least
    squares; without it the unweighted path is unchanged.
    """
    cluster = cluster or fe1
    cols = [y] + list(xs)
    d = df.dropna(subset=cols + [fe1, fe2, cluster] + ([weights] if weights else []))
    c1, u1 = pd.factorize(d[fe1])
    c2, u2 = pd.factorize(d[fe2])
    k1, k2 = len(u1), len(u2)
    w = None
    if weights:
        w = d[weights].to_numpy(float)
        if np.any(w <= 0):
            raise ValueError("observation weights must be strictly positive")
    M = demean_2way(d[cols].to_numpy(float), c1, c2, k1, k2, w=w)
    yt, X = M[:, 0], M[:, 1:]
    Xw = X if w is None else X * w[:, None]
    XtX = Xw.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (Xw.T @ yt)
    e = yt - X @ beta
    clab, uc = pd.factorize(d[cluster])
    G = len(uc)
    K = X.shape[1]
    meat = np.zeros((K, K))
    for g in range(G):
        Xg = Xw[clab == g]
        sg = Xg.T @ e[clab == g]
        meat += np.outer(sg, sg)
    N = len(yt)
    adj = (G / (G - 1)) * ((N - 1) / (N - (k1 + k2 - 1) - K))
    V = adj * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.diag(V))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), G - 1)
    out = pd.DataFrame({"beta": beta, "se": se, "t": t, "p": p}, index=list(xs))
    if return_vcov:
        return out, pd.DataFrame(V, index=list(xs), columns=list(xs)), G
    return out


def _cluster_t(xt: np.ndarray, yv: np.ndarray, clab: np.ndarray, G: int,
               sxx: float, adj: float) -> tuple[float, float]:
    """Slope and cluster-robust t of yv on xt in the (already two-way-demeaned) space."""
    beta = (xt @ yv) / sxx
    e = yv - beta * xt
    meat = 0.0
    for g in range(G):
        sg = xt[clab == g] @ e[clab == g]
        meat += sg * sg
    se = np.sqrt(adj * meat) / sxx
    return beta, beta / se


def wild_cluster_bootstrap(df: pd.DataFrame, y: str, x: str, fe1: str, fe2: str,
                           cluster: str | None = None, B: int = 1999,
                           seed: int = 20260921) -> dict:
    """Restricted wild-cluster bootstrap (Cameron-Gelbach-Miller) p-value for H0: beta_x = 0.

    Reliable inference with FEW clusters (the analytic cluster-robust t is unreliable at ~20
    clusters). Works in the two-way-demeaned space; imposes the null; resamples cluster-level
    Rademacher signs on the restricted residuals. Deterministic given `seed`.
    """
    cluster = cluster or fe1
    d = df.dropna(subset=[y, x, fe1, fe2, cluster])
    c1, u1 = pd.factorize(d[fe1])
    c2, u2 = pd.factorize(d[fe2])
    k1, k2 = len(u1), len(u2)
    M = demean_2way(np.column_stack([d[y].to_numpy(float), d[x].to_numpy(float)]), c1, c2, k1, k2)
    yt, xt = M[:, 0], M[:, 1]
    clab, uc = pd.factorize(d[cluster])
    G = len(uc)
    N = len(yt)
    sxx = xt @ xt
    adj = (G / (G - 1)) * ((N - 1) / (N - (k1 + k2 - 1) - 1))
    beta_hat, t_hat = _cluster_t(xt, yt, clab, G, sxx, adj)

    # H0: beta=0 -> restricted residual in the demeaned space is yt itself; resample its signs.
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(B):
        w = rng.choice((-1.0, 1.0), size=G)[clab]
        _, t_star = _cluster_t(xt, w * yt, clab, G, sxx, adj)
        if abs(t_star) >= abs(t_hat):
            count += 1
    return {"beta": beta_hat, "t": t_hat, "p_wcb": (count + 1) / (B + 1),
            "B": B, "n_cluster": G, "n": N}


def validate_against_statsmodels() -> None:
    """Smoke-test the estimator against dummy-variable OLS with clustered SE.

    This keeps the validation claim versioned without making statsmodels part of the
    estimator itself. The synthetic panel is deterministic and unbalanced enough to exercise
    the alternating-projection within transformation.
    """
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(20260921)
    n_firms, n_times = 45, 16
    idx = pd.MultiIndex.from_product(
        [range(n_firms), range(n_times)], names=["unit", "time"]
    ).to_frame(index=False)
    idx = idx[rng.random(len(idx)) > 0.12].reset_index(drop=True)
    unit_fe = rng.normal(size=n_firms)
    time_fe = rng.normal(size=n_times)
    x = rng.normal(size=len(idx)) + 0.15 * idx["unit"].to_numpy() / n_firms
    y = 1.75 * x + unit_fe[idx["unit"].to_numpy()] + time_fe[idx["time"].to_numpy()]
    y = y + rng.normal(scale=0.5, size=len(idx))
    df = idx.assign(x=x, y=y, unit_s=idx["unit"].astype(str), time_s=idx["time"].astype(str))

    ours = twfe_cluster(df, "y", "x", "unit", "time")
    sm = smf.ols("y ~ x + C(unit_s) + C(time_s)", data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["unit_s"], "use_correction": True},
    )
    beta_diff = abs(ours["beta"] - sm.params["x"])
    se_diff = abs(ours["se"] - sm.bse["x"])
    print(f"ours: beta={ours['beta']:.8f} se={ours['se']:.8f}")
    print(f"sm   : beta={sm.params['x']:.8f} se={sm.bse['x']:.8f}")
    print(f"diff : beta={beta_diff:.3g} se={se_diff:.3g}")
    assert beta_diff < 1e-10
    assert se_diff < 1e-10
    print("twfe validation OK")


def validate_multi_against_statsmodels() -> None:
    """Validate twfe_ols (multiple regressors) against dummy-variable OLS with clustered SE."""
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(7)
    idx = pd.MultiIndex.from_product([range(50), range(14)], names=["unit", "time"]).to_frame(index=False)
    idx = idx[rng.random(len(idx)) > 0.1].reset_index(drop=True)
    x1 = rng.normal(size=len(idx))
    x2 = rng.normal(size=len(idx))
    y = 1.2 * x1 - 0.7 * x2 + rng.normal(size=50)[idx["unit"]] + rng.normal(size=14)[idx["time"]] \
        + rng.normal(scale=0.4, size=len(idx))
    df = idx.assign(x1=x1, x2=x2, y=y, us=idx["unit"].astype(str), ts=idx["time"].astype(str))
    ours = twfe_ols(df, "y", ["x1", "x2"], "unit", "time")
    sm = smf.ols("y ~ x1 + x2 + C(us) + C(ts)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["us"], "use_correction": True})
    for term in ["x1", "x2"]:
        assert abs(ours.loc[term, "beta"] - sm.params[term]) < 1e-9
        assert abs(ours.loc[term, "se"] - sm.bse[term]) < 1e-9
    print("twfe_ols (multi-regressor) validation OK")


def validate_weighted_against_statsmodels() -> None:
    """Validate weighted twfe_ols against dummy-variable WLS with clustered SE."""
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(11)
    idx = pd.MultiIndex.from_product([range(40), range(12)], names=["unit", "time"]).to_frame(index=False)
    idx = idx[rng.random(len(idx)) > 0.1].reset_index(drop=True)
    x1 = rng.normal(size=len(idx))
    x2 = rng.normal(size=len(idx))
    y = 0.8 * x1 + 0.3 * x2 + rng.normal(size=40)[idx["unit"]] + rng.normal(size=12)[idx["time"]] \
        + rng.normal(scale=0.5, size=len(idx))
    wts = rng.uniform(0.2, 3.0, size=len(idx))
    df = idx.assign(x1=x1, x2=x2, y=y, w=wts, us=idx["unit"].astype(str), ts=idx["time"].astype(str))
    ours = twfe_ols(df, "y", ["x1", "x2"], "unit", "time", weights="w")
    sm = smf.wls("y ~ x1 + x2 + C(us) + C(ts)", data=df, weights=df["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": df["us"], "use_correction": True})
    for term in ["x1", "x2"]:
        assert abs(ours.loc[term, "beta"] - sm.params[term]) < 1e-9
        assert abs(ours.loc[term, "se"] - sm.bse[term]) < 1e-9
    print("twfe_ols (weighted) validation OK")


if __name__ == "__main__":
    validate_against_statsmodels()
    validate_multi_against_statsmodels()
    validate_weighted_against_statsmodels()
