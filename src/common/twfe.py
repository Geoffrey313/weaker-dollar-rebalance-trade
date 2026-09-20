"""Self-contained two-way fixed-effects estimator with one continuous regressor and
cluster-robust standard errors. Used for the sector and firm panels (eq:sector_panel,
eq:twfe_firm) so the reproduction does not depend on a heavy FE library.

Method: iterative alternating-projection demeaning of the outcome and the regressor by both
fixed-effect factors (exact two-way within transformation for unbalanced panels), then OLS of
the demeaned outcome on the demeaned regressor. Standard errors are clustered on `cluster`,
with a reghdfe-style small-sample adjustment. Validated against a dummy-variable OLS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _demean_col(x: np.ndarray, codes: np.ndarray, k: int) -> np.ndarray:
    """Subtract the group mean of x within each group (groups labelled 0..k-1)."""
    sums = np.bincount(codes, weights=x, minlength=k)
    counts = np.bincount(codes, minlength=k)
    means = sums / np.maximum(counts, 1)
    return x - means[codes]


def demean_2way(mat: np.ndarray, c1: np.ndarray, c2: np.ndarray,
                k1: int, k2: int, iters: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Two-way within transformation by alternating projections until convergence."""
    X = mat.astype(float).copy()
    for _ in range(iters):
        X0 = X.copy()
        for j in range(X.shape[1]):
            X[:, j] = _demean_col(X[:, j], c1, k1)
            X[:, j] = _demean_col(X[:, j], c2, k2)
        if np.max(np.abs(X - X0)) < tol:
            break
    return X


def twfe_cluster(df: pd.DataFrame, y: str, x: str, fe1: str, fe2: str,
                 cluster: str | None = None) -> dict:
    """Two-way FE regression y ~ beta*x | fe1 + fe2, SE clustered on `cluster` (default fe1)."""
    cluster = cluster or fe1
    d = df.dropna(subset=[y, x, fe1, fe2, cluster])
    c1, k1 = pd.factorize(d[fe1]); c2, k2 = pd.factorize(d[fe2])
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
