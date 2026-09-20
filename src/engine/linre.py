"""Linear rational-expectations solver (Klein 2000, generalized Schur / QZ).

Solves systems written as

    A E_t x_{t+1} = B x_t + C eps_t,     x_t = [k_t (predetermined); u_t (jump)]

returning the decision rule u_t = F k_t and the state law of motion k_{t+1} = P k_t (+ shocks),
using the ordered generalized Schur (QZ) decomposition with stable generalized eigenvalues
(|lambda| < 1) sorted first. Requires the Blanchard-Kahn count: #stable eigenvalues = #k.

Validated in `validate()` against a hand-solved forward-looking model (u_t = [b/(1-a*rho)] k_t).
This is the engine that solves the two-country DCP model in src/engine/dsge.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import ordqz


@dataclass
class Solution:
    P: np.ndarray  # k_{t+1} = P k_t   (nk x nk)
    F: np.ndarray  # u_t      = F k_t   (nu x nk)
    nk: int
    stable: int


def solve(A: np.ndarray, B: np.ndarray, nk: int, tol: float = 1e-9) -> Solution:
    """Solve A E_t x_{t+1} = B x_t for the stable saddle-path rule (x=[k (nk); u])."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    n = A.shape[0]
    # For A E_t x_{t+1} = B x_t the dynamic multipliers are the eigenvalues of the pencil (B, A):
    # lambda_dyn = T_ii/S_ii. scipy's ordqz(A, B) reports alpha/beta = eig of (A, B) = 1/lambda_dyn,
    # so a dynamically STABLE mode (|lambda_dyn|<1) has scipy |alpha/beta|>1 -> sort 'ouc' puts the
    # stable subspace top-left. S=AA (from A), T=BB (from B); Z-partition gives the saddle-path rule.
    S, T, alpha, beta, Q, Z = ordqz(A, B, sort="ouc", output="real")
    with np.errstate(divide="ignore", invalid="ignore"):
        lam_dyn = np.abs(beta / alpha)  # |T/S| = |1/(alpha/beta)|
    stable = int(np.sum(lam_dyn < 1 - tol))
    if stable != nk:
        raise ValueError(f"Blanchard-Kahn failed: {stable} stable eigenvalues != {nk} predetermined "
                         f"(dynamic |lambda|={np.sort(lam_dyn)}).")
    Z11 = Z[:nk, :nk]; Z21 = Z[nk:, :nk]
    S11 = S[:nk, :nk]; T11 = T[:nk, :nk]
    Z11_inv = np.linalg.inv(Z11)
    F = Z21 @ Z11_inv                                  # u_t = F k_t
    P = Z11 @ np.linalg.solve(S11, T11) @ Z11_inv       # k_{t+1} = P k_t
    return Solution(P=np.real(P), F=np.real(F), nk=nk, stable=stable)


def validate() -> None:
    """Hand-solved check: k_{t+1}=rho k_t (predetermined), u_t = a E_t u_{t+1} + b k_t (jump).

    Guess u_t = c k_t => c = a c rho + b => c = b/(1-a*rho); and P = rho.
    In A E_t x_{t+1} = B x_t form (x=[k;u]):
        A = [[1,0],[0,a]],  B = [[rho,0],[-b,1]].
    """
    for rho, a, b in [(0.8, 0.5, 1.0), (0.5, 0.3, -2.0), (0.95, 0.6, 0.4)]:
        A = np.array([[1.0, 0.0], [0.0, a]])
        B = np.array([[rho, 0.0], [-b, 1.0]])
        sol = solve(A, B, nk=1)
        c_true = b / (1 - a * rho)
        assert abs(sol.F[0, 0] - c_true) < 1e-9, (sol.F[0, 0], c_true)
        assert abs(sol.P[0, 0] - rho) < 1e-9, (sol.P[0, 0], rho)
    print("linre.solve validation OK (F and P match the hand-solved forward-looking model)")


if __name__ == "__main__":
    validate()
