"""Compact two-country New-Keynesian model with dominant-currency pricing (Phase 2, dynamic).

A calibrated log-linear open-economy NK model solved with the validated Klein solver
(src/engine/linre). It is deliberately compact (not the full 20-equation NOEM) but genuine:
it carries the mechanism the paper needs — dollar-price stickiness (DCP), a modified UIP with a
capital-controls/portfolio wedge, NFA dynamics, and a bilateral net-export block — so we can
ask the H3 dynamic question: does an engineered weaker dollar (RMB appreciation) improve the
bilateral balance, and at what output-gap cost, as theta_dollar and chi vary?

Variables x_t = [ d, pm, tau, z | y, pi, e, nx ] (deviations from steady state):
  states k (predetermined): d = NFA, pm = dollar import price (DCP, sticky), tau = tariff (AR1),
    z = engineered-depreciation / UIP shock (AR1).
  jumps u: y = home output gap, pi = home inflation, e = RMB value (e>0 = RMB appreciation = weaker dollar),
    nx = bilateral net exports.

Sign convention: e is the RMB value (the framework's E, USD per RMB). e>0 = RMB appreciation =
WEAKER DOLLAR; e<0 = RMB depreciation. As solved, the z shock engineers a weaker dollar:
z=+1 raises the equilibrium e (RMB appreciates), i.e. the Mar-a-Lago experiment. (Verified in
the IRF: z=+1 -> e>0.)

Equations (A E_t x_{t+1} = B x_t):
  d_{t+1} = (1/beta) d_t + nx_t                                   (NFA accumulation)
  pm_{t+1} = theta_$ pm_t + (1-theta_$) E_t e_{t+1}               (DCP: predetermined dollar price;
                                                                  a share 1-theta_$ resets each quarter)
  tau_{t+1} = rho_tau tau_t ; z_{t+1} = rho_z z_t                 (exogenous AR1)
  (1+phi_y/sigma) y_t + (phi_pi/sigma) pi_t - gamma nx_t = E y_{t+1} + (1/sigma) E pi_{t+1}
                                                                  (IS + Taylor, with net exports)
  beta E pi_{t+1} = pi_t - kappa y_t                              (NK Phillips curve)
  E e_{t+1} = phi_pi pi_t + phi_y y_t + e_t + Phi(1+chi) d_t - z_t (modified UIP + Taylor rate)
  nx_t = s_x eta_star e_t + s_m (eta-1) pm_t + s_m eta tau_t      (bilateral net exports relative to
                                                                  steady-state bilateral trade;
                                                                  s_m = import_share, s_x = 1 - s_m)
The manuscript writes the same system with omega for beta, r_t for the policy rate, n_t for nx_t,
p^m_t for pm_t, s for import_share, and psi for portfolio_cost (section "Structural Model").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.engine.calibration import Params, BASELINE
from src.engine.linre import solve, Solution

VARS = ["d", "pm", "tau", "z", "y", "pi", "e", "nx"]
IDX = {v: i for i, v in enumerate(VARS)}
NK = 4  # predetermined: d, pm, tau, z


def build_system(p: Params = BASELINE) -> tuple[np.ndarray, np.ndarray]:
    """Assemble A, B for A E_t x_{t+1} = B x_t."""
    n = len(VARS)
    A = np.zeros((n, n)); B = np.zeros((n, n))
    i = IDX
    kap = p.kappa

    # d_{t+1} = (1/beta) d_t + nx_t
    A[0, i["d"]] = 1.0
    B[0, i["d"]] = 1.0 / p.beta; B[0, i["nx"]] = 1.0
    # pm_{t+1} - (1-theta_$) e_{t+1} = theta_$ pm_t
    A[1, i["pm"]] = 1.0; A[1, i["e"]] = -(1.0 - p.theta_dollar)
    B[1, i["pm"]] = p.theta_dollar
    # tau_{t+1} = rho_tau tau_t
    A[2, i["tau"]] = 1.0; B[2, i["tau"]] = p.rho_tau
    # z_{t+1} = rho_z z_t
    A[3, i["z"]] = 1.0; B[3, i["z"]] = p.rho_z
    # Open-economy IS + Taylor, with net exports in aggregate demand (coupling nx -> y):
    #   y_t = E y_{t+1} - (1/sigma)(i_t - E pi_{t+1}) + gamma*nx_t,  i_t = phi_pi pi + phi_y y
    #   => (1+phi_y/sigma) y_t + (phi_pi/sigma) pi_t - gamma*nx_t = E y_{t+1} + (1/sigma) E pi_{t+1}
    A[4, i["y"]] = 1.0; A[4, i["pi"]] = 1.0 / p.sigma
    B[4, i["y"]] = 1.0 + p.phi_y / p.sigma; B[4, i["pi"]] = p.phi_pi / p.sigma
    B[4, i["nx"]] = -p.gamma
    # NKPC: beta E pi_{t+1} = pi_t - kappa y_t
    A[5, i["pi"]] = p.beta
    B[5, i["pi"]] = 1.0; B[5, i["y"]] = -kap
    # modified UIP: E e_{t+1} = phi_pi pi_t + phi_y y_t + e_t + Phi(1+chi) d_t - z_t
    A[6, i["e"]] = 1.0
    B[6, i["pi"]] = p.phi_pi; B[6, i["y"]] = p.phi_y; B[6, i["e"]] = 1.0
    B[6, i["d"]] = p.portfolio_cost * (1.0 + p.chi); B[6, i["z"]] = -1.0
    # Net exports (static), trade-share weighted so the DCP-muted IMPORT channel dominates:
    #   export value (share s_x): +s_x*eta_star*e   [weaker dollar -> RMB price of US goods falls]
    #   import value (share s_m): (eta-1)*pm + eta*tau, muted because pm is sticky in e under DCP
    #     -> the exchange rate reaches imports only slowly via pm.
    s_m = p.import_share; s_x = 1.0 - p.import_share
    B[7, i["nx"]] = -1.0; B[7, i["e"]] = s_x * p.eta_star
    B[7, i["pm"]] = s_m * (p.eta - 1.0); B[7, i["tau"]] = s_m * p.eta
    return A, B


def solve_model(p: Params = BASELINE) -> Solution:
    A, B = build_system(p)
    return solve(A, B, nk=NK)


def irf(shock: str, periods: int = 24, size: float = 1.0, p: Params = BASELINE) -> pd.DataFrame:
    """Impulse responses to a unit shock in a state ('tau' or 'z')."""
    sol = solve_model(p)
    k = np.zeros(NK)
    k[IDX[shock]] = size  # innovation to the exogenous/predetermined state
    rows = []
    for t in range(periods):
        u = sol.F @ k
        row = {**{VARS[j]: k[j] for j in range(NK)},
               **{VARS[NK + j]: u[j] for j in range(len(u))}}
        rows.append(row)
        k = sol.P @ k
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda x: f"{x:+.4f}")
    sol = solve_model()
    print("DSGE solved. Blanchard-Kahn OK (stable eigenvalues = predetermined states =", sol.nk, ")\n")

    print("IRF to a TARIFF shock (tau=+1): expect import price pm up, net exports nx up (imports fall)")
    t = irf("tau", periods=6)
    print(t[["tau", "pm", "nx", "y", "pi", "e"]].head(6).to_string())

    print("\nIRF to an ENGINEERED WEAKER DOLLAR (z=+1 -> RMB appreciates, e>0):")
    print("  under DCP the weaker dollar still boosts US exports (RMB price of US goods falls),")
    print("  while the import channel is muted (pm sticky); nx improves, at an output-gap cost")
    z = irf("z", periods=6)
    print(z[["z", "e", "pm", "nx", "y"]].head(6).to_string())
