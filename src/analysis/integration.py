"""Phase 4 (first increment): discipline the structural block with empirical elasticities.

The sector pass-through panel identifies the semi-elasticity of US import value from China to
the effective tariff (beta ~ -2.18). Under H1 the dollar border price is flat in the tariff, so
that value response provides a reduced-form elasticity discipline for the rebalancing block:

    ln value = (1 - eta) ln p - eta ln(1+tau);  with d ln p / d tau ~ 0  =>  d ln value / d tau ~ -eta,

so the level-tariff beta is a local approximation, while a log(1+tau) regressor gives the
closer Armington mapping. Feeding these empirical elasticity disciplines into the DCP
rebalancing block checks whether Conclusion 1 rests on an assumed eta.

This is a reduced-form mapping (single elasticity from a single reduced-form coefficient); the
full structural estimation would target the model's cross-equation restrictions. It is stated
as such.
"""
from __future__ import annotations

import pandas as pd

from src.analysis.sector_passthrough import run as sector_run, run_tariff_transform
from src.engine.calibration import BASELINE
from src.engine.model import rebalancing_power, required_depreciation, is_feasible, replace


def data_implied_elasticities() -> pd.DataFrame:
    """Reduced-form elasticity disciplines implied by alternative tariff regressors."""
    beta_tau = float(sector_run().loc[0, "beta"])  # log import value ~ effective tariff
    transforms = run_tariff_transform()
    beta_log1p = float(transforms.loc[
        transforms["tariff_regressor"].eq("log(1+tau)"), "beta"
    ].iloc[0])
    return pd.DataFrame([
        {
            "source": "level tau local approximation",
            "beta": beta_tau,
            "eta_hat": abs(beta_tau),
            "mapping": "d log value / d tau ~= -eta locally",
        },
        {
            "source": "log(1+tau) Armington mapping",
            "beta": beta_log1p,
            "eta_hat": abs(beta_log1p),
            "mapping": "d log value / d log(1+tau) ~= -eta",
        },
    ])


def data_implied_eta() -> float:
    """Backward-compatible primary eta: the level-tau local approximation."""
    return float(data_implied_elasticities().loc[0, "eta_hat"])


def disciplined_conclusion1() -> pd.DataFrame:
    elasticities = data_implied_elasticities()
    scenarios = [("baseline eta=1.5 (assumed)", "calibration", BASELINE)]
    for row in elasticities.itertuples(index=False):
        scenarios.extend([
            (
                f"{row.source}: eta only ({row.eta_hat:.2f})",
                row.mapping,
                replace(BASELINE, eta=row.eta_hat),
            ),
            (
                f"{row.source}: eta and eta_star ({row.eta_hat:.2f})",
                f"{row.mapping}; symmetric foreign elasticity imposed",
                replace(BASELINE, eta=row.eta_hat, eta_star=row.eta_hat),
            ),
        ])
    rows = []
    for name, mapping, p in scenarios:
        rows.append({"scenario": name, "mapping": mapping, "eta": p.eta, "eta_star": p.eta_star,
                     "R_observed": rebalancing_power(p),
                     "required_deprec": required_depreciation(p), "feasible": is_feasible(p)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("PHASE 4 (increment): structural block disciplined by empirical elasticities\n")
    print("Reduced-form elasticity inputs from the sector pass-through")
    print(data_implied_elasticities().to_string(index=False))
    print()
    print(disciplined_conclusion1().to_string(index=False))
    print("\nConclusion 1, empirically disciplined: replacing the assumed eta with reduced-form")
    print("elasticities from the 2018-19 tariff episode still leaves the required depreciation far")
    print("above any bounded-cost level. The eta-only rows are the conservative mapping; the")
    print("eta+eta_star rows impose symmetry for the foreign elasticity. This is not a full")
    print("Lucas-robust structural estimate; it is a reduced-form bridge into the DCP block.")
