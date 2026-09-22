"""Phase 4 (first increment): discipline the structural block with empirical elasticities.

The sector pass-through panel identifies the semi-elasticity of US import value from China to
the effective tariff (beta ~ -2.18). Under H1 the dollar border price is flat in the tariff, so
that value response provides a reduced-form elasticity discipline for the rebalancing block:

    ln value = (1 - eta) ln p - eta ln(1+tau);  with d ln p / d tau ~ 0  =>  d ln value / d tau ~ -eta,

so the level-tariff beta is a local approximation, while a log(1+tau) regressor gives the
closer Armington mapping. Feeding these empirical elasticity disciplines into the DCP
rebalancing block checks whether the H2 impact bound rests on an assumed eta.

This is a reduced-form mapping (single elasticity from a single reduced-form coefficient); the
full structural estimation would target the model's cross-equation restrictions. It is stated
as such.
"""
from __future__ import annotations

import pandas as pd

from src.analysis.sector_passthrough import run as sector_run, run_tariff_transform
from src.engine.model import rebalancing_power, required_depreciation, is_feasible


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


def disciplined_h2() -> pd.DataFrame:
    """The impact bound under the literature calibration and under the data-disciplined baseline."""
    from src.analysis.baseline import baseline
    from src.engine.calibration import LITERATURE
    rows = []
    for name, p in (("literature calibration", LITERATURE), ("data-disciplined baseline", baseline())):
        rows.append({"scenario": name, "eta": p.eta, "eta_star": p.eta_star, "gamma": p.gamma,
                     "import_share": p.import_share, "imbalance0": p.imbalance0,
                     "theta_dollar": p.theta_dollar, "R_observed": rebalancing_power(p),
                     "required_deprec": required_depreciation(p), "feasible": is_feasible(p)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("Structural block: literature calibration versus data-disciplined baseline\n")
    print("Reduced-form elasticity inputs from the sector pass-through")
    print(data_implied_elasticities().to_string(index=False))
    print()
    print(disciplined_h2().to_string(index=False))
    print("\nThe data-disciplined baseline replaces the literature values of eta, the invoicing")
    print("friction, the tariff persistence, the import share, bilateral openness and the initial")
    print("imbalance with estimates from the project's data (src.analysis.parameter_estimation).")
