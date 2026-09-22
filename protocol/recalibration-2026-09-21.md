# Structural recalibration, 2026-09-21

The structural baseline now uses every parameter the project's data identify
(src/analysis/parameter_estimation.py, src/analysis/baseline.py): the import-demand
elasticity (log-gross-tariff product panel), the dollar-invoicing friction (exchange-rate
pass-through), the tariff persistence (AR(1)), and the import share, bilateral openness and
initial imbalance computed from 2017 Census trade and BEA GDP. The impact bound is
generalized to unbalanced trade (src/engine/model.py). Empirical estimates are unchanged;
only structural outputs move.

## Unchanged empirical keys

- `sector_value_beta` = -2.1791
- `border_price_beta` = 0.2383
- `border_price_p` = 0.169
- `border_price_wcb_p` = 0.141
- `data_implied_eta` = 2.1791
- `staggered_post_mean_beta` = -0.299
- `staggered_post_mean_beta_weighted` = -0.297
- `event_post_mean_baseline` = -1.509
- `event_post_mean_reference_2017Q4` = -1.665
- `event_post_mean_split_sample` = -1.985

## Changed structural keys (old -> new)

- `ge_efficiency_theta_dollar_0`: 0.7899 -> 0.3599
- `ge_efficiency_chi_0`: 1.7081 -> 1.1297
- `ge_efficiency_chi_4`: 0.1209 -> 0.2046
- `ge_sensitivity_determinate`: 11/13 -> 18/18
- `ge_sensitivity_invoicing_ratio_range`: [1.061, 1.381] -> [0.747, 1.278]
- `ge_sensitivity_wedge_ratio_range`: [0.033, 0.139] -> [0.059, 0.751]

## Removed keys (replaced by the estimated-baseline keys)

- `reduced_form_required_deprec_observed` = 3.0
- `reduced_form_feasible_observed` = False
- `ge_efficiency_theta_dollar_095` = 0.8916

## Added keys

- `estimated_eta` = 2.5255
- `estimated_theta_dollar` = 0.8851
- `estimated_rho_tau` = 0.963
- `estimated_import_share` = 0.7953
- `estimated_gamma` = 0.0324
- `estimated_imbalance0` = 0.0191
- `impact_bound_required_deprec_estimated` = 6.762
- `impact_bound_required_deprec_no_friction` = 0.389
- `impact_bound_feasible_estimated` = False
- `ge_efficiency_theta_dollar_estimated` = 0.2941

## Addition: Monte Carlo propagation of parameter uncertainty (same day)

`src/analysis/parameter_uncertainty.py` propagates parameter uncertainty with 10,000 draws per
layer (seed 20260921). The first layer draws only the estimated parameters (eta, theta_dollar)
from their sampling distributions. The second also draws the trade ratios over their 2015-2021
range and the calibrated parameters over documented ranges. No existing key changed; the
following keys were added:

- `mc_estimation_share_ranking` = 1.0
- `mc_estimation_share_determinate` = 1.0
- `mc_estimation_invoicing_effect_median` = -0.164
- `mc_estimation_wedge_effect_median` = -0.736
- `mc_estimation_share_feasible` = 0.0
- `mc_full_share_ranking` = 0.9852
- `mc_full_share_determinate` = 0.9659
- `mc_full_invoicing_effect_median` = 0.045
- `mc_full_wedge_effect_median` = -0.476
- `mc_full_share_feasible` = 0.0
