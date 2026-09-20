"""Border-price pass-through (H1 price side): does the tariff move the dollar border price?

On the NAICS x month panel (src/data/sector_prices), estimates

    log P_{n,t} = alpha_n + delta_t + beta * tau_{n,t} + u_{n,t}

with NAICS and month fixed effects and SE clustered by NAICS (twfe_cluster). P is the BLS
dollar import price index of Chinese-origin goods (pre-duty border price); tau is the Census
effective tariff. H1 predicts beta ~ 0: the tariff is NOT offset in the dollar border price,
so it passes through to the tariff-inclusive price paid by US importers. Combined with the
sector event study (import value/quantity falls), a near-zero beta here completes the H1
decomposition: the adjustment is in quantities, not in a border-price offset.
"""
from __future__ import annotations

import pandas as pd

from src.common.twfe import twfe_cluster, wild_cluster_bootstrap
from src.data.sector_prices import load_price_tariff_panel


def run() -> dict:
    return twfe_cluster(load_price_tariff_panel(), "log_price", "effective_tariff",
                        "naics", "period", cluster="naics")


def run_wcb() -> dict:
    """Wild-cluster bootstrap p-value — reliable inference with only ~22 NAICS clusters."""
    return wild_cluster_bootstrap(load_price_tariff_panel(), "log_price", "effective_tariff",
                                  "naics", "period", cluster="naics")


if __name__ == "__main__":
    p = load_price_tariff_panel()
    r = run()
    print("Border-price pass-through, TWFE (NAICS + month FE, SE clustered by NAICS)")
    print("  log(BLS China import price index) ~ beta * effective_tariff\n")
    print(f"  beta = {r['beta']:.4f}  se = {r['se']:.4f}  p = {r['p']:.4f}")
    print(f"  N = {r['n']} NAICS-months, NAICS = {r['n_fe1']}, months = {r['n_fe2']}\n")
    print("H1 reading: beta near 0 (not significantly negative) => the dollar BORDER price of")
    print("Chinese imports does not fall with the tariff, i.e. no border-price offset; the tariff")
    print("passes through to US importers and the adjustment shows up in quantities/value")
    print("(sector event study), not in the border price. A large negative beta would instead")
    print("indicate Chinese exporters cutting dollar prices to absorb the tariff.")
    wcb = run_wcb()
    print(f"\nWild-cluster bootstrap ({wcb['B']} reps, {wcb['n_cluster']} NAICS clusters):")
    print(f"  beta = {wcb['beta']:.4f}  analytic t = {wcb['t']:.3f}  bootstrap p = {wcb['p_wcb']:.3f}")
    print("With few clusters the wild-cluster bootstrap is the reliable inference. A bootstrap p")
    print("well above 0.05 confirms the border price does not respond to the tariff: NO OFFSET")
    print("(the H1 result is the sign/non-response, robust to the small number of clusters).")
