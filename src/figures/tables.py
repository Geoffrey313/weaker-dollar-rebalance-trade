"""Results tables and in-text numbers of the paper, in English and French.

Inputs: the analysis modules (product and border-price panels, event studies, stacked
difference-in-differences, impact bound, general-equilibrium counterfactual) and the model
calibration. Outputs: LaTeX tabular fragments and a file of number macros in
manuscript/<lang>/ssrn/tables/, one set per language. The manuscripts \\input these files, so
every number in a table or in the prose comes from this script and is never typed by hand.
Captions and notes live in the manuscripts; the fragments hold only the numbers.

Serves: Table 1 (H1 incidence), Table 2 (staggered robustness), Table 3 (sample
restrictions), Table 4 (H2 impact bound), Table 5 (H3 general equilibrium), and the appendix
tables (event-study coefficients, general-equilibrium sensitivity, calibration, descriptive
statistics).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from src.common.paths import DATA_DIR, PROJECT_ROOT
from src.figures.latex import (PARAMETER_SYMBOLS, coef, integer, macro, num, raw_num, se,
                               write_fragment)

LANGS = ("en", "fr")
TXT = {
    "en": {
        "dep": "Dependent variable", "logv": "Log import | value $v_{i,t}$",
        "logp": "Log border | price $\\ln P_{j,t}$", "tau": "Effective tariff $\\tau$",
        "log1p": "Log gross tariff $\\ln(1+\\tau)$", "p_cl": "Cluster-robust $p$-value",
        "p_wcb": "Wild-cluster bootstrap $p$-value", "fe_unit": "Product or industry fixed effects",
        "fe_time": "Month fixed effects", "cl_unit": "Clustering unit", "product": "Product",
        "industry": "Industry", "clusters": "Clusters", "months": "Months",
        "obs": "Observations", "yes": "Yes", "unw": "Unweighted", "wtd": "Weighted",
        "panel_a_stack": "Panel A. Event-time coefficients $\\beta^{S}_k$",
        "panel_b_stack": "Panel B. Mean post-treatment effect by treatment cutoff",
        "panel_c_stack": "Panel C. Cohort samples",
        "ref": "reference", "pp": "percentage points",
        "stacked_obs": "Product-quarter-cohort obs.",
        "stacks": "Treatment cohorts", "treated": "Treated products",
        "controls": "Never-treated controls per cohort sample",
        "min_cell": "Minimum cell value (US dollars)", "coef_se": "Coefficient $\\beta^{V}$",
        "rb_R": "Rebalancing | power $R$", "rb_req": "Required | appreciation (\\%)",
        "no": "No", "panel_a_rb": "Panel A. Trade elasticity $\\eta=\\eta^{\\ast}$",
        "panel_b_rb": "Panel B. One-way calibration changes",
        "data_level": "data, level tariff", "data_log": "data, log gross tariff",
        "baseline": "baseline", "gamma": "Openness $\\gamma", "imb": "Initial imbalance $\\bar{b}",
        "emax": "Tolerated appreciation $\\bar{e}", "mu": "Damping scale $\\mu",
        "panel_a_ge": "Panel A. Dollar-invoicing friction $\\theta$ (at $\\chi=1$)",
        "panel_b_ge": "Panel B. Capital-controls wedge $\\chi$ (at the estimated $\\theta$)",
        "estimated": "estimated", "rb_fr": "Panel A. Frictions", "rb_el": "Panel B. Trade elasticities",
        "rb_ow": "Panel C. One-way changes", "sc_obs": "Estimated frictions",
        "sc_noinv": "No invoicing friction, $\\theta=0$", "sc_open": "Open account, $\\chi=0$",
        "sc_none": "No friction, $\\theta=0$ and $\\chi=0$", "el_base": "Baseline",
        "el_level": "Level-tariff estimate of $\\eta$", "el_star": "Foreign elasticity",
        "par_est": "Panel A. Estimated",
        "par_comp": "Panel B. Computed from trade and output data, 2017",
        "par_cal": "Panel C. Calibrated", "method": "Method or source",
        "horizon": "Horizon (quarters)", "cpt": "Cumulative pass-through $\\zeta_h$",
        "implied": "Implied friction $\\theta_h$",
        "ow_s": "Import share $s", "ow_theta": "Invoicing friction $\\theta",
        "peak": "Peak | appreciation", "nxcum": "Net | exports", "gapcum": "Output-gap | cost",
        "eff": "Efficiency $\\Lambda$", "quarter": "Quarter", "calib": "Calibration",
        "ge_inv": "Invoicing gradient", "ge_wedge": "Wedge gradient", "indet": "no unique stable solution",
        "param": "Parameter", "symbol": "Symbol", "value": "Value", "source": "Source",
        "evtime": "Event time", "coef": "Coefficient", "stderr": "Standard error", "pval": "$p$-value",
        "id_base": "Baseline", "id_ref": "Reference 2017Q4", "id_split": "Split sample",
        "id_refq": "Reference quarter", "id_q1": "Coefficient, 2018Q1", "id_q2": "Coefficient, 2018Q2",
        "id_presig": "Pre-episode coefficients significant at 5\\%", "id_maxt": "Largest pre-episode $|t|$",
        "id_post": "Average post-episode effect", "id_postsig": "Post-episode coefficients significant at 5\\%",
        "id_products": "Products",
        "panel_pre": "Panel A. Pre-episode quarters", "panel_post": "Panel B. Post-episode quarters",
        "variable": "Variable", "n": "N", "mean": "Mean", "sd": "SD", "min": "Min",
        "p25": "P25", "median": "Median", "p75": "P75", "max": "Max",
        "mc_est": "Estimated parameters only", "mc_full": "All parameters",
        "mc_int": "90\\% interval", "mc_h3": "Panel A. Ranking of the frictions",
        "mc_h2": "Panel B. Impact bound", "mc_inv": "Invoicing effect (\\%)",
        "mc_wedge": "Capital-controls effect (\\%)", "mc_diff": "Difference (\\%)",
        "mc_share": "Share of draws (\\%)", "mc_rank": "with the ranking of \\cref{hyp:h3}",
        "mc_mag": "with the larger effect for capital controls",
        "mc_det": "with a unique stable equilibrium",
        "mc_req": "Required appreciation (\\%)", "mc_feas": "Feasible draws (\\%)",
        "mc_n": "Draws", "mc_dist": "Distribution", "mc_rho": "Rank correlation with the",
        "mc_rho_inv": "invoicing | effect", "mc_rho_wedge": "capital-controls | effect",
        "mc_rho_diff": "difference", "mc_h2only": "impact bound only",
        "normal": "Normal", "uniform": "Uniform", "log-uniform": "Log-uniform",
        "cv_lit": "Literature", "cv_data": "Data-disciplined",
        "cv_req_obs": "Observed | frictions (\\%)",
        "cv_req_none": "No | friction (\\%)",
        "cv_obs_feas": "Observed-friction | bound feasible",
        "cv_rank": "H3 | holds",
    },
    "fr": {
        "dep": "Variable expliquée", "logv": "Log de la valeur | importée $v_{i,t}$",
        "logp": "Log du prix | frontière $\\ln P_{j,t}$", "tau": "Tarif effectif $\\tau$",
        "log1p": "Log du tarif brut $\\ln(1+\\tau)$", "p_cl": "Valeur $p$ robuste par grappe",
        "p_wcb": "Valeur $p$, bootstrap sauvage",
        "fe_unit": "Effets fixes produit ou industrie", "fe_time": "Effets fixes mois",
        "cl_unit": "Niveau de regroupement", "product": "Produit", "industry": "Industrie",
        "clusters": "Grappes", "months": "Mois", "obs": "Observations", "yes": "Oui",
        "unw": "Non pondéré", "wtd": "Pondéré",
        "panel_a_stack": "Panneau A. Coefficients en temps d'événement $\\beta^{S}_k$",
        "panel_b_stack": "Panneau B. Effet moyen post-traitement par seuil",
        "panel_c_stack": "Panneau C. Échantillons par cohorte",
        "ref": "référence", "pp": "points de pourcentage",
        "stacked_obs": "Obs. produit-trimestre-cohorte",
        "stacks": "Cohortes de traitement", "treated": "Produits traités",
        "controls": "Témoins jamais traités par échantillon de cohorte",
        "min_cell": "Valeur minimale de la cellule (dollars)", "coef_se": "Coefficient $\\beta^{V}$",
        "rb_R": "Pouvoir de | rééquilibrage $R$", "rb_req": "Appréciation | requise (\\%)",
        "no": "Non",
        "panel_a_rb": "Panneau A. Élasticité commerciale $\\eta=\\eta^{\\ast}$",
        "panel_b_rb": "Panneau B. Variations de calibration une à une",
        "data_level": "données, tarif en niveau",
        "data_log": "données, log du tarif brut", "baseline": "référence",
        "gamma": "Ouverture $\\gamma", "imb": "Déséquilibre initial $\\bar{b}",
        "emax": "Appréciation tolérée $\\bar{e}", "mu": "Échelle d'amortissement $\\mu",
        "panel_a_ge": "Panneau A. Friction de facturation en dollars $\\theta$ (à $\\chi=1$)",
        "panel_b_ge": "Panneau B. Coin de contrôle des capitaux $\\chi$ (au $\\theta$ estimé)",
        "estimated": "estimé", "rb_fr": "Panneau A. Frictions", "rb_el": "Panneau B. Élasticités commerciales",
        "rb_ow": "Panneau C. Variations une à une", "sc_obs": "Frictions estimées",
        "sc_noinv": "Sans friction de facturation, $\\theta=0$", "sc_open": "Compte ouvert, $\\chi=0$",
        "sc_none": "Sans friction, $\\theta=0$ et $\\chi=0$", "el_base": "Référence",
        "el_level": "Estimation de $\\eta$ par le tarif en niveau", "el_star": "Élasticité étrangère",
        "par_est": "Panneau A. Estimés",
        "par_comp": "Panneau B. Calculés à partir des données de commerce et de production, 2017",
        "par_cal": "Panneau C. Calibrés", "method": "Méthode ou source",
        "horizon": "Horizon (trimestres)", "cpt": "Transmission cumulée $\\zeta_h$",
        "implied": "Friction implicite $\\theta_h$",
        "ow_s": "Part des importations $s", "ow_theta": "Friction de facturation $\\theta",
        "peak": "Appréciation | maximale", "nxcum": "Exportations | nettes",
        "gapcum": "Coût en écart | de production", "eff": "Efficacité $\\Lambda$",
        "quarter": "Trimestre", "calib": "Calibration", "ge_inv": "Gradient de facturation",
        "ge_wedge": "Gradient du coin", "indet": "pas de solution stable unique",
        "param": "Paramètre", "symbol": "Symbole", "value": "Valeur", "source": "Source",
        "evtime": "Temps d'événement", "coef": "Coefficient", "stderr": "Erreur type", "pval": "Valeur $p$",
        "id_base": "Référence", "id_ref": "Référence 2017T4", "id_split": "Échantillon scindé",
        "id_refq": "Trimestre de référence", "id_q1": "Coefficient, 2018T1", "id_q2": "Coefficient, 2018T2",
        "id_presig": "Coefficients antérieurs significatifs à 5\\%", "id_maxt": "Plus grand $|t|$ antérieur",
        "id_post": "Effet moyen postérieur", "id_postsig": "Coefficients postérieurs significatifs à 5\\%",
        "id_products": "Produits",
        "panel_pre": "Panneau A. Trimestres antérieurs à l'épisode",
        "panel_post": "Panneau B. Trimestres postérieurs à l'épisode",
        "variable": "Variable", "n": "N", "mean": "Moyenne", "sd": "Écart-type", "min": "Min",
        "p25": "P25", "median": "Médiane", "p75": "P75", "max": "Max",
        "mc_est": "Paramètres estimés seuls", "mc_full": "Tous les paramètres",
        "mc_int": "Intervalle à 90~\\%", "mc_h3": "Panneau A. Classement des frictions",
        "mc_h2": "Panneau B. Borne d'impact", "mc_inv": "Effet de la facturation (\\%)",
        "mc_wedge": "Effet du contrôle des capitaux (\\%)", "mc_diff": "Différence (\\%)",
        "mc_share": "Part des tirages (\\%)", "mc_rank": "vérifiant le classement de l'\\cref{hyp:h3}",
        "mc_mag": "où le contrôle des capitaux pèse le plus",
        "mc_det": "dotés d'un équilibre stable unique",
        "mc_req": "Appréciation requise (\\%)", "mc_feas": "Tirages faisables (\\%)",
        "mc_n": "Tirages", "mc_dist": "Distribution", "mc_rho": "Corrélation de rang avec",
        "mc_rho_inv": "l'effet de la | facturation", "mc_rho_wedge": "l'effet du contrôle | des capitaux",
        "mc_rho_diff": "la différence", "mc_h2only": "borne d'impact seulement",
        "normal": "Normale", "uniform": "Uniforme", "log-uniform": "Log-uniforme",
        "cv_lit": "Littérature", "cv_data": "Données",
        "cv_req_obs": "Frictions | observées (\\%)",
        "cv_req_none": "Sans | friction (\\%)",
        "cv_obs_feas": "Borne faisable | frictions observées",
        "cv_rank": "H3 | vérifié",
    },
}

GE_SENSITIVITY_LABELS = {
    "baseline": ("Baseline", "Référence"),
    "trade elasticity 2.18 (data-implied)": ("Trade elasticity, data-implied",
                                             "Élasticité commerciale issue des données"),
    "trade elasticity 5": ("Trade elasticity $5$", "Élasticité commerciale $5$"),
    "openness 0.10": ("Openness $0.10$", "Ouverture $0{,}10$"),
    "openness 0.30": ("Openness $0.30$", "Ouverture $0{,}30$"),
    "import share 0.50": ("Import share $0.50$", "Part des importations $0{,}50$"),
    "import share 0.80": ("Import share $0.80$", "Part des importations $0{,}80$"),
    "portfolio cost 0.01": ("Portfolio cost $0.01$", "Coût de portefeuille $0{,}01$"),
    "portfolio cost 0.05": ("Portfolio cost $0.05$", "Coût de portefeuille $0{,}05$"),
    "risk aversion 1": ("Risk aversion $1$", "Aversion au risque $1$"),
    "risk aversion 5": ("Risk aversion $5$", "Aversion au risque $5$"),
    "Calvo stickiness 0.50": ("Calvo stickiness $0.50$", "Rigidité de Calvo $0{,}50$"),
    "Calvo stickiness 0.90": ("Calvo stickiness $0.90$", "Rigidité de Calvo $0{,}90$"),
}

# (field, symbol, EN description, FR description, EN source, FR source)
CALIBRATION_ROWS = [
    ("beta", "\\omega", "Discount factor (quarterly)", "Facteur d'actualisation (trimestriel)",
     "Standard", "Standard"),
    ("sigma", "\\sigma", "Inverse intertemporal elasticity", "Inverse de l'élasticité intertemporelle",
     "Standard", "Standard"),
    ("theta", "\\xi", "Calvo price stickiness", "Rigidité des prix de Calvo",
     "Four-quarter price duration", "Durée des prix de quatre trimestres"),
    ("phi_pi", "\\phi_{\\pi}", "Policy response to inflation", "Réaction de la politique à l'inflation",
     "\\citet{taylor1993}", "\\citet{taylor1993}"),
    ("phi_y", "\\phi_{y}", "Policy response to the output gap",
     "Réaction de la politique à l'écart de production", "\\citet{taylor1993}", "\\citet{taylor1993}"),
    ("eta", "\\eta", "Home import-demand elasticity", "Élasticité de la demande d'importations domestique",
     "\\citet{backus1994}", "\\citet{backus1994}"),
    ("eta_star", "\\eta^{\\ast}", "Foreign import-demand elasticity",
     "Élasticité de la demande d'importations étrangère", "\\citet{backus1994}", "\\citet{backus1994}"),
    ("gamma", "\\gamma", "Openness", "Ouverture", "Calibrated", "Calibré"),
    ("import_share", "s", "Import share of bilateral trade", "Part des importations dans le commerce bilatéral",
     "Calibrated", "Calibré"),
    ("portfolio_cost", "\\psi", "Portfolio adjustment cost", "Coût d'ajustement de portefeuille",
     "\\citet{schmittgrohe2003}", "\\citet{schmittgrohe2003}"),
    ("rho_tau", "\\rho_{\\tau}", "Persistence of the tariff shock", "Persistance du choc tarifaire",
     "Standard", "Standard"),
    ("rho_z", "\\rho_{z}", "Persistence of the depreciation shock", "Persistance du choc de dépréciation",
     "Standard", "Standard"),
    ("theta_dollar", "\\theta", "Dollar-invoicing friction", "Friction de facturation en dollars",
     "\\citet{gopinath2020,boz2022}", "\\citet{gopinath2020,boz2022}"),
    ("chi", "\\chi", "Capital-controls wedge", "Coin de contrôle des capitaux",
     "Calibrated", "Calibré"),
    ("imbalance0", "\\bar{b}", "Initial bilateral imbalance (share of output)",
     "Déséquilibre bilatéral initial (part de la production)",
     "Calibrated", "Calibré"),
    ("max_depreciation", "\\bar{e}", "Tolerated real appreciation", "Appréciation réelle tolérée",
     "Calibrated", "Calibré"),
]


# ---------------------------------------------------------------------------------------
# Collect every result once
# ---------------------------------------------------------------------------------------

def collect() -> dict:
    """Run the analyses once and return every object the tables and macros need."""
    from src.analysis.sector_passthrough import (build_panel, run as sector_run,
                                                 run_tariff_transform, run_value_thresholds)
    from src.analysis.price_passthrough import run as price_run, run_wcb
    from src.analysis.sector_event_study import build_quarter_panel
    from src.analysis import sector_event_study_staggered as stg
    from src.analysis.rebalancing_threshold import (calibration_sensitivity, conclusion1,
                                                    damping_sensitivity, elasticity_sensitivity)
    from src.analysis import parameter_estimation as pe
    from src.analysis.baseline import baseline
    from src.analysis.dsge_counterfactual import (chi_grid, intervention_persistence,
                                                  invoicing_interval, sensitivity,
                                                  theta_dollar_grid)
    from src.analysis.calibration_validation import compare as calibration_compare
    from src.common.twfe import twfe_ols
    from src.data.sector_prices import load_price_tariff_panel
    from src.engine.model import r_min, rebalancing_power, required_depreciation, trade_term

    res: dict = {}
    res["sector"] = sector_run().iloc[0].to_dict()
    tr = run_tariff_transform()
    res["sector_log1p"] = tr[tr["tariff_regressor"].eq("log(1+tau)")].iloc[0].to_dict()
    res["price"] = price_run()
    res["price_wcb"] = run_wcb()
    res["cells"] = run_value_thresholds()

    panel, terms = build_quarter_panel()
    ev = twfe_ols(panel, "log_value", terms, "hs4", "q", cluster="hs4")
    ev.index = [t.replace("e_", "") for t in ev.index]
    res["event"] = ev
    from src.analysis.sector_event_study import identification_tests
    res["ident"] = identification_tests()
    res["event_n"] = len(panel)
    res["event_products"] = int(panel["hs4"].nunique())

    st = stg.build_stacked()
    res["stack_counts"] = stg.stack_counts(st)
    res["stack_n"] = len(st)
    res["stacked"] = {w: stg.run(weighted=w) for w in (False, True)}
    res["stacked_thresholds"] = {w: stg.threshold_robustness(weighted=w) for w in (False, True)}

    res["h2_frictions"] = conclusion1()
    res["elasticity"] = elasticity_sensitivity()
    res["calibration_sens"] = calibration_sensitivity()
    res["damping"] = damping_sensitivity()
    res["eta_hat"] = pe.estimate_eta()
    res["passthrough"] = pe.estimate_passthrough()
    res["rho_tau"] = pe.estimate_rho_tau()
    res["shares"] = pe.trade_shares()
    res["share_range"] = pe.trade_share_range()
    b = baseline()
    res["params"] = b
    res["baseline_R"] = rebalancing_power(b)
    res["baseline_req"] = required_depreciation(b)
    res["r_min"] = r_min(b)
    res["trade_term"] = trade_term(b)

    res["ge_theta"] = theta_dollar_grid()
    res["ge_chi"] = chi_grid()
    res["ge_sens"] = sensitivity()
    res["calibration_validation"] = calibration_compare()
    res["ge_interval"] = invoicing_interval()
    res["ge_persistence"] = intervention_persistence()
    from src.analysis import parameter_uncertainty as pu
    res["mc"] = {layer: pu.summary(layer) for layer in pu.LAYERS}
    res["mc_ranges"] = pu.parameter_ranges()
    res["mc_drivers"] = pu.drivers()
    res["mc_failures"] = pu.ranking_failures()
    from src.figures.fig_ge_irf import responses
    res["irf"] = responses()
    from src.analysis.dsge_counterfactual import HORIZON
    from src.engine.dsge import irf
    res["irf_base"] = irf("z", periods=HORIZON, p=b)

    d = build_panel()
    d = d[np.isfinite(d["log_value"])]
    exposure = pd.read_parquet(DATA_DIR / "china_input_exposure.parquet")
    res["desc"] = [
        ("tau", d["effective_tariff"]), ("v", d["log_value"]),
        ("P", load_price_tariff_panel()["price_index"]),
        ("m", exposure["china_input_share"]),
        ("dtau", panel.drop_duplicates("hs4")["dtau"]),
    ]
    agg = pe.quarterly_inputs()
    res["desc"] += [("tau_agg", agg["tau"]), ("pm", agg["p"]), ("e", agg["e"])]
    # Correlation matrices of the analysis variables, in unit-consistent blocks.
    pm = d.merge(panel[["hs4", "dtau"]].drop_duplicates("hs4"), on="hs4", how="inner")
    res["corr_product"] = pm[["effective_tariff", "log_value", "dtau"]].corr()
    res["corr_aggregate"] = agg[["tau", "p", "e"]].corr()
    res["firm"] = _firm_results()
    return res


def _firm_results() -> dict | None:
    """Firm-level supplement: recomputed from licensed data if present, else the stored manifest."""
    try:
        from src.analysis.firm_incidence import run_annual
        gm = run_annual()
        row = gm[gm["outcome"] == "gross margin"].iloc[0]
        return {"beta": row["beta_firm_time"], "p": row["p_firm_time"],
                "beta_trend": row["beta_firm_sectorXtime"], "p_trend": row["p_firm_sectorXtime"],
                "n": int(row["n"])}
    except FileNotFoundError:
        path = PROJECT_ROOT / "protocol" / "results_manifest_wrds.json"
        if not path.exists():
            return None
        stored = json.loads(path.read_text())["results"]
        return {"beta": stored["firm_gross_margin_beta_firm_time"], "p": float("nan"),
                "beta_trend": stored["firm_gross_margin_beta_sectorXtime"], "p_trend": float("nan"),
                "n": 0}


# ---------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------

def _two_lines(text: str) -> str:
    """Header cell broken over two centered lines at the first ' | '."""
    top, _, bottom = text.partition(" | ")
    return f"\\begin{{tabular}}[b]{{@{{}}c@{{}}}}{top}\\\\{bottom}\\end{{tabular}}"


def _rule_row(label: str, cells: list[str]) -> str:
    return label + " & " + " & ".join(cells) + " \\\\\n"


def table_incidence(res: dict, lang: str) -> str:
    T = TXT[lang]
    s, s2, p, w = res["sector"], res["sector_log1p"], res["price"], res["price_wcb"]
    out = "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
    out += (f"{T['dep']} & \\multicolumn{{2}}{{c}}{{{_two_lines(T['logv'])}}}"
            f" & {_two_lines(T['logp'])} \\\\\n")
    out += "\\cmidrule(lr){2-3}\\cmidrule(l){4-4}\n & (1) & (2) & (3) \\\\\n\\midrule\n"
    out += _rule_row(T["tau"], [coef(s["beta"], s["p"], 3, lang), "", coef(p["beta"], p["p"], 3, lang)])
    out += _rule_row("", [se(s["se"], 3, lang), "", se(p["se"], 3, lang)])
    out += _rule_row(T["log1p"], ["", coef(s2["beta"], s2["p"], 3, lang), ""])
    out += _rule_row("", ["", se(s2["se"], 3, lang), ""])
    out += "\\addlinespace\n"
    out += _rule_row(T["p_cl"], [num(s["p"], 3, lang), num(s2["p"], 3, lang), num(p["p"], 3, lang)])
    out += _rule_row(T["p_wcb"], ["", "", num(w["p_wcb"], 3, lang)])
    out += "\\midrule\n"
    out += _rule_row(T["fe_unit"], [T["yes"]] * 3)
    out += _rule_row(T["fe_time"], [T["yes"]] * 3)
    out += _rule_row(T["cl_unit"], [T["product"], T["product"], T["industry"]])
    out += _rule_row(T["clusters"], [integer(s["n_cluster"], lang), integer(s2["n_cluster"], lang),
                                     integer(p["n_cluster"], lang)])
    out += _rule_row(T["months"], [integer(s["n_fe2"], lang), integer(s["n_fe2"], lang), integer(p["n_fe2"], lang)])
    out += _rule_row(T["obs"], [integer(s["n"], lang), integer(s2["n"], lang), integer(p["n"], lang)])
    return out + "\\bottomrule\n\\end{tabular}\n"


def table_staggered(res: dict, lang: str) -> str:
    T = TXT[lang]
    unw, wtd = res["stacked"][False], res["stacked"][True]
    out = "\\begin{tabular}{@{}lcc@{}}\n\\toprule\n"
    out += f" & {T['unw']} & {T['wtd']} \\\\\n & (1) & (2) \\\\\n\\midrule\n"
    out += _panel_head(3, T['panel_a_stack'])
    for k in range(-4, 7):
        label = f"$k={raw_num(k, 0, lang)}$"
        if k == -1:
            out += _rule_row(label, [f"$0$ ({T['ref']})"] * 2)
            continue
        out += _rule_row(label, [coef(r.loc[k, "beta"], r.loc[k, "p"], 3, lang) for r in (unw, wtd)])
        out += _rule_row("", [se(r.loc[k, "se"], 3, lang) for r in (unw, wtd)])
    out += "\\addlinespace\n"
    out += _panel_head(3, T['panel_b_stack'])
    tu, tw = res["stacked_thresholds"][False], res["stacked_thresholds"][True]
    for i in range(len(tu)):
        label = f"{integer(tu.loc[i, 'threshold_pp'], lang)} {T['pp']}"
        out += _rule_row(label, [num(tu.loc[i, "post_mean_beta"], 3, lang),
                                 num(tw.loc[i, "post_mean_beta"], 3, lang)])
    out += "\\addlinespace\n"
    out += _panel_head(3, T['panel_c_stack'])
    counts = res["stack_counts"]
    ctrl = f"{integer(counts['n_control'].min(), lang)}--{integer(counts['n_control'].max(), lang)}"
    out += f"{T['stacked_obs']} & \\multicolumn{{2}}{{c}}{{{integer(res['stack_n'], lang)}}} \\\\\n"
    out += f"{T['stacks']} & \\multicolumn{{2}}{{c}}{{{integer(len(counts), lang)}}} \\\\\n"
    out += f"{T['treated']} & \\multicolumn{{2}}{{c}}{{{integer(counts['n_treated'].sum(), lang)}}} \\\\\n"
    out += f"{T['controls']} & \\multicolumn{{2}}{{c}}{{{ctrl}}} \\\\\n"
    return out + "\\bottomrule\n\\end{tabular}\n"


def table_cells(res: dict, lang: str) -> str:
    T = TXT[lang]
    c = res["cells"]
    out = "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
    out += T["min_cell"] + " & " + " & ".join(integer(v, lang) for v in c["min_value_usd"]) + " \\\\\n\\midrule\n"
    out += _rule_row(T["coef_se"], [coef(b, p, 3, lang) for b, p in zip(c["beta"], c["p"])])
    out += _rule_row("", [se(v, 3, lang) for v in c["se"]])
    out += _rule_row(T["obs"], [integer(v, lang) for v in c["n"]])
    return out + "\\bottomrule\n\\end{tabular}\n"


def _feasible(flag: bool, lang: str) -> str:
    return TXT[lang]["yes"] if flag else TXT[lang]["no"]


def table_rebalancing(res: dict, lang: str) -> str:
    T = TXT[lang]
    out = "\\begin{tabular}{@{}lcc@{}}\n\\toprule\n"
    out += f" & {_two_lines(T['rb_R'])} & {_two_lines(T['rb_req'])} \\\\\n\\midrule\n"

    def line(label: str, r) -> str:
        return _rule_row(label, [num(r["R"], 4, lang), num(100 * r["required_deprec"], 0, lang)])

    out += _panel_head(3, T['rb_fr'])
    for key, (_, r) in zip(("sc_obs", "sc_noinv", "sc_open", "sc_none"), res["h2_frictions"].iterrows()):
        out += line(T[key], r)
    out += "\\addlinespace\n"
    out += _panel_head(3, T['rb_el'])
    el = res["elasticity"]
    labels = [T["el_base"], f"{T['el_level']} $({raw_num(res['eta_hat']['eta_level'], 2, lang)})$"]
    labels += [f"{T['el_star']} $\\eta^{{\\ast}}={raw_num(v, 2, lang)}$" for v in el["eta_star"].iloc[2:]]
    for label, (_, r) in zip(labels, el.iterrows()):
        out += line(label, r)
    out += "\\addlinespace\n"
    out += _panel_head(3, T['rb_ow'])
    names = {"gamma": (T["gamma"], 4), "imbalance0": (T["imb"], 4), "import_share": (T["ow_s"], 3),
             "theta_dollar": (T["ow_theta"], 3), "max_depreciation": (T["emax"], 2)}
    for _, r in res["calibration_sens"].iterrows():
        label, nd = names[r["parameter"]]
        out += line(f"{label} = {raw_num(r['value'], nd, lang)}$", r)
    for _, r in res["damping"].iterrows():
        if r["damping_scale_k"] == 1.0:
            continue
        out += line(f"{T['mu']} = {raw_num(r['damping_scale_k'], 0, lang)}$", r)
    return out + "\\bottomrule\n\\end{tabular}\n"


def table_ge(res: dict, lang: str) -> str:
    T = TXT[lang]
    out = "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
    out += (f" & {_two_lines(T['peak'])} & {_two_lines(T['nxcum'])} & {_two_lines(T['gapcum'])}"
            f" & {T['eff']} \\\\\n\\midrule\n")
    for panel, df, col, sym, nd in (("panel_a_ge", res["ge_theta"], "theta_dollar", "\\theta", 2),
                                    ("panel_b_ge", res["ge_chi"], "chi", "\\chi", 1)):
        out += _panel_head(5, T[panel])
        for r in df.itertuples(index=False):
            d = r._asdict()
            tag = f" ({T['estimated']})" if d.get("estimated") else ""
            digits = 3 if d.get("estimated") else nd
            out += _rule_row(f"${sym} = {raw_num(d[col], digits, lang)}${tag}",
                             [num(d["peak_rmb_appreciation"], 3, lang), num(d["NX_cum"], 3, lang),
                              num(d["GAP_cum"], 3, lang), num(d["efficiency"], 3, lang)])
        if panel == "panel_a_ge":
            out += "\\addlinespace\n"
    return out + "\\bottomrule\n\\end{tabular}\n"


SENS_LABELS = {
    "baseline": ("Baseline", "Référence"),
    "eta_level": ("Level-tariff estimate of $\\eta$", "Estimation de $\\eta$ par le tarif en niveau"),
    "eta_star": ("Foreign elasticity $\\eta^{\\ast}$", "Élasticité étrangère $\\eta^{\\ast}$"),
    "gamma": ("Openness $\\gamma$", "Ouverture $\\gamma$"),
    "import_share": ("Import share $s$", "Part des importations $s$"),
    "portfolio_cost": ("Portfolio cost $\\psi$", "Coût de portefeuille $\\psi$"),
    "sigma": ("Risk aversion $\\sigma$", "Aversion au risque $\\sigma$"),
    "calvo": ("Calvo stickiness $\\xi$", "Rigidité de Calvo $\\xi$"),
    "phi_pi": ("Taylor response $\\phi_{\\pi}$", "Réaction de Taylor $\\phi_{\\pi}$"),
    "phi_y": ("Taylor response $\\phi_{y}$", "Réaction de Taylor $\\phi_{y}$"),
}


def table_ge_sensitivity(res: dict, lang: str) -> str:
    T = TXT[lang]
    i = 0 if lang == "en" else 1
    out = "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n"
    out += (f" & \\multicolumn{{3}}{{c}}{{{T['ge_inv']}}} & \\multicolumn{{3}}{{c}}{{{T['ge_wedge']}}} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n")
    out += (f"{T['calib']} & $\\Lambda_{{\\theta=0}}$ & $\\Lambda_{{\\hat{{\\theta}}}}$ & ratio"
            f" & $\\Lambda_{{\\chi=0}}$ & $\\Lambda_{{\\chi=4}}$ & ratio \\\\\n\\midrule\n")
    for r in res["ge_sens"].itertuples(index=False):
        d = r._asdict()
        label = SENS_LABELS[d["key"]][i]
        if d["key"] not in ("baseline",):
            nd = 4 if d["key"] == "gamma" else (3 if d["key"] == "import_share" else 2)
            label = f"{label} $= {raw_num(d['value'], nd, lang)}$"
        if not d["determinate"]:
            out += f"{label} & \\multicolumn{{6}}{{c}}{{\\emph{{{T['indet']}}}}} \\\\\n"
            continue
        out += _rule_row(label, [num(d["eff_theta_0"], 3, lang), num(d["eff_theta_hat"], 3, lang),
                                 num(d["invoicing_ratio"], 2, lang), num(d["eff_chi_0"], 3, lang),
                                 num(d["eff_chi_4"], 3, lang), num(d["wedge_ratio"], 2, lang)])
    return out + "\\bottomrule\n\\end{tabular}\n"


def table_calibration_validation(res: dict, lang: str) -> str:
    """Literature-versus-data validation of H2 and H3 structural diagnostics."""
    T = TXT[lang]
    labels = {"literature": T["cv_lit"], "data_disciplined": T["cv_data"]}
    out = "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
    out += (f"{T['calib']} & {_two_lines(T['cv_req_obs'])} & {_two_lines(T['cv_req_none'])}"
            f" & $\\Lambda_{{\\theta}}/\\Lambda_{{0}}$"
            f" & $\\Lambda_{{\\chi=4}}/\\Lambda_{{\\chi=0}}$ & {_two_lines(T['cv_rank'])} \\\\\n"
            "\\midrule\n")
    for _, r in res["calibration_validation"].iterrows():
        out += _rule_row(labels[r["calibration"]], [
            num(100 * r["required_observed"], 0, lang),
            num(100 * r["required_no_friction"], 0, lang),
            num(r["invoicing_ratio"], 2, lang),
            num(r["wedge_ratio"], 2, lang),
            _feasible(bool(r["ranking_holds"]), lang),
        ])
    return out + "\\bottomrule\n\\end{tabular}\n"


def table_event(res: dict, lang: str) -> str:
    T = TXT[lang]
    ev = res["event"]
    quarters = sorted(list(ev.index) + ["2018Q2"], key=lambda q: (int(q[:4]), int(q[-1])))
    ref = quarters.index("2018Q2")
    out = "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
    out += (f"{T['quarter']} & {T['evtime']} $k$ & {T['coef']} $\\beta_k$ & {T['stderr']}"
            f" & {T['pval']} \\\\\n\\midrule\n")
    for panel, rows in (("panel_pre", quarters[:ref + 1]), ("panel_post", quarters[ref + 1:])):
        out += _panel_head(5, T[panel])
        for q in rows:
            k = quarters.index(q) - ref - 1
            label = q.replace("Q", "T") if lang == "fr" else q
            if q == "2018Q2":
                out += _rule_row(label, [f"${raw_num(k, 0, lang)}$", f"$0$ ({T['ref']})", "", ""])
                continue
            r = ev.loc[q]
            out += _rule_row(label, [f"${raw_num(k, 0, lang)}$", coef(r["beta"], r["p"], 3, lang),
                                     num(r["se"], 3, lang), num(r["p"], 3, lang)])
        if panel == "panel_pre":
            out += "\\addlinespace\n"
    return out + "\\bottomrule\n\\end{tabular}\n"


def table_identification(res: dict, lang: str) -> str:
    T = TXT[lang]
    cols = [res["ident"][k] for k in ("baseline", "reference_2017Q4", "split_sample")]

    def qlabel(q: str) -> str:
        return q.replace("Q", "T") if lang == "fr" else q

    def coef_or_ref(r: dict, q: str) -> tuple[str, str]:
        if q == r["ref_q"]:
            return f"$0$ ({T['ref']})", ""
        c = r["coef"].loc[q]
        return coef(c["beta"], c["p"], 3, lang), se(c["se"], 3, lang)

    out = "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n"
    out += f" & {T['id_base']} & {T['id_ref']} & {T['id_split']} \\\\\n & (1) & (2) & (3) \\\\\n\\midrule\n"
    out += _rule_row(T["id_refq"], [qlabel(r["ref_q"]) for r in cols])
    for q, key in (("2018Q1", "id_q1"), ("2018Q2", "id_q2")):
        pairs = [coef_or_ref(r, q) for r in cols]
        out += _rule_row(T[key], [a for a, _ in pairs])
        out += _rule_row("", [b for _, b in pairs])
    out += "\\addlinespace\n"
    out += _rule_row(T["id_presig"], [f"{integer(r['pre_sig'], lang)}/{integer(r['pre_n'], lang)}" for r in cols])
    out += _rule_row(T["id_maxt"], [num(r["pre_max_abs_t"], 2, lang) for r in cols])
    out += "\\addlinespace\n"
    out += _rule_row(T["id_post"], [coef(r["post_mean"], r["post_mean_p"], 3, lang) for r in cols])
    out += _rule_row("", [se(r["post_mean_se"], 3, lang) for r in cols])
    out += _rule_row(T["id_postsig"], [f"{integer(r['post_sig'], lang)}/{integer(r['post_n'], lang)}" for r in cols])
    out += "\\midrule\n"
    out += _rule_row(T["obs"], [integer(r["n"], lang) for r in cols])
    out += _rule_row(T["id_products"], [integer(r["products"], lang) for r in cols])
    return out + "\\bottomrule\n\\end{tabular}\n"


def table_calibration(res: dict, lang: str) -> str:
    """Parameter table in three tiers: estimated, computed from data, calibrated.

    Each row shows the literature value and the retained baseline side by side, so the six
    parameters that the data replace (Panels A and B) are visible against those left calibrated
    (Panel C, where the two columns coincide).
    """
    from src.engine.calibration import LITERATURE as L
    T = TXT[lang]
    i = 0 if lang == "en" else 1
    lit_h, base_h = ("Literature", "Littérature")[i], ("Baseline", "Référence")[i]
    p, eta, pt, rho, sh = res["params"], res["eta_hat"], res["passthrough"], res["rho_tau"], res["shares"]
    out = ("\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{4.4cm}ccc"
           ">{\\raggedright\\arraybackslash}p{4.4cm}@{}}\n\\toprule\n")
    out += f"{T['param']} & {T['symbol']} & {lit_h} & {base_h} & {T['method']} \\\\\n\\midrule\n"
    est = [
        ("Import-demand elasticity", "Élasticité de la demande d'importations", "\\eta", eta["eta"], eta["se"], L.eta,
         "Product panel, log gross tariff, \\cref{lem:mapping}", "Panel de produits, log du tarif brut, \\cref{lem:mapping}"),
        ("Dollar-invoicing friction", "Friction de facturation en dollars", "\\theta", pt["theta"], pt["se"], L.theta_dollar,
         "Exchange-rate pass-through, \\cref{eq:erpt}", "Transmission du change, \\cref{eq:erpt}"),
        ("Persistence of the tariff", "Persistance du tarif", "\\rho_{\\tau}", rho["rho_tau"], rho["se"], L.rho_tau,
         "First-order autoregression of the aggregate effective tariff", "Autorégression d'ordre un du tarif effectif agrégé"),
    ]
    out += _panel_head(5, T['par_est'])
    for en, fr, sym, v, sd, lit, m_en, m_fr in est:
        out += _rule_row((en, fr)[i], [f"${sym}$", num(lit, 2, lang), f"{num(v, 3, lang)} {se(sd, 3, lang)}", (m_en, m_fr)[i]])
    comp = [
        ("Import share of bilateral trade", "Part des importations dans le commerce bilatéral", "s",
         sh["import_share"], 3, L.import_share, "$M/(X+M)$, Census Bureau", "$M/(X+M)$, Bureau du recensement"),
        ("Bilateral openness", "Ouverture bilatérale", "\\gamma", sh["gamma"], 4, L.gamma,
         "$(X+M)/Y$, Census Bureau and Bureau of Economic Analysis", "$(X+M)/Y$, Bureau du recensement et Bureau of Economic Analysis"),
        ("Initial imbalance, share of output", "Déséquilibre initial, part de la production", "\\bar{b}",
         sh["imbalance0"], 4, L.imbalance0, "$(M-X)/Y$, Census Bureau and Bureau of Economic Analysis", "$(M-X)/Y$, Bureau du recensement et Bureau of Economic Analysis"),
    ]
    out += "\\addlinespace\n"
    out += _panel_head(5, T['par_comp'])
    for en, fr, sym, v, nd, lit, m_en, m_fr in comp:
        out += _rule_row((en, fr)[i], [f"${sym}$", num(lit, 2, lang), num(v, nd, lang), (m_en, m_fr)[i]])
    ge_s, rb = "\\cref{tab:ge-sensitivity}", "\\cref{tab:rebalancing}"
    cal = [
        ("Foreign import-demand elasticity", "Élasticité de la demande d'importations étrangère", "\\eta^{\\ast}",
         p.eta_star, "\\citet{backus1994}; varied, \\cref{tab:rebalancing,tab:ge-sensitivity}",
         "\\citet{backus1994}~; varié, \\cref{tab:rebalancing,tab:ge-sensitivity}"),
        ("Capital-controls wedge", "Coin de contrôle des capitaux", "\\chi", p.chi,
         "Moderately closed account; varied, \\cref{tab:ge}",
         "Compte modérément fermé~; varié, \\cref{tab:ge}"),
        ("Damping scale", "Échelle d'amortissement", "\\mu", 1.0, f"Varied, {rb}", f"Varié, {rb}"),
        ("Tolerated appreciation", "Appréciation tolérée", "\\bar{e}", p.max_depreciation,
         f"Varied, {rb}", f"Varié, {rb}"),
        ("Portfolio adjustment cost", "Coût d'ajustement de portefeuille", "\\psi", p.portfolio_cost,
         f"\\citet{{schmittgrohe2003}}; varied, {ge_s}", f"\\citet{{schmittgrohe2003}}~; varié, {ge_s}"),
        ("Inverse intertemporal elasticity", "Inverse de l'élasticité intertemporelle", "\\sigma", p.sigma,
         f"Conventional; varied, {ge_s}", f"Valeur usuelle~; varié, {ge_s}"),
        ("Calvo price stickiness", "Rigidité des prix de Calvo", "\\xi", p.theta,
         f"Four-quarter price duration; varied, {ge_s}", f"Durée des prix de quatre trimestres~; varié, {ge_s}"),
        ("Policy response to inflation", "Réaction de la politique à l'inflation", "\\phi_{\\pi}", p.phi_pi,
         f"\\citet{{taylor1993}}; varied, {ge_s}", f"\\citet{{taylor1993}}~; varié, {ge_s}"),
        ("Policy response to the output gap", "Réaction de la politique à l'écart de production", "\\phi_{y}",
         p.phi_y, f"\\citet{{taylor1993}}; varied, {ge_s}", f"\\citet{{taylor1993}}~; varié, {ge_s}"),
        ("Discount factor (quarterly)", "Facteur d'actualisation (trimestriel)", "\\omega", p.beta,
         "Conventional", "Valeur usuelle"),
        ("Persistence of the depreciation shock", "Persistance du choc de dépréciation", "\\rho_{z}", p.rho_z,
         "Policy experiment; varied, \\cref{sec:rob}", "Expérience de politique~; varié, \\cref{sec:rob}"),
    ]
    out += "\\addlinespace\n"
    out += _panel_head(5, T['par_cal'])
    for en, fr, sym, v, m_en, m_fr in cal:
        out += _rule_row((en, fr)[i], [f"${sym}$", num(v, 2, lang), num(v, 2, lang), (m_en, m_fr)[i]])
    return out + "\\bottomrule\n\\end{tabular}\n"


def table_passthrough(res: dict, lang: str) -> str:
    """Cumulative exchange-rate pass-through into the dollar import price, by horizon."""
    T = TXT[lang]
    path = res["passthrough"]["path"]
    out = "\\begin{tabular}{@{}l" + "c" * len(path) + "@{}}\n\\toprule\n"
    out += T["horizon"] + " & " + " & ".join(f"${int(h)}$" for h in path["horizon"]) + " \\\\\n\\midrule\n"
    out += _rule_row(T["cpt"], [num(v, 3, lang) for v in path["cpt"]])
    out += _rule_row("", [se(v, 3, lang) for v in path["se"]])
    out += "\\addlinespace\n"
    out += _rule_row(T["implied"], ["" if h == 0 else num(t, 3, lang)
                                    for h, t in zip(path["horizon"], path["theta"])])
    return out + "\\bottomrule\n\\end{tabular}\n"


def _panel_head(span: int, label: str) -> str:
    """Bold-italic panel header (e.g. Panel A) spanning `span` columns."""
    return ("\\multicolumn{" + str(span) + "}{@{}l}{\\textbf{\\emph{"
            + label + "}}} \\\\\n")


def _pct(x: float, nd: int, lang: str) -> str:
    return num(100 * x, nd, lang)


def _interval(q: dict, lang: str) -> str:
    sep = ",\\," if lang == "en" else "\\,;\\,"
    return f"$[{raw_num(100 * q['q05'], 0, lang)}{sep}{raw_num(100 * q['q95'], 0, lang)}]$"


def table_uncertainty(res: dict, lang: str) -> str:
    """Monte Carlo summary: effects of the two frictions and the impact bound, by layer."""
    T = TXT[lang]
    est, full = res["mc"]["estimation"], res["mc"]["full"]
    width = "7.2cm" if lang == "en" else "6.2cm"
    out = f"\\begin{{tabular}}{{@{{}}>{{\\raggedright\\arraybackslash}}p{{{width}}}cccc@{{}}}}\n\\toprule\n"
    out += (f" & \\multicolumn{{2}}{{c}}{{{T['mc_est']}}} & \\multicolumn{{2}}{{c}}{{{T['mc_full']}}} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n")
    out += f" & {T['median']} & {T['mc_int']} & {T['median']} & {T['mc_int']} \\\\\n\\midrule\n"
    out += _panel_head(5, T['mc_h3'])
    for key, label in (("invoicing", "mc_inv"), ("wedge", "mc_wedge"), ("difference", "mc_diff")):
        out += _rule_row(T[label], [_pct(est[key]["q50"], 0, lang), _interval(est[key], lang),
                                    _pct(full[key]["q50"], 0, lang), _interval(full[key], lang)])
    out += f"{T['mc_share']} & & & & \\\\\n"
    for key, label in (("share_ranking", "mc_rank"), ("share_larger_magnitude", "mc_mag")):
        out += _rule_row("\\quad " + T[label], [f"\\multicolumn{{2}}{{c}}{{{_pct(est[key], 1, lang)}}}",
                                    f"\\multicolumn{{2}}{{c}}{{{_pct(full[key], 1, lang)}}}"])
    out += _rule_row("\\quad " + T["mc_det"], [f"\\multicolumn{{2}}{{c}}{{{_pct(est['n_determinate'] / est['n'], 1, lang)}}}",
                                   f"\\multicolumn{{2}}{{c}}{{{_pct(full['n_determinate'] / full['n'], 1, lang)}}}"])
    out += "\\addlinespace\n"
    out += _panel_head(5, T['mc_h2'])
    out += _rule_row(T["mc_req"], [_pct(est["required_deprec"]["q50"], 0, lang), _interval(est["required_deprec"], lang),
                                   _pct(full["required_deprec"]["q50"], 0, lang), _interval(full["required_deprec"], lang)])
    out += _rule_row(T["mc_feas"], [f"\\multicolumn{{2}}{{c}}{{{_pct(est['share_feasible'], 1, lang)}}}",
                                    f"\\multicolumn{{2}}{{c}}{{{_pct(full['share_feasible'], 1, lang)}}}"])
    out += "\\addlinespace\n"
    out += _rule_row(T["mc_n"], [f"\\multicolumn{{2}}{{c}}{{{integer(est['n'], lang)}}}",
                                 f"\\multicolumn{{2}}{{c}}{{{integer(full['n'], lang)}}}"])
    return out + "\\bottomrule\n\\end{tabular}\n"


def _distribution(row, lang: str) -> str:
    T = TXT[lang]
    sep = ",\\," if lang == "en" else "\\,;\\,"
    nd = 4 if row["parameter"] in ("gamma", "imbalance0") else (3 if row["parameter"] in ("theta_dollar", "import_share", "portfolio_cost") else 2)
    a, b = raw_num(row["lo"], nd, lang), raw_num(row["hi"], nd, lang)
    if row["distribution"] == "normal":
        return f"{T['normal']} $({a}{sep}{b})$"
    return f"{T[row['distribution']]} $[{a}{sep}{b}]$"


def table_uncertainty_ranges(res: dict, lang: str) -> str:
    """Distribution of every drawn parameter and its rank correlation with the effects."""
    T = TXT[lang]
    drivers = res["mc_drivers"].set_index("parameter")
    out = "\\begin{tabular}{@{}clcccc@{}}\n\\toprule\n"
    out += (f" & & & \\multicolumn{{3}}{{c}}{{{T['mc_rho']}}} \\\\\n\\cmidrule(l){{4-6}}\n"
            f"{T['symbol']} & {T['mc_dist']} & {T['el_base']} & {_two_lines(T['mc_rho_inv'])}"
            f" & {_two_lines(T['mc_rho_wedge'])} & {T['mc_rho_diff']} \\\\\n\\midrule\n")
    for _, row in res["mc_ranges"].iterrows():
        k = row["parameter"]
        nd = 4 if k in ("gamma", "imbalance0") else (3 if k in ("theta_dollar", "import_share", "portfolio_cost") else 2)
        cells = [_distribution(row, lang), num(row["baseline"], nd, lang)]
        if k in drivers.index:
            d = drivers.loc[k]
            cells += [num(d["rho_invoicing_effect"], 2, lang), num(d["rho_wedge_effect"], 2, lang),
                      num(d["rho_difference"], 2, lang)]
        else:
            cells += [f"\\multicolumn{{3}}{{c}}{{\\emph{{{T['mc_h2only']}}}}}"]
        out += _rule_row(f"${PARAMETER_SYMBOLS[k]}$", cells)
    return out + "\\bottomrule\n\\end{tabular}\n"


DESC_ROWS = {
    "tau": ("Effective tariff $\\tau$", "Tarif effectif $\\tau$", 3),
    "v": ("Log import value $v$", "Log de la valeur importée $v$", 3),
    "P": ("Border price $P$", "Prix frontière $P$", 3),
    "m": ("China-input exposure $m$", "Exposition aux intrants chinois $m$", 3),
    "dtau": ("Tariff-shock intensity $\\Delta\\tau$", "Intensité du choc tarifaire $\\Delta\\tau$", 3),
    "tau_agg": ("Aggregate effective tariff $\\tau_t$", "Tarif effectif agrégé $\\tau_t$", 3),
    "pm": ("Log aggregate import price $p^{m}_t$", "Log du prix agrégé à l'importation $p^{m}_t$", 3),
    "e": ("Log renminbi value $e_t$", "Log de la valeur du renminbi $e_t$", 3),
}


def table_descriptive(res: dict, lang: str) -> str:
    T = TXT[lang]
    i = 0 if lang == "en" else 1
    out = "\\setlength{\\tabcolsep}{4pt}\n"
    out += "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{3.4cm}rrrrrrrr@{}}\n\\toprule\n"
    out += " & ".join([T[k] for k in ("variable", "n", "mean", "sd", "min", "p25", "median", "p75", "max")])
    out += " \\\\\n\\midrule\n"
    for key, series in res["desc"]:
        s = pd.Series(series).dropna().astype(float)
        name, nd = DESC_ROWS[key][i], DESC_ROWS[key][2]
        q = s.quantile([0.25, 0.5, 0.75])
        vals = [s.mean(), s.std(), s.min(), q[0.25], q[0.5], q[0.75], s.max()]
        out += _rule_row(name, [integer(len(s), lang)] + [num(v, nd, lang) for v in vals])
    return out + "\\bottomrule\n\\end{tabular}\n"


# ---------------------------------------------------------------------------------------
# In-text numbers
# ---------------------------------------------------------------------------------------

def numbers(res: dict, lang: str) -> str:
    """Number macros used in the prose; each is computed here, never typed by hand."""
    s, s2, p, w = res["sector"], res["sector_log1p"], res["price"], res["price_wcb"]
    ev = res["event"]
    pre = ev.loc[[q for q in ev.index if q < "2018Q2"]]
    post = ev.loc[[q for q in ev.index if q > "2018Q2"]]
    trough_q = post["beta"].idxmin()
    unw, wtd = res["stacked"][False], res["stacked"][True]
    tu, tw = res["stacked_thresholds"][False], res["stacked_thresholds"][True]
    cells = res["cells"]
    th, ch, gs = res["ge_theta"], res["ge_chi"], res["ge_sens"]
    det = gs[gs["determinate"]]
    counts = res["stack_counts"]
    params = res["params"]

    def post_mean(r: pd.DataFrame) -> float:
        return float(r.loc[[k for k in r.index if k >= 0], "beta"].mean())

    def quarter_label(qi: int) -> str:
        label = f"{int(qi) // 4}Q{int(qi) % 4 + 1}"
        return label.replace("Q", "T") if lang == "fr" else label

    m = ""
    m += macro("ValueBeta", num(s["beta"], 2, lang)) + macro("ValueSE", num(s["se"], 2, lang))
    m += macro("ValueBetaLog", num(s2["beta"], 2, lang))
    m += macro("TenPointEffect", num(100 * (1 - math.exp(0.10 * s["beta"])), 1, lang))
    m += macro("NobsProduct", integer(s["n"], lang)) + macro("Nproducts", integer(s["n_fe1"], lang))
    m += macro("Nmonths", integer(s["n_fe2"], lang))
    m += macro("PriceBeta", num(p["beta"], 2, lang)) + macro("PriceSE", num(p["se"], 2, lang))
    m += macro("PriceT", num(w["t"], 2, lang)) + macro("PriceP", num(p["p"], 2, lang))
    m += macro("PriceWCBp", num(w["p_wcb"], 2, lang)) + macro("WCBreps", integer(w["B"], lang))
    m += macro("NobsPrice", integer(p["n"], lang)) + macro("Nindustries", integer(p["n_cluster"], lang))
    m += macro("NobsEvent", integer(res["event_n"], lang))
    m += macro("NproductsEvent", integer(res["event_products"], lang))
    m += macro("EventPreN", integer(len(pre), lang))
    m += macro("EventPreSig", integer(int((pre["p"] < 0.05).sum()), lang))
    m += macro("EventTroughQ", trough_q.replace("Q", "T") if lang == "fr" else trough_q)
    m += macro("EventTrough", num(ev.loc[trough_q, "beta"], 2, lang))
    m += macro("EventFirstWaveA", num(ev.loc["2018Q3", "beta"], 2, lang))
    m += macro("EventFirstWaveB", num(ev.loc["2018Q4", "beta"], 2, lang))
    m += macro("EventLast", num(ev.loc["2021Q4", "beta"], 2, lang))
    mean_dtau = float(dict(res["desc"])["dtau"].mean())
    m += macro("MeanDtau", num(mean_dtau, 3, lang))
    m += macro("EventTroughMeanEffect",
               num(100 * (1 - math.exp(ev.loc[trough_q, "beta"] * mean_dtau)), 0, lang))
    m += macro("EventPostSig", integer(int((post["p"] < 0.05).sum()), lang))
    m += macro("EventPostN", integer(len(post), lang))
    m += macro("NobsStacked", integer(res["stack_n"], lang))
    m += macro("Nstacks", integer(len(counts), lang))
    m += macro("NtreatedStacked", integer(counts["n_treated"].sum(), lang))
    m += macro("NcontrolsMin", integer(counts["n_control"].min(), lang))
    m += macro("NcontrolsMax", integer(counts["n_control"].max(), lang))
    m += macro("FirstCohort", quarter_label(int(counts["stack"].min())))
    m += macro("LastCohort", quarter_label(int(counts["stack"].max())))
    m += macro("LargestCohort", integer(int(counts["n_treated"].max()), lang))
    m += macro("NcellsProduct", integer(s["n_fe1"] * s["n_fe2"], lang))
    m += macro("StackedUnw", num(post_mean(unw), 2, lang)) + macro("StackedW", num(post_mean(wtd), 2, lang))
    m += macro("StackedWimpact", num(wtd.loc[0, "beta"], 2, lang))
    m += macro("StackedWpreMinP", num(wtd.loc[[-4, -3, -2], "p"].min(), 2, lang))
    m += macro("StackedUnwPreMinP", num(unw.loc[[-4, -3, -2], "p"].min(), 2, lang))
    for name, t in (("Unw", tu), ("W", tw)):
        for i, tag in enumerate(("Three", "Five", "Ten")):
            m += macro(f"Stacked{name}{tag}", num(t.loc[i, "post_mean_beta"], 2, lang))
    idt = res["ident"]
    for tag, key in (("Base", "baseline"), ("Ref", "reference_2017Q4"), ("Split", "split_sample")):
        r = idt[key]
        m += macro(f"Id{tag}PostMean", num(r["post_mean"], 2, lang))
        m += macro(f"Id{tag}PostSE", num(r["post_mean_se"], 2, lang))
        m += macro(f"Id{tag}PreMaxT", num(r["pre_max_abs_t"], 2, lang))
        m += macro(f"Id{tag}PostSig", integer(r["post_sig"], lang))
    r = idt["reference_2017Q4"]["coef"]
    m += macro("IdRefQOne", num(r.loc["2018Q1", "beta"], 2, lang))
    m += macro("IdRefQTwo", num(r.loc["2018Q2", "beta"], 2, lang))
    m += macro("IdRefQOneP", num(r.loc["2018Q1", "p"], 2, lang))
    m += macro("IdRefQTwoP", num(r.loc["2018Q2", "p"], 2, lang))
    m += macro("IdSplitN", integer(idt["split_sample"]["n"], lang))
    m += macro("IdSplitProducts", integer(idt["split_sample"]["products"], lang))
    m += macro("CellsMax", num(cells["beta"].iloc[-1], 2, lang))
    m += macro("CellsMaxN", integer(cells["n"].iloc[-1], lang))
    fr_ = res["h2_frictions"].reset_index(drop=True)
    eta, pt, rho, sh = res["eta_hat"], res["passthrough"], res["rho_tau"], res["shares"]
    m += macro("EtaHat", num(eta["eta"], 2, lang)) + macro("EtaHatSE", num(eta["se"], 2, lang))
    m += macro("EtaLevel", num(eta["eta_level"], 2, lang))
    m += macro("ThetaHat", num(pt["theta"], 3, lang)) + macro("ThetaHatSE", num(pt["se"], 3, lang))
    m += macro("ThetaLo", num(pt["theta"] - 1.96 * pt["se"], 2, lang))
    m += macro("ThetaHi", num(pt["theta"] + 1.96 * pt["se"], 2, lang))
    m += macro("ThetaDuration", num(1.0 / (1.0 - pt["theta"]), 1, lang))
    m += macro("CPTFour", num(pt["cpt"], 2, lang)) + macro("CPTFourSE", num(pt["cpt_se"], 2, lang))
    m += macro("CPTZero", num(pt["path"].loc[0, "cpt"], 2, lang))
    implied = pt["path"]["theta"].dropna()
    m += macro("ThetaHorizonMin", num(implied.min(), 2, lang))
    m += macro("ThetaHorizonMax", num(implied.max(), 2, lang))
    m += macro("ERPTn", integer(pt["n"], lang))
    m += macro("ERPTTariff", num(pt["tariff_coef"], 2, lang))
    m += macro("ERPTTariffSE", num(pt["tariff_se"], 2, lang))
    m += macro("RhoTau", num(rho["rho_tau"], 2, lang)) + macro("RhoTauSE", num(rho["se"], 2, lang))
    m += macro("BaseYear", str(sh["year"]))
    m += macro("ExportsBase", num(sh["exports_bn"], 1, lang))
    m += macro("ImportsBase", num(sh["imports_bn"], 1, lang))
    m += macro("GDPBase", integer(round(sh["gdp_bn"]), lang))
    m += macro("ImportShare", num(params.import_share, 2, lang))
    m += macro("Openness", num(params.gamma, 3, lang))
    m += macro("OpennessPct", num(100 * params.gamma, 1, lang))
    m += macro("Imbalance", num(100 * params.imbalance0, 1, lang))
    m += macro("EtaStar", num(params.eta_star, 1, lang))
    m += macro("TradeTerm", num(res["trade_term"], 2, lang))
    m += macro("RObserved", num(res["baseline_R"], 4, lang))
    m += macro("RMin", num(res["r_min"], 3, lang))
    m += macro("RMax", num(fr_.loc[3, "R"], 3, lang))
    m += macro("MaxDeprec", num(100 * params.max_depreciation, 0, lang))
    m += macro("RequiredDeprec", num(100 * res["baseline_req"], 0, lang))
    m += macro("RequiredNoInvoicing", num(100 * fr_.loc[1, "required_deprec"], 0, lang))
    m += macro("RequiredOpenAccount", num(100 * fr_.loc[2, "required_deprec"], 0, lang))
    m += macro("RequiredNoFriction", num(100 * fr_.loc[3, "required_deprec"], 0, lang))
    el = res["elasticity"].reset_index(drop=True)
    m += macro("RequiredEtaLevel", num(100 * el.loc[1, "required_deprec"], 0, lang))
    m += macro("RequiredEtaStarFive", num(100 * el.loc[4, "required_deprec"], 0, lang))
    cs = res["calibration_sens"]
    m += macro("RequiredMinAll", num(100 * pd.concat([cs["required_deprec"], el["required_deprec"],
                                                      res["damping"]["required_deprec"]]).min(), 0, lang))
    m += macro("ExportMargin", num((1 - params.import_share) * params.eta_star, 2, lang))
    m += macro("ImportMargin", num(params.import_share * (params.eta - 1.0), 2, lang))
    est_row = th[th["estimated"]].iloc[0]
    m += macro("EffThetaZero", num(th.iloc[0]["efficiency"], 2, lang))
    m += macro("EffThetaHat", num(est_row["efficiency"], 2, lang))
    m += macro("EffThetaHigh", num(th.iloc[-1]["efficiency"], 2, lang))
    m += macro("ThetaHigh", num(th.iloc[-1]["theta_dollar"], 2, lang))
    iv = res["ge_interval"]
    m += macro("EffThetaLo", num(iv["eff_lo"], 2, lang)) + macro("EffThetaHi", num(iv["eff_hi"], 2, lang))
    m += macro("InvoicingLoPct", num(100 * (iv["eff_lo"] / iv["eff_zero"] - 1), 0, lang))
    m += macro("InvoicingHiPct", num(100 * (iv["eff_hi"] / iv["eff_zero"] - 1), 0, lang))
    m += macro("WedgeCutLo", num(100 * (1 - max(iv["wedge_lo"], iv["wedge_hi"])), 0, lang))
    m += macro("WedgeCutHi", num(100 * (1 - min(iv["wedge_lo"], iv["wedge_hi"])), 0, lang))
    base_path = res["irf_base"]
    m += macro("ReversalBase", integer(int((base_path["e"] < 0).idxmax()), lang))
    m += macro("ImportPeakQ", integer(int(base_path["pm"].idxmax()), lang))
    m += macro("InvoicingCut", num(100 * (1 - est_row["efficiency"] / th.iloc[0]["efficiency"]), 0, lang))
    m += macro("EffChiZero", num(ch.iloc[0]["efficiency"], 2, lang))
    m += macro("WedgeCutOne", num(100 * (1 - ch[ch["chi"] == 1.0]["efficiency"].iloc[0] / ch.iloc[0]["efficiency"]), 0, lang))
    m += macro("EffChiFour", num(ch[ch["chi"] == 4.0]["efficiency"].iloc[0], 2, lang))
    m += macro("WedgeCut", num(100 * (1 - ch[ch["chi"] == 4.0]["efficiency"].iloc[0] / ch.iloc[0]["efficiency"]), 0, lang))
    m += macro("PeakThetaZero", num(th.iloc[0]["peak_rmb_appreciation"], 2, lang))
    m += macro("PeakThetaHat", num(est_row["peak_rmb_appreciation"], 2, lang))
    m += macro("GapThetaZero", num(th.iloc[0]["GAP_cum"], 2, lang))
    m += macro("GapThetaHat", num(est_row["GAP_cum"], 2, lang))
    m += macro("GapThetaHigh", num(th.iloc[-1]["GAP_cum"], 2, lang))
    m += macro("NXThetaZero", num(th.iloc[0]["NX_cum"], 2, lang))
    m += macro("NXThetaHat", num(est_row["NX_cum"], 2, lang))
    m += macro("NXThetaHigh", num(th.iloc[-1]["NX_cum"], 2, lang))
    for tag, path in (("Open", res["irf"][0.0]), ("Closed", res["irf"][4.0])):
        m += macro(f"Reversal{tag}", integer(int((path["e"] < 0).idxmax()), lang))
        m += macro(f"NXNegative{tag}", integer(int((path["nx"] < 0).idxmax()), lang))
    m += macro("NXChiZero", num(ch.iloc[0]["NX_cum"], 2, lang))
    m += macro("NXChiFour", num(ch[ch["chi"] == 4.0]["NX_cum"].iloc[0], 2, lang))
    m += macro("GapChiZero", num(ch.iloc[0]["GAP_cum"], 2, lang))
    m += macro("GapChiFour", num(ch[ch["chi"] == 4.0]["GAP_cum"].iloc[0], 2, lang))
    pz = res["ge_persistence"].set_index("rho_z")
    for tag, rz in (("Low", 0.75), ("High", 0.95)):
        r = pz.loc[rz]
        m += macro(f"RhoZ{tag}", num(rz, 2, lang))
        m += macro(f"PersNX{tag}", num(r["nx_hat"], 2, lang))
        m += macro(f"PersNXOpen{tag}", num(r["nx_chi_0"], 2, lang))
        m += macro(f"PersNXClosed{tag}", num(r["nx_chi_4"], 3 if tag == "Low" else 2, lang))
    r = pz.loc[0.95]
    m += macro("PersInvRatio", num(r["eff_hat"] / r["eff_theta_0"], 2, lang))
    m += macro("PersWedgeRatio", num(r["eff_chi_4"] / r["eff_chi_0"], 2, lang))
    m += macro("PersNXTransitory", num(pz.loc[0.5, "nx_hat"], 2, lang))
    for tag, layer in (("Est", "estimation"), ("Full", "full")):
        mc = res["mc"][layer]
        m += macro(f"McRank{tag}", num(100 * mc["share_ranking"], 1, lang))
        m += macro(f"McMagnitude{tag}", num(100 * mc["share_larger_magnitude"], 1, lang))
        m += macro(f"McDet{tag}", num(100 * mc["n_determinate"] / mc["n"], 1, lang))
        m += macro(f"McInvNeg{tag}", num(100 * mc["share_invoicing_negative"], 1, lang))
        m += macro(f"McWedgeNeg{tag}", num(100 * mc["share_wedge_negative"], 1, lang))
        for key, name in (("invoicing", "Inv"), ("wedge", "Wedge"), ("difference", "Diff"),
                          ("required_deprec", "Req")):
            for qk, qn in (("q05", "Lo"), ("q50", "Med"), ("q95", "Hi")):
                m += macro(f"Mc{name}{qn}{tag}", num(100 * mc[key][qk], 0, lang))
        m += macro(f"McFeasible{tag}", num(100 * mc["share_feasible"], 1, lang))
    m += macro("McDraws", integer(res["mc"]["full"]["n"], lang))
    m += macro("McBeyond", num(100 * max(res["mc"][k]["share_beyond_100"] for k in res["mc"]), 1, lang))
    fl = res["mc_failures"]
    m += macro("McFailChi", num(fl["chi_median"], 2, lang))
    m += macro("McFailPsi", num(fl["psi_median"], 3, lang))
    m += macro("McRankChiHigh", num(100 * fl["share_ranking_chi_high"], 1, lang))
    m += macro("McRankPsiLow", num(100 * fl["share_ranking_psi_low"], 1, lang))
    drv = res["mc_drivers"].set_index("parameter")["rho_difference"]
    m += macro("McRhoPsi", num(drv["portfolio_cost"], 2, lang))
    m += macro("McRhoChi", num(drv["chi"], 2, lang))
    m += macro("McRhoXi", num(drv["theta"], 2, lang))
    m += macro("McRhoTheta", num(drv["theta_dollar"], 2, lang))
    m += macro("McRhoMaxOther", num(drv.drop(["portfolio_cost", "chi", "theta", "theta_dollar"]).abs().max(), 2, lang))
    m += macro("GeSensN", integer(len(gs), lang)) + macro("GeSensDet", integer(len(det), lang))
    m += macro("GeInvMin", num(det["invoicing_ratio"].min(), 2, lang))
    m += macro("GeInvMax", num(det["invoicing_ratio"].max(), 2, lang))
    m += macro("GeWedgeMin", num(det["wedge_ratio"].min(), 2, lang))
    m += macro("GeWedgeMax", num(det["wedge_ratio"].max(), 2, lang))
    m += macro("GeRankHolds", integer(int((det["wedge_ratio"] < det["invoicing_ratio"]).sum()), lang))
    cv = res["calibration_validation"].set_index("calibration")
    for tag, key in (("Lit", "literature"), ("Data", "data_disciplined")):
        r = cv.loc[key]
        m += macro(f"Calib{tag}Required", num(100 * r["required_observed"], 0, lang))
        m += macro(f"Calib{tag}NoFriction", num(100 * r["required_no_friction"], 0, lang))
        m += macro(f"Calib{tag}InvRatio", num(r["invoicing_ratio"], 2, lang))
        m += macro(f"Calib{tag}WedgeRatio", num(r["wedge_ratio"], 2, lang))
    m += macro("CalibRankHolds", integer(int(cv["ranking_holds"].sum()), lang))
    m += macro("CalibN", integer(len(cv), lang))
    f = res["firm"]
    if f is not None:
        m += macro("FirmBeta", num(f["beta"], 2, lang)) + macro("FirmP", num(f["p"], 2, lang))
        m += macro("FirmBetaTrend", num(f["beta_trend"], 2, lang))
        m += macro("FirmPTrend", num(f["p_trend"], 2, lang))
        m += macro("FirmN", integer(f["n"], lang))
    return m


def table_correlation(res: dict, lang: str) -> str:
    """Pairwise correlations of the analysis variables, in unit-consistent blocks."""
    i = 0 if lang == "en" else 1
    ncol = 3
    blocks = [
        (("Panel A. Product panel", "Panneau A. Panel de produits"),
         res["corr_product"], ["effective_tariff", "log_value", "dtau"],
         [("Effective tariff", "Tarif effectif", "\\tau"),
          ("Import value", "Valeur importée", "v"),
          ("Tariff-shock intensity", "Intensité du choc tarifaire", "\\Delta\\tau_i")]),
        (("Panel B. Aggregate quarterly series", "Panneau B. Séries trimestrielles agrégées"),
         res["corr_aggregate"], ["tau", "p", "e"],
         [("Aggregate effective tariff", "Tarif effectif agrégé", "\\tau_t"),
          ("Aggregate import price", "Prix agrégé à l'importation", "p^{m}_t"),
          ("Renminbi value", "Valeur du renminbi", "e_t")]),
    ]
    out = ("\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{6.2cm}" + "c" * ncol
           + "@{}}\n\\toprule\n")
    out += " & " + " & ".join(f"({j + 1})" for j in range(ncol)) + " \\\\\n\\midrule\n"
    parts = []
    for (pen, pfr), M, cols, vardefs in blocks:
        b = _panel_head(ncol + 1, (pen, pfr)[i])
        Mv = M.loc[cols, cols].to_numpy()
        for r, (en, fr, sym) in enumerate(vardefs):
            cells = [num(Mv[r, c], 2, lang) if c <= r else "" for c in range(ncol)]
            b += _rule_row(f"({r + 1}) \\emph{{{(en, fr)[i]}}} ${sym}$", cells)
        parts.append(b)
    return out + "\\addlinespace\n".join(parts) + "\\bottomrule\n\\end{tabular}\n"


# ---------------------------------------------------------------------------------------

TABLES = {
    "tab_incidence": table_incidence, "tab_staggered": table_staggered, "tab_cells": table_cells,
    "tab_rebalancing": table_rebalancing, "tab_ge": table_ge,
    "tab_identification": table_identification,
    "tab_ge_sensitivity": table_ge_sensitivity, "tab_event": table_event,
    "tab_calibration_validation": table_calibration_validation,
    "tab_calibration": table_calibration, "tab_descriptive": table_descriptive,
    "tab_correlation": table_correlation,
    "tab_passthrough": table_passthrough, "tab_uncertainty": table_uncertainty,
    "tab_uncertainty_ranges": table_uncertainty_ranges,
}


def make_all(res: dict | None = None) -> int:
    """Write every table and the number macros for both languages; return the file count."""
    res = res or collect()
    n = 0
    for lang in LANGS:
        for name, fn in TABLES.items():
            write_fragment(lang, name, fn(res, lang))
            n += 1
        write_fragment(lang, "numbers", numbers(res, lang))
        n += 1
    return n


if __name__ == "__main__":
    print(f"wrote {make_all()} table fragments (EN+FR)")
