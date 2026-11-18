"""Run the Problem 3 pipeline: top-bracket bunching, Pareto/MVPF calibration, tables, and figures."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source.bunching.loader import load_counts, pool_counts
from source.bunching.model import estimate
from source.bunching.mvpf import (
    PS1_PARETO_ALPHAS,
    calibrate_mvpf_table,
    estimate_pareto_alpha_binned,
)
from source.plots import FIGURES_DIR, save_fig
from source.plots.p3 import plot_mvpf_calibration, plot_top_bracket_bunching


def main():
    synth_path = ROOT / "data" / "synthetic_bunch_counts.csv"
    thr_path = ROOT / "data" / "threshold.csv"

    z23, c23 = load_counts(str(synth_path), year=2023)
    thr_map = pd.read_csv(thr_path).set_index("year")["threshold_clp"].to_dict()
    years = sorted(thr_map.keys())
    counts_all = {y: load_counts(str(synth_path), year=y) for y in years}

    # 1. Top-bracket bunching specifications (tau_below = 0.35, tau_above = 0.40)
    # 150 UTA benchmark using paper UTA (9,907,434 / 13.5 = 733,884 CLP/UTA -> 110,082,600 CLP)
    z_150_bench = (150.0 / 13.5) * 9_907_434
    z_150_dec = (150.0 / 13.5) * 10_402_932
    z_150_mode = 113_500_000.0
    z_120_bench = (120.0 / 13.5) * 9_907_434

    fit_150_mode = estimate(
        z23, c23, z_150_mode, 80e6, 155e6, 110e6, 116e6, degree=4, tau0=0.35, tau1=0.40
    )
    fit_150_bench = estimate(
        z23, c23, z_150_bench, 80e6, 155e6, 107e6, 113e6, degree=4, tau0=0.35, tau1=0.40
    )
    fit_150_dec = estimate(
        z23, c23, z_150_dec, 80e6, 155e6, 112e6, 118e6, degree=4, tau0=0.35, tau1=0.40
    )
    fit_120_bench = estimate(
        z23, c23, z_120_bench, 60e6, 130e6, 85e6, 91e6, degree=4, tau0=0.35, tau1=0.40
    )

    # Pooled 2005-2023 normalized by 120 UTA
    scale_120 = {y: float(thr_map[y]) * (120.0 / 13.5) for y in years}
    zp_120, cp_120 = pool_counts(counts_all, scale_120, width=0.01)
    fit_pool_120 = estimate(
        zp_120, cp_120, 1.0, 0.65, 1.45, 0.90, 1.10, degree=4, tau0=0.35, tau1=0.40
    )

    # Synthetic first-kink reference (K = 9,907,434 CLP, tau0=0, tau1=0.04)
    fit_kink1 = estimate(
        z23, c23, 9_907_434.0, 6.5e6, 20e6, 8.5e6, 10.5e6, degree=4, tau0=0.0, tau1=0.04
    )

    bunch_rows = [
        {
            "spec": "2023 Top Kink: 150 UTA empirical mode ($113.50M CLP)",
            "spec_short": "2023 150 UTA mode (113.5M)",
            "z_star_str": "\\$113.50M",
            "window_str": "[\\$80M, \\$155M]",
            "excluded_str": "[\\$110M, \\$116M)",
            "B": fit_150_mode["B"],
            "b_str": f"\\${fit_150_mode['b']:,.0f}",
            "epsilon": fit_150_mode["epsilon"],
        },
        {
            "spec": "2023 Top Kink: 150 UTA statutory ($110.08M CLP)",
            "spec_short": "2023 150 UTA bench (110.1M)",
            "z_star_str": "\\$110.08M",
            "window_str": "[\\$80M, \\$155M]",
            "excluded_str": "[\\$107M, \\$113M)",
            "B": fit_150_bench["B"],
            "b_str": f"\\${fit_150_bench['b']:,.0f}",
            "epsilon": fit_150_bench["epsilon"],
        },
        {
            "spec": "2023 Top Kink: 150 UTA Dec.UTA ($115.59M CLP)",
            "spec_short": "2023 150 UTA Dec (115.6M)",
            "z_star_str": "\\$115.59M",
            "window_str": "[\\$80M, \\$155M]",
            "excluded_str": "[\\$112M, \\$118M)",
            "B": fit_150_dec["B"],
            "b_str": f"\\${fit_150_dec['b']:,.0f}",
            "epsilon": fit_150_dec["epsilon"],
        },
        {
            "spec": "2023 Upper Bracket: 120 UTA statutory ($88.07M CLP)",
            "spec_short": "2023 120 UTA bench (88.1M)",
            "z_star_str": "\\$88.07M",
            "window_str": "[\\$60M, \\$130M]",
            "excluded_str": "[\\$85M, \\$91M)",
            "B": fit_120_bench["B"],
            "b_str": f"\\${fit_120_bench['b']:,.0f}",
            "epsilon": fit_120_bench["epsilon"],
        },
        {
            "spec": "Pooled 2005--2023: 120 UTA normalized ($\\tilde{z}^*=1.0$)",
            "spec_short": "Pooled 05-23 (120 UTA)",
            "z_star_str": "$1.00\\,K_{120}$",
            "window_str": "$[0.65, 1.45]$",
            "excluded_str": "$[0.90, 1.10)$",
            "B": fit_pool_120["B"],
            "b_str": f"{fit_pool_120['b']:.4f} $K_{{120}}$",
            "epsilon": fit_pool_120["epsilon"],
        },
    ]
    bunch_df = pd.DataFrame(bunch_rows)

    out_tables = ROOT / "output" / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    bunch_df.to_csv(out_tables / "p3_bunching_specs.csv", index=False)

    # 2. Check upper-tail Pareto on synthetic_bunch_counts.csv (for diagnostic reporting)
    pareto_diag = estimate_pareto_alpha_binned(z23, c23, z_120_bench)
    print("Synthetic 2023 tail above 88.07M:", pareto_diag)

    # 3. Calibrate MVPF across PS1 Pareto alphas and ETI scenarios
    eps_scenarios = {
        "Top-bracket bunching (150 UTA mode, $\\hat{\\epsilon}=0.0245$)": fit_150_mode["epsilon"],
        "Top-bracket bunching (150 UTA statutory, $\\hat{\\epsilon}=0.0155$)": fit_150_bench["epsilon"],
        "Upper-bracket pooled (120 UTA, $\\hat{\\epsilon}=0.0260$)": fit_pool_120["epsilon"],
        "Synthetic first-kink reference ($\\hat{\\epsilon}=0.2570$)": fit_kink1["epsilon"],
    }
    alpha_scenarios = {
        "PS1 Vermeulen MLE ($w_{\\min}=380\\text{M}$, $\\hat{\\alpha}=1.859$)": 1.859,
        "PS1 SII Top Bracket ($>310\\text{ UTA}$, $\\hat{\\alpha}=2.085$)": 2.085,
        "PS1 Gabaix--Ibragimov Reg ($w_{\\min}=380\\text{M}$, $\\hat{\\alpha}=2.135$)": 2.135,
        "PS1 Vermeulen MLE ($w_{\\min}=P99$, $\\hat{\\alpha}=2.531$)": 2.531,
    }

    mvpf_df = calibrate_mvpf_table(eps_scenarios, alpha_scenarios, tau0=0.40, tau1=0.35)
    mvpf_df.to_csv(out_tables / "p3_mvpf_calibration.csv", index=False)

    # Write LaTeX table 1: Top-bracket bunching estimates
    tex_bunch_rows = []
    for r in bunch_df.itertuples(index=False):
        tex_bunch_rows.append(
            f"{r.spec_short} & {r.z_star_str} & {r.window_str} & {r.excluded_str} "
            f"& {r.B:,.1f} & {r.b_str} & {r.epsilon:.4f} \\\\"
        )
    tex_bunch_Rows_str = "\n".join(tex_bunch_rows)
    tex_bunch = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Top-bracket bunching elasticity estimates from \\texttt{{synthetic\\_bunch\\_counts.csv}} ($\\tau_{{\\text{{below}}}}=0.35$, $\\tau_{{\\text{{above}}}}=0.40$)}}
\\label{{tab:p3_bunching}}
\\small
\\begin{{adjustbox}}{{max width=\\textwidth}}
\\begin{{tabular}}{{lcccrrr}}
\\toprule
Specification & Cutoff $z^*$ & Window $[z_l, z_r]$ & Excluded $[z_L, z_R)$ & $\\hat{{B}}$ & $\\hat{{b}}$ & $\\hat{{\\epsilon}}$ \\\\
\\midrule
{tex_bunch_Rows_str}
\\midrule
Synthetic 1st kink ($0\\%\\to 4\\%$) & \\$9.91M & [\\$6.5M, \\$20M] & [\\$8.5M, \\$10.5M) & {fit_kink1['B']:,.1f} & \\${fit_kink1['b']:,.0f} & {fit_kink1['epsilon']:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{adjustbox}}
\\begin{{minipage}}{{\\textwidth}}
\\vspace{{4pt}}
\\footnotesize\\textit{{Note:}} Fourth-degree polynomial counterfactuals with Chetty et al.\\ (2011) iterative right-tail reweighting and \\textit{{mean}} density normalization. Upper-bracket specifications evaluate Equation~(1) at $\\ln[(1-0.35)/(1-0.40)] = \\ln(13/12) \\approx 0.08004$; the bottom reference row evaluates the first kink ($K = \\$9{{,}}907{{,}}434$ CLP) on the synthetic dataset at $\\ln(1/0.96) \\approx 0.04082$.
\\end{{minipage}}
\\end{{table}}
"""
    (ROOT / "report" / "tables" / "p3_bunching.tex").write_text(tex_bunch)

    # Write LaTeX table 2: MVPF calibration
    tex_mvpf_rows = []
    current_eps = None
    for r in mvpf_df.itertuples(index=False):
        if current_eps is not None and r.epsilon_label != current_eps:
            tex_mvpf_rows.append("\\midrule")
        current_eps = r.epsilon_label
        tex_mvpf_rows.append(
            f"{r.alpha_label} & {r.epsilon:.4f} & {r.alpha:.3f} & {r.k_eps_alpha:.4f} "
            f"& {100*r.leakage_tau0:.2f}\\% & {r.mvpf_tau0:.4f} & {r.mvpf_tau1:.4f} "
            f"& {r.mvpf_midpoint:.4f} & {r.mvpf_exact:.4f} \\\\"
        )
    tex_mvpf_body = "\n".join(tex_mvpf_rows)
    tex_mvpf = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Calibrated top-bracket $|\\mathrm{{MVPF}}|$ across Problem Set 1 Pareto parameters ($\\hat{{\\alpha}}$) and bunching elasticities ($\\hat{{\\epsilon}}$)}}
\\label{{tab:p3_mvpf}}
\\small
\\begin{{adjustbox}}{{max width=\\textwidth}}
\\begin{{tabular}}{{lrrrrrrrr}}
\\toprule
Pareto tail specification (PS1) & $\\hat{{\\epsilon}}$ & $\\hat{{\\alpha}}$ & $\\hat{{\\epsilon}}\\hat{{\\alpha}}$ & Offset ($\\tau_0$) & $|\\mathrm{{MVPF}}(\\tau_0)|$ & $|\\mathrm{{MVPF}}(\\tau_1)|$ & $|\\mathrm{{MVPF}}(\\bar{{\\tau}})|$ & $\\overline{{|\\mathrm{{MVPF}}|}}_{{\\mathrm{{exact}}}}$ \\\\
\\midrule
{tex_mvpf_body}
\\bottomrule
\\end{{tabular}}
\\end{{adjustbox}}
\\begin{{minipage}}{{\\textwidth}}
\\vspace{{4pt}}
\\footnotesize\\textit{{Note:}} Local MVPFs evaluated at $\\tau_0 = 0.40$, $\\tau_1 = 0.35$, and midpoint $\\bar{{\\tau}} = (\\tau_0+\\tau_1)/2 = 0.375$, alongside the exact analytical integral $\\overline{{|\\mathrm{{MVPF}}|}}_{{\\mathrm{{exact}}}}$ over $[0.35, 0.40]$. ``Offset ($\\tau_0$)'' denotes the behavioral revenue share $-\\Delta R^B / \\Delta R^M = \\frac{{\\tau_0}}{{1-\\tau_0}}\\hat{{\\epsilon}}\\hat{{\\alpha}}$ at $\\tau_0 = 0.40$. The four groups of rows correspond to: (1) 2023 top-kink bunching at the 150 UTA empirical mode ($\\hat{{\\epsilon}}=0.0245$), (2) 2023 150 UTA statutory benchmark ($\\hat{{\\epsilon}}=0.0155$), (3) pooled 2005--2023 120 UTA bunching ($\\hat{{\\epsilon}}=0.0260$), and (4) the synthetic first-kink upper-bound benchmark ($\\hat{{\\epsilon}}=0.2570$).
\\end{{minipage}}
\\end{{table}}
"""
    (ROOT / "report" / "tables" / "p3_mvpf.tex").write_text(tex_mvpf)

    # 4. Generate figures
    p3_fig_dir = FIGURES_DIR / "p3"
    p3_fig_dir.mkdir(parents=True, exist_ok=True)

    fig1, axes1 = plt.subplots(1, 2, figsize=(8.8, 3.3))
    plot_top_bracket_bunching(
        z23,
        c23,
        fit_150_mode["c_hat"],
        z_150_mode,
        110e6,
        116e6,
        (80e6, 155e6),
        bunch_df,
        fig=fig1,
        axes=axes1,
    )
    save_fig(p3_fig_dir / "p3_top_bunching.pdf", fig=fig1)
    plt.close(fig1)

    alpha_plot_dict = {
        r"PS1 MLE ($w_{\min}=380\mathrm{M}$, $\hat{\alpha}=1.86$)": 1.859,
        r"PS1 SII Top Bracket ($\hat{\alpha}=2.09$)": 2.085,
        r"PS1 Reg ($w_{\min}=380\mathrm{M}$, $\hat{\alpha}=2.14$)": 2.135,
        r"PS1 MLE ($w_{\min}=P99$, $\hat{\alpha}=2.53$)": 2.531,
    }
    fig2, axes2 = plt.subplots(1, 2, figsize=(8.8, 3.4))
    plot_mvpf_calibration(
        alpha_plot_dict,
        eps_top=fit_150_mode["epsilon"],
        eps_kink1=fit_kink1["epsilon"],
        tau0=0.40,
        tau1=0.35,
        fig=fig2,
        axes=axes2,
    )
    save_fig(p3_fig_dir / "p3_mvpf_calibration.pdf", fig=fig2)
    plt.close(fig2)

    print("Successfully generated tables and figures for Problem 3.")
    print(mvpf_df[["alpha_label", "epsilon", "alpha", "mvpf_tau0", "mvpf_tau1", "mvpf_midpoint", "mvpf_exact"]].to_string(index=False))


if __name__ == "__main__":
    main()
