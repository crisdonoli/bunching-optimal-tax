"""Plotting helpers for Section 3: top-bracket bunching and MVPF calibration."""

import numpy as np
import matplotlib.pyplot as plt

from . import FONTSIZES, FIGURES_DIR, save_fig
from ..bunching.mvpf import mvpf_local, mvpf_average_midpoint, mvpf_average_exact


def plot_top_bracket_bunching(
    z,
    c,
    c_hat_150,
    z_star_150,
    z_L_150,
    z_R_150,
    xlim_150,
    sens_df,
    fig=None,
    axes=None,
):
    """Two-panel diagnostic for top-bracket bunching in synthetic_bunch_counts.csv.

    Left panel: observed vs. Chetty-reweighted counterfactual counts in the upper
    tail around the 150 UTA top-bracket cutoff (2023).
    Right panel: sensitivity of the estimated top-bracket elasticity across
    candidate upper-bracket cutoffs and normalization rules.
    """
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.3))

    ax0, ax1 = axes
    z_m = np.asarray(z, float) / 1e6
    c = np.asarray(c, float)
    c_hat = np.asarray(c_hat_150, float)

    mask = (z >= xlim_150[0]) & (z <= xlim_150[1])
    ax0.axvspan(z_L_150 / 1e6, z_R_150 / 1e6, color="0.88", alpha=0.65, label="Excluded band")
    ax0.bar(
        z_m[mask],
        c[mask],
        width=0.085,
        color="steelblue",
        alpha=0.55,
        label="Observed counts ($c_j$)",
    )
    ax0.plot(
        z_m[mask],
        c_hat[mask],
        color="darkred",
        lw=1.8,
        label=r"Counterfactual ($\hat{c}_j$)",
    )
    ax0.axvline(
        z_star_150 / 1e6,
        color="black",
        ls=":",
        lw=1.2,
        label=rf"Top kink $z^* = {z_star_150/1e6:.1f}\mathrm{{M}}$",
    )
    ax0.set_xlim(xlim_150[0] / 1e6, xlim_150[1] / 1e6)
    ax0.set_xlabel("Taxable income $z$ (million CLP)", fontsize=FONTSIZES["label"])
    ax0.set_ylabel("Taxpayers per bin", fontsize=FONTSIZES["label"])
    ax0.set_title("Upper-tail bunching around 150 UTA, 2023", fontsize=FONTSIZES["title"])
    ax0.grid(True, linestyle=":", alpha=0.45)
    ax0.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="0.85", fontsize=FONTSIZES["legend"] - 1)
    ax0.tick_params(labelsize=FONTSIZES["tick"])

    # Right panel: horizontal comparison of estimated elasticities across top-bracket specs
    y_pos = np.arange(len(sens_df))
    ax1.barh(
        y_pos,
        sens_df["epsilon"],
        height=0.55,
        color="darkblue",
        alpha=0.78,
        edgecolor="black",
        lw=0.6,
    )
    for i, val in enumerate(sens_df["epsilon"]):
        ax1.text(
            val + 0.0006,
            i,
            f"{val:.4f}",
            va="center",
            fontsize=FONTSIZES["annotation"] - 1,
        )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sens_df["spec_short"], fontsize=FONTSIZES["tick"] - 0.5)
    ax1.set_xlim(0.0, max(sens_df["epsilon"]) * 1.28)
    ax1.set_xlabel(r"Estimated top-bracket ETI $\hat{\epsilon}$", fontsize=FONTSIZES["label"])
    ax1.set_title("Sensitivity across top-bracket specifications", fontsize=FONTSIZES["title"])
    ax1.grid(True, axis="x", linestyle=":", alpha=0.45)
    ax1.tick_params(labelsize=FONTSIZES["tick"])
    ax1.invert_yaxis()

    fig.tight_layout()
    return fig, axes


def plot_mvpf_calibration(
    alphas_dict,
    eps_top=0.0245,
    eps_kink1=0.2570,
    tau0=0.40,
    tau1=0.35,
    fig=None,
    axes=None,
):
    """Two-panel MVPF calibration figure.

    Left panel: |MVPF(tau_bar)| as a function of the ETI epsilon across PS1 Pareto
    tail indices alpha, marking both the upper-tail bunching estimate and the
    primary-kink benchmark.
    Right panel: Local |MVPF(tau)| trajectory over tau in [0.32, 0.43] showing
    tau_1 = 0.35, tau_bar = 0.375, exact integral average, and tau_0 = 0.40.
    """
    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))

    ax0, ax1 = axes
    tau_bar = 0.5 * (tau0 + tau1)
    eps_grid = np.linspace(0.0, 0.45, 300)

    colors = ["darkblue", "darkred", "seagreen", "darkorange"]
    linestyles = ["-", "--", "-.", ":"]

    for (label, alpha), col, ls in zip(alphas_dict.items(), colors, linestyles):
        mvpf_curve = mvpf_local(tau_bar, eps_grid, alpha)
        ax0.plot(eps_grid, mvpf_curve, color=col, ls=ls, lw=1.7, label=label)

    ax0.axvline(
        eps_top,
        color="black",
        ls=":",
        lw=1.2,
        label=rf"Top-tail bunching ($\hat{{\epsilon}} = {eps_top:.4f}$)",
    )
    ax0.axvline(
        eps_kink1,
        color="0.40",
        ls="--",
        lw=1.1,
        label=rf"First-kink synthetic ($\hat{{\epsilon}} = {eps_kink1:.4f}$)",
    )
    ax0.set_xlim(0.0, 0.45)
    ax0.set_ylim(0.98, 2.38)
    ax0.set_xlabel(r"Elasticity of taxable income $\epsilon$", fontsize=FONTSIZES["label"])
    ax0.set_ylabel(r"$|\mathrm{MVPF}(\bar{\tau})|$ at $\bar{\tau} = 0.375$", fontsize=FONTSIZES["label"])
    ax0.set_title(r"$|\mathrm{MVPF}|$ across ETI and PS1 Pareto $\hat{\alpha}$", fontsize=FONTSIZES["title"])
    ax0.grid(True, linestyle=":", alpha=0.45)
    ax0.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="0.85", fontsize=FONTSIZES["legend"] - 1.6)
    ax0.tick_params(labelsize=FONTSIZES["tick"])

    # Right panel: Local MVPF(tau) across the reform interval [0.35, 0.40]
    # Show both a baseline top-tail panel and the first-kink panel via dual y or two curves
    tau_grid = np.linspace(0.32, 0.43, 300)
    alpha_bench = 1.859  # PS1 Vermeulen MLE baseline
    alpha_reg = 2.135    # PS1 Gabaix-Ibragimov Reg

    for eps_val, eps_tag, col in [
        (eps_top, rf"Top-tail $\hat{{\epsilon}}={eps_top:.4f}$", "darkblue"),
        (eps_kink1, rf"First-kink $\hat{{\epsilon}}={eps_kink1:.4f}$", "darkred"),
    ]:
        y_mle = mvpf_local(tau_grid, eps_val, alpha_bench)
        y_reg = mvpf_local(tau_grid, eps_val, alpha_reg)
        ax1.plot(
            tau_grid * 100,
            y_mle,
            color=col,
            ls="-",
            lw=1.7,
            label=rf"{eps_tag}, $\hat{{\alpha}}={alpha_bench:.2f}$ (MLE)",
        )
        ax1.plot(
            tau_grid * 100,
            y_reg,
            color=col,
            ls="--",
            lw=1.4,
            label=rf"{eps_tag}, $\hat{{\alpha}}={alpha_reg:.2f}$ (Reg)",
        )

        # Mark tau1=35%, tau_bar=37.5%, tau0=40% on MLE curve
        pts_tau = np.array([tau1, tau_bar, tau0])
        ax1.scatter(
            pts_tau * 100,
            mvpf_local(pts_tau, eps_val, alpha_bench),
            color=col,
            edgecolor="black",
            s=28,
            zorder=5,
        )

    ax1.axvspan(tau1 * 100, tau0 * 100, color="0.88", alpha=0.55, label=r"Reform band $[35\%, 40\%]$")
    ax1.set_xlim(32, 43)
    ax1.set_ylim(0.98, 1.92)
    ax1.set_xlabel(r"Top marginal tax rate $\tau$ (\%)", fontsize=FONTSIZES["label"])
    ax1.set_ylabel(r"Local $|\mathrm{MVPF}(\tau)|$", fontsize=FONTSIZES["label"])
    ax1.set_title(r"Trajectory of $|\mathrm{MVPF}(\tau)|$ over $[35\%, 40\%]$", fontsize=FONTSIZES["title"])
    ax1.grid(True, linestyle=":", alpha=0.45)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="0.85", fontsize=FONTSIZES["legend"] - 1.8)
    ax1.tick_params(labelsize=FONTSIZES["tick"])

    fig.tight_layout()
    return fig, axes
