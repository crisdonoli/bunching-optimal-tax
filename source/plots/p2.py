"""Illustrative figure for Question 2(ii): alternative counterfactuals in the z x g(z) plane."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from . import FONTSIZES, FIGURES_DIR, save_fig


def _baseline_density(z):
    """Smooth monotone-decreasing baseline density on [0.55, 1.55]."""
    return 1.35 * np.exp(-0.75 * (z - 0.55)) + 0.12


def _baseline_deriv(z):
    """Analytical first derivative of _baseline_density."""
    return -0.75 * 1.35 * np.exp(-0.75 * (z - 0.55))


def build_counterfactual_illustration():
    """Construct stylized observed density with kink spike and two counterfactuals:
    Alt. 1: Shifted-kink / reference-distribution counterfactual g_ref(z)
    Alt. 2: Deep ensemble of censored Normalizing Flows {g_theta^(m)(z)}
    """
    z = np.linspace(0.60, 1.45, 800)
    z_star = 1.00
    z_L, z_R = 0.86, 1.16

    g_true = _baseline_density(z)

    # Observed density h(z): smooth outside [z_L, z_R), spike just below/at z*,
    # and missing-mass deficit on (z*, z_R).
    spike = 0.42 * np.exp(-0.5 * ((z - z_star) / 0.018) ** 2)
    deficit = -0.105 * np.exp(-0.5 * ((z - 1.065) / 0.038) ** 2)
    window = np.where((z > z_L) & (z < z_R), 1.0, 0.0)
    h_obs = g_true + window * (spike + deficit)

    # Alternative 1: Reference / shifted-kink distribution counterfactual g_ref(z)
    # Preserves non-polynomial local curvature (e.g., round-number/institutional wave)
    # inherited from an unperturbed comparison year/group.
    local_wave = 0.045 * np.exp(-0.5 * ((z - 0.99) / 0.11) ** 2) - 0.015 * np.sin(
        2 * np.pi * (z - 0.70) / 0.65
    ) * np.exp(-0.5 * ((z - 1.00) / 0.22) ** 2)
    g_ref = g_true + local_wave

    # Alternative 2: Deep ensemble of censored Normalizing Flows {g_theta^(m)(z)}
    # All flow members fit the uncontaminated exterior region [z_l, z_r] \ [z_L, z_R)
    # nearly identically (tiny out-of-window spread), while exhibiting smooth C^1
    # epistemic variation across the censored band [z_L, z_R).
    # Smooth C^1 bump supported strictly inside [z_L, z_R]
    in_band = (z > z_L) & (z < z_R)
    s = np.zeros_like(z)
    s[in_band] = np.sin(np.pi * (z[in_band] - z_L) / (z_R - z_L)) ** 2

    # Ensemble members with varying spline-knot inductive bias (curvature & skewness)
    shifts = [-0.055, -0.025, 0.005, 0.035, 0.068]
    skews = [0.018, 0.008, -0.002, -0.012, -0.022]
    nf_members = []
    for a, b in zip(shifts, skews):
        # Tiny exterior jitter + smooth C^1 interpolation inside [z_L, z_R)
        perturbation = s * (a + b * (z - z_star) / (0.5 * (z_R - z_L)))
        nf_members.append(g_true + perturbation)

    nf_members = np.asarray(nf_members)
    g_nf_low = np.min(nf_members, axis=0)
    g_nf_high = np.max(nf_members, axis=0)

    return z, z_star, z_L, z_R, h_obs, g_ref, nf_members, g_nf_low, g_nf_high


def plot_counterfactual_alternatives(fig, ax):
    """Plot observed spike at z* against Reference counterfactual and Normalizing Flow ensemble."""
    (
        z,
        z_star,
        z_L,
        z_R,
        h_obs,
        g_ref,
        nf_members,
        g_nf_low,
        g_nf_high,
    ) = build_counterfactual_illustration()

    # Excluded / censored bunching band
    ax.axvspan(
        z_L,
        z_R,
        color="0.90",
        alpha=0.55,
        zorder=0,
        label=r"Censored region $[z_L, z_R)$",
    )
    ax.axvline(z_star, color="black", ls=":", lw=1.1, zorder=2)

    # Censored Normalizing Flow deep ensemble (Alternative 2)
    ax.fill_between(
        z,
        g_nf_low,
        g_nf_high,
        color="seagreen",
        alpha=0.22,
        zorder=2,
        label=r"Alt.\ 2: Censored NF ensemble $\{g_{\theta^{(m)}}(z)\}$",
    )
    for m, g_m in enumerate(nf_members):
        ax.plot(
            z,
            g_m,
            color="seagreen",
            ls="-",
            lw=0.85,
            alpha=0.55 if m not in (0, len(nf_members) - 1) else 0.9,
            zorder=3,
        )

    # Reference-distribution counterfactual (Alternative 1)
    ax.plot(
        z,
        g_ref,
        color="darkred",
        ls="--",
        lw=1.7,
        zorder=4,
        label=r"Alt.\ 1: Reference counterfactual $g^{\mathrm{ref}}(z)$",
    )

    # Observed density with spike at z*
    ax.plot(
        z,
        h_obs,
        color="darkblue",
        lw=1.9,
        zorder=5,
        label=r"Observed density $h(z)$",
    )

    # Mark interval of counterfactual heights at z*
    idx_star = int(np.argmin(np.abs(z - z_star)))
    ax.scatter(
        [z_star, z_star, z_star],
        [g_nf_low[idx_star], g_ref[idx_star], g_nf_high[idx_star]],
        color=["seagreen", "darkred", "seagreen"],
        edgecolor="black",
        s=34,
        zorder=6,
    )

    # Annotations
    ax.annotate(
        r"Bunching spike at $z^*$",
        xy=(z_star - 0.006, h_obs[idx_star] - 0.01),
        xytext=(0.68, 1.42),
        fontsize=FONTSIZES["annotation"],
        arrowprops=dict(arrowstyle="->", color="0.25", lw=0.9),
    )
    ax.annotate(
        r"Missing mass",
        xy=(1.065, h_obs[np.argmin(np.abs(z - 1.065))]),
        xytext=(1.19, 0.72),
        fontsize=FONTSIZES["annotation"],
        arrowprops=dict(arrowstyle="->", color="0.25", lw=0.9),
    )
    ax.annotate(
        r"Epistemic range for $g(z^*)$",
        xy=(z_star - 0.006, 0.5 * (g_nf_low[idx_star] + g_nf_high[idx_star])),
        xytext=(0.63, 0.67),
        fontsize=FONTSIZES["annotation"],
        arrowprops=dict(arrowstyle="->", color="seagreen", lw=0.9),
    )

    ax.set_xticks([z_L, z_star, z_R])
    ax.set_xticklabels([r"$z_L$", r"$z^*$", r"$z_R$"], fontsize=FONTSIZES["tick"])
    ax.set_yticks([])
    ax.set_xlim(0.60, 1.45)
    ax.set_ylim(0.55, 1.56)
    ax.set_xlabel(r"Taxable income $z$", fontsize=FONTSIZES["label"])
    ax.set_ylabel(r"Density $h(z),\; g(z)$", fontsize=FONTSIZES["label"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
        fontsize=FONTSIZES["legend"] - 0.5,
    )

    fig.tight_layout()
    return fig, ax


def plot_uniform_ability_and_income(
    fig,
    axes,
    n_min=1.0,
    n_max=3.0,
    tau0=0.15,
    epsilon=0.5,
):
    """Illustrate the uniform ability density f_N(n) and counterfactual income density g(z)."""
    scale = (1.0 - tau0) ** epsilon
    z_min, z_max = n_min * scale, n_max * scale
    f_height = 1.0 / (n_max - n_min)
    g_height = f_height / scale

    ax0, ax1 = axes

    # Left panel: f_N(n)
    ax0.plot([n_min, n_min, n_max, n_max], [0.0, f_height, f_height, 0.0], color="darkblue", lw=1.8)
    ax0.fill_between([n_min, n_max], [f_height, f_height], color="darkblue", alpha=0.15)
    ax0.set_xlim(0.4, 3.4)
    ax0.set_ylim(0.0, g_height * 1.28)
    ax0.set_xticks([n_min, n_max])
    ax0.set_xticklabels([r"$\underline{n}$", r"$\overline{n}$"], fontsize=FONTSIZES["tick"])
    ax0.set_yticks([0.0, f_height])
    ax0.set_yticklabels([r"$0$", r"$\frac{1}{\overline{n}-\underline{n}}$"], fontsize=FONTSIZES["tick"])
    ax0.set_xlabel(r"Ability $n$", fontsize=FONTSIZES["label"])
    ax0.set_ylabel(r"Density", fontsize=FONTSIZES["label"])
    # ax0.set_title(r"Ability distribution $f_N(n)$", fontsize=FONTSIZES["title"])
    ax0.text(
        0.5 * (n_min + n_max),
        0.5 * f_height,
        r"Area $= 1$",
        ha="center",
        va="center",
        fontsize=FONTSIZES["annotation"],
        color="darkblue",
    )

    # Right panel: g(z) vs dashed f_N(n)
    ax1.plot(
        [n_min, n_min, n_max, n_max],
        [0.0, f_height, f_height, 0.0],
        color="0.55",
        ls="--",
        lw=1.2,
        label=r"$f_N(n)$ (reference)",
    )
    ax1.plot(
        [z_min, z_min, z_max, z_max],
        [0.0, g_height, g_height, 0.0],
        color="darkred",
        lw=1.8,
        label=r"$g(z)$ (compressed by $(1-\tau_0)^\epsilon$)",
    )
    ax1.fill_between([z_min, z_max], [g_height, g_height], color="darkred", alpha=0.15)
    ax1.set_xlim(0.4, 3.4)
    ax1.set_xticks([z_min, z_max])
    ax1.set_xticklabels(
        [r"$\underline{n}(1-\tau_0)^\epsilon$", r"$\overline{n}(1-\tau_0)^\epsilon$"],
        fontsize=FONTSIZES["tick"] - 0.5,
    )
    ax1.set_xlabel(r"Counterfactual income $z$", fontsize=FONTSIZES["label"])
    # ax1.set_title(r"Counterfactual density $g(z)$", fontsize=FONTSIZES["title"])
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        fontsize=FONTSIZES["legend"] - 0.5,
    )
    ax1.text(
        0.5 * (z_min + z_max),
        0.5 * g_height,
        r"Area $= 1$",
        ha="center",
        va="center",
        fontsize=FONTSIZES["annotation"],
        color="darkred",
    )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=FONTSIZES["tick"])

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig, axes


def plot_friction_regimes(
    fig,
    ax,
    z_star=1.0,
    tau0=0.0,
    tau1=0.25,
    epsilon=0.8,
    phi_frac=0.42,
):
    """Visualize the three ability/income regimes under adjustment frictions phi > 0."""
    z0_nB = z_star * ((1.0 - tau0) / (1.0 - tau1)) ** epsilon
    z_max = z0_nB + 0.18 * (z0_nB - z_star)
    z0_min = z_star - 0.25 * (z0_nB - z_star)

    # Exact Delta U(z0) on [z*, z0(n_B)]
    z_band = np.linspace(z_star, z0_nB, 300)
    du_band = (tau1 - tau0) * (z_band - z_star) - (1.0 - tau0) * (
        z_band / (1.0 + epsilon)
        - z_star
        + (epsilon / (1.0 + epsilon)) * z_band * (z_star / z_band) ** (1.0 + 1.0 / epsilon)
    )
    phi_bar = float(du_band[-1])
    phi = phi_frac * phi_bar
    z0_low = float(z_band[np.searchsorted(du_band, phi)])

    # Upper types z0 > z0(n_B): best alternative is z1(n) > z*, not z*
    z_up = np.linspace(z0_nB, z_max, 120)
    du_zstar_up = (tau1 - tau0) * (z_up - z_star) - (1.0 - tau0) * (
        z_up / (1.0 + epsilon)
        - z_star
        + (epsilon / (1.0 + epsilon)) * z_up * (z_star / z_up) ** (1.0 + 1.0 / epsilon)
    )

    # Regime shading
    ax.axvspan(z0_min, z_star, color="0.93", alpha=0.7, label=r"Lower types ($n \le n_0$): stay at $z_0$")
    ax.axvspan(z_star, z0_low, color="goldenrod", alpha=0.18, label=r"Frictional non-bunchers ($\Delta U \le \phi$)")
    ax.axvspan(z0_low, z0_nB, color="seagreen", alpha=0.22, label=r"Actual bunchers ($\Delta U > \phi$)")
    ax.axvspan(z0_nB, z_max, color="steelblue", alpha=0.14, label=r"Upper types ($n > n_B$): move to $z_1 > z^*$")

    # Curves
    ax.plot([z0_min, z_star], [0.0, 0.0], color="0.35", lw=1.8)
    ax.plot(z_band, du_band, color="darkred", lw=2.0, label=r"$\Delta U(z_0) = U(z^*;n) - U(z_0;n)$")
    ax.plot(z_up, du_zstar_up, color="darkred", ls=":", lw=1.3, alpha=0.65)
    ax.axhline(phi, color="black", ls="--", lw=1.3, label=r"Adjustment cost $\phi$")

    ax.axvline(z_star, color="black", ls=":", lw=1.0)
    ax.axvline(z0_low, color="seagreen", ls=":", lw=1.1)
    ax.axvline(z0_nB, color="black", ls=":", lw=1.0)
    ax.scatter([z0_low, z0_nB], [phi, phi_bar], color=["seagreen", "darkred"], edgecolor="black", s=32, zorder=5)

    ax.annotate(
        r"$\overline{\phi} = \Delta U(n_B)$",
        xy=(z0_nB, phi_bar),
        xytext=(z0_nB - 0.10 * (z0_nB - z_star), phi_bar * 1.10),
        fontsize=FONTSIZES["annotation"] - 0.5,
        ha="right",
        arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8),
    )

    ax.set_xticks([z_star, z0_low, z0_nB])
    ax.set_xticklabels([r"$z^*$", r"$\underline{z}_0(\phi)$", r"$z_0(n_B)$"], fontsize=FONTSIZES["tick"])
    ax.set_yticks([0.0, phi, phi_bar])
    ax.set_yticklabels([r"$0$", r"$\phi$", r"$\overline{\phi}$"], fontsize=FONTSIZES["tick"])
    ax.set_xlim(z0_min, z_max)
    ax.set_ylim(-0.08 * phi_bar, 1.32 * phi_bar)
    ax.set_xlabel(r"Pre-kink income $z_0(n)$", fontsize=FONTSIZES["label"])
    ax.set_ylabel(r"Utility gain $\Delta U$", fontsize=FONTSIZES["label"])
    # ax.set_title(r"Three regimes under optimization frictions ($\phi > 0$)", fontsize=FONTSIZES["title"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.8),
        ncol=2,
        frameon=False,
        fontsize=FONTSIZES["legend"] - 1.0,
    )
    ax.tick_params(labelsize=FONTSIZES["tick"])

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    return fig, ax


def plot_nf_mechanism(fig, axes):
    # 1. Latent space u ~ N(0, 1)
    u = np.linspace(-3.2, 3.2, 500)
    p_u = norm.pdf(u)

    # Monotonic diffeomorphism z = T_inv(u) (smooth log-normal-like warping)
    z = 0.55 + 0.45 * np.exp(0.42 * u)
    dz_du = 0.45 * 0.42 * np.exp(0.42 * u)
    g_z = p_u / dz_du  # Change-of-variables formula: p_U(T(z)) * |dT/dz|

    z_L, z_R = 0.86, 1.16
    u_L = np.log((z_L - 0.55) / 0.45) / 0.42
    u_R = np.log((z_R - 0.55) / 0.45) / 0.42
    cens_u = (u >= u_L) & (u <= u_R)
    cens_z = (z >= z_L) & (z <= z_R)

    # Panel 1: Base distribution
    ax = axes[0]
    ax.plot(u, p_u, color="dimgray", lw=1.8)
    ax.fill_between(u, 0, p_u, color="lightgray", alpha=0.45)
    ax.fill_between(u[cens_u], 0, p_u[cens_u], color="seagreen", alpha=0.25)
    ax.set_title(r"1. Base density $p_U(u) = \mathcal{N}(0,1)$", fontsize=10)
    ax.set_xlabel(r"Latent variable $u$", fontsize=9.5)
    ax.set_ylabel("Density", fontsize=9.5)
    ax.set_yticks([])

    # Panel 2: Monotonic spline diffeomorphism
    ax = axes[1]
    ax.axhspan(z_L, z_R, color="0.90", alpha=0.6, label=r"Censored $[z_L, z_R)$")
    ax.plot(u[~cens_u], z[~cens_u], color="darkblue", lw=2.0, label="Fitted on exterior")
    ax.plot(u[cens_u], z[cens_u], color="seagreen", ls="--", lw=2.2, label="Smooth interpolation")
    ax.set_title(r"2. Monotonic map $z = T_\theta^{-1}(u)$", fontsize=10)
    ax.set_xlabel(r"Latent variable $u$", fontsize=9.5)
    ax.set_ylabel(r"Income $z$", fontsize=9.5)
    ax.set_yticks([z_L, z_R])
    ax.set_yticklabels([r"$z_L$", r"$z_R$"], fontsize=9)
    ax.legend(loc="upper left", fontsize=7.8, frameon=True, framealpha=0.9)

    # Panel 3: Induced counterfactual density
    ax = axes[2]
    ax.axvspan(z_L, z_R, color="0.90", alpha=0.6)
    ax.plot(z[~cens_z], g_z[~cens_z], color="darkblue", lw=1.8, label=r"Exterior fit \\($z \notin [z_L, z_R)$)")
    ax.plot(z[cens_z], g_z[cens_z], color="seagreen", ls="--", lw=2.1, label=r"Counterfactual $g_\theta(z)$")
    ax.fill_between(z[cens_z], 0, g_z[cens_z], color="seagreen", alpha=0.22)
    ax.set_title(r"3. Counterfactual $g_\theta(z) = p_U(T_\theta(z))|T'_\theta(z)|$", fontsize=10)
    ax.set_xlabel(r"Taxable income $z$", fontsize=9.5)
    ax.set_ylabel("Density", fontsize=9.5)
    ax.set_xticks([z_L, z_R])
    ax.set_xticklabels([r"$z_L$", r"$z_R$"], fontsize=9)
    ax.set_yticks([])
    ax.set_xlim(0.60, 1.55)
    ax.legend(loc="lower center", fontsize=7.8, frameon=True, framealpha=0.9)

    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig, axes