import numpy as np

from . import FONTSIZES, _cmap


def plot_kink_vs_notch(bins, kink_counts, notch_counts, fig, axes, z_star=10.0, notch_size=1.5, tau1=0.04):
    """Kink vs. notch in the budget set and in the resulting density.

    Two rows: top row for the kink, bottom row for the notch. Left column shows
    the budget set c(z) with slopes 1 and 1 - tau1 (kink) or a vertical drop of
    notch_size (notch); right column shows the simulated density of declared
    income with the excess mass and the dominated region marked.

    Parameters
    ----------
    bins : array_like
        Histogram edges in million CLP, centred so that ``z_star`` is the centre
        of a bin (e.g. ``np.arange(-0.125, 20.001, 0.25)``).
    kink_counts, notch_counts : array_like
        Bin counts for the kink and the notch simulation, length ``len(bins)-1``.
    fig : matplotlib.figure.Figure
    axes : 2x2 array of matplotlib.axes.Axes
    z_star : float, optional
        Threshold in million CLP. Default 10.0.
    notch_size : float, optional
        Vertical drop in net income at the notch, million CLP. Default 1.5.
    tau1 : float, optional
        Marginal rate above the kink. Default 0.04.

    Returns
    -------
    fig, axes
    """
    bins = np.asarray(bins)
    kink_counts = np.asarray(kink_counts)
    notch_counts = np.asarray(notch_counts)

    width = float(np.diff(bins)[0])
    peak_mask = np.abs(bins[:-1] + width / 2 - z_star) < width / 2
    kink_peak = kink_counts[peak_mask].sum()
    notch_peak = notch_counts[peak_mask].sum()

    x = np.linspace(8, 13, 300)
    c_kink = np.where(x <= z_star, x, z_star + (1 - tau1) * (x - z_star))

    axes[0, 0].plot(x, c_kink, color="black", lw=1.8)
    axes[0, 0].axvline(z_star, color="black", ls=":", lw=1.0)
    axes[0, 0].text(8.3, 9.05, "slope 1", fontsize=FONTSIZES["annotation"])
    axes[0, 0].text(11.1, 10.75, f"slope {1 - tau1:.2f}", fontsize=FONTSIZES["annotation"])
    axes[0, 0].text(10.1, 9.55, "continuous", fontsize=FONTSIZES["annotation"])
    axes[0, 0].set_xlim(8, 13)
    axes[0, 0].set_ylim(8, 12.5)
    axes[0, 0].set_ylabel("Net income, c", fontsize=FONTSIZES["label"])
    axes[0, 0].set_title("Kink: budget set", fontsize=FONTSIZES["title"])

    axes[0, 1].step(bins[:-1], kink_counts, where="post", color="darkblue", lw=1.2)
    axes[0, 1].fill_between(bins[:-1], kink_counts, step="post", color="darkblue", alpha=0.08, lw=0)
    axes[0, 1].axvline(z_star, color="black", ls=":", lw=1.0)
    axes[0, 1].annotate(
        "excess mass",
        xy=(10.0, kink_peak),
        xytext=(11.6, kink_peak * 1.18),
        fontsize=FONTSIZES["annotation"],
        arrowprops=dict(arrowstyle="->", color="0.3"),
    )
    axes[0, 1].set_xlim(7, 15)
    axes[0, 1].set_ylim(0, kink_peak * 1.35)
    axes[0, 1].set_ylabel("Taxpayers", fontsize=FONTSIZES["label"])
    axes[0, 1].set_title("Kink: bunching just below the threshold", fontsize=FONTSIZES["title"])

    axes[1, 0].plot([8, z_star], [8, z_star], color="black", lw=1.8)
    axes[1, 0].plot(
        [z_star, z_star], [z_star, z_star - notch_size], color="black", lw=1.8
    )
    axes[1, 0].plot(
        [z_star, 13], [z_star - notch_size, 13 - notch_size], color="black", lw=1.8
    )
    axes[1, 0].axvspan(z_star, z_star + notch_size, color="darkred", alpha=0.15)
    axes[1, 0].axvline(z_star, color="black", ls=":", lw=1.0)
    axes[1, 0].annotate(
        "jump in level",
        xy=(z_star, z_star - notch_size / 2),
        xytext=(10.9, 7.9),
        fontsize=FONTSIZES["annotation"],
        arrowprops=dict(arrowstyle="->", color="0.3"),
    )
    axes[1, 0].text(10.75, 11.5, "dominated region", fontsize=FONTSIZES["annotation"], ha="center")
    axes[1, 0].set_xlim(8, 13)
    axes[1, 0].set_ylim(7.5, 12.2)
    axes[1, 0].set_xlabel("Taxable income, z (million CLP)", fontsize=FONTSIZES["label"])
    axes[1, 0].set_ylabel("Net income, c", fontsize=FONTSIZES["label"])
    axes[1, 0].set_title("Notch: budget set", fontsize=FONTSIZES["title"])

    axes[1, 1].step(bins[:-1], notch_counts, where="post", color="darkblue", lw=1.2)
    axes[1, 1].fill_between(bins[:-1], notch_counts, step="post", color="darkblue", alpha=0.08, lw=0)
    axes[1, 1].axvline(z_star, color="black", ls=":", lw=1.0)
    axes[1, 1].annotate(
        "bunching",
        xy=(10.0, notch_peak),
        xytext=(8.0, notch_peak * 0.85),
        fontsize=FONTSIZES["annotation"],
        arrowprops=dict(arrowstyle="->", color="0.3"),
    )
    axes[1, 1].annotate(
        "missing mass",
        xy=(10.75, 0.02 * notch_peak),
        xytext=(11.4, notch_peak * 0.35),
        fontsize=FONTSIZES["annotation"],
        arrowprops=dict(arrowstyle="->", color="0.3"),
    )
    axes[1, 1].set_xlim(7, 15)
    axes[1, 1].set_ylim(0, notch_peak * 1.2)
    axes[1, 1].set_xlabel("Taxable income, z (million CLP)", fontsize=FONTSIZES["label"])
    axes[1, 1].set_ylabel("Taxpayers", fontsize=FONTSIZES["label"])
    axes[1, 1].set_title("Notch: missing mass and bunching", fontsize=FONTSIZES["title"])

    for ax in axes.flat:
        ax.tick_params(labelsize=FONTSIZES["tick"])

    fig.tight_layout()
    return fig, axes


def plot_disutility_curvature(z, fig, axes, n=1.0, epsilons=(0.2, 0.5, 1.0, 2.0)):
    """Curvature of isoelastic disutility and marginal cost across ETI values.

    Two panels: left panel shows total disutility v(z; n, eps) and right panel
    shows the marginal disutility cost v_z(z; n, eps) = (z / n)^(1 / eps) for
    different values of the elasticity of taxable income (eps).

    Parameters
    ----------
    z : array_like
        Grid of declared taxable income values.
    fig : matplotlib.figure.Figure
    axes : 1x2 array of matplotlib.axes.Axes
    n : float, optional
        Normalized ability parameter (benchmark income at tau = 0). Default 1.0.
    epsilons : sequence of float, optional
        Elasticity values to compare. Default (0.2, 0.5, 1.0, 2.0).

    Returns
    -------
    fig, axes
    """
    z = np.asarray(z)
    dark_colors = ["darkblue", "darkgreen", "darkred", "darkorange", "darkviolet", "darkcyan"]

    for idx, eps in enumerate(epsilons):
        color = dark_colors[idx % len(dark_colors)]
        disutility = (n / (1.0 + 1.0 / eps)) * (z / n) ** (1.0 + 1.0 / eps)
        marginal_cost = (z / n) ** (1.0 / eps)

        axes[0].plot(z, disutility, color=color, lw=1.8, label=rf"$\epsilon = {eps}$")
        axes[1].plot(z, marginal_cost, color=color, lw=1.8)

    # Panel 0: Total disutility
    axes[0].axvline(n, color="black", ls=":", lw=1.0)
    axes[0].annotate(
        r"ability $z = n$",
        xy=(n, 0.8),
        xytext=(n - 0.52, 1.22),
        fontsize=FONTSIZES["annotation"],
        arrowprops=dict(arrowstyle="->", color="0.3"),
    )
    axes[0].set_xlim(z.min(), z.max())
    axes[0].set_ylim(0, 1.5)
    axes[0].set_xlabel(r"Taxable income, $z$", fontsize=FONTSIZES["label"])
    axes[0].set_ylabel(r"Disutility, $v(z; n, \epsilon)$", fontsize=FONTSIZES["label"])
    axes[0].set_title(r"Total disutility curvature", fontsize=FONTSIZES["title"])

    # Panel 1: Marginal cost (FOC)
    axes[1].axvline(n, color="black", ls=":", lw=1.0)
    axes[1].axhline(1.0, color="0.4", ls="--", lw=1.0)
    axes[1].text(
        z.min() + 0.04,
        1.06,
        r"net return $1 - \tau = 1$",
        fontsize=FONTSIZES["annotation"],
    )
    axes[1].set_xlim(z.min(), z.max())
    axes[1].set_ylim(0, 2.5)
    axes[1].set_xlabel(r"Taxable income, $z$", fontsize=FONTSIZES["label"])
    axes[1].set_ylabel(r"Marginal cost, $(z/n)^{1/\epsilon}$", fontsize=FONTSIZES["label"])
    axes[1].set_title(r"Marginal cost of generating income", fontsize=FONTSIZES["title"])

    for ax in axes.flat:
        ax.tick_params(labelsize=FONTSIZES["tick"])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        fontsize=FONTSIZES["legend"],
        frameon=True,
    )

    fig.tight_layout()
    return fig, axes

def plot_directional_utility_kink(z, trajectories, fig, axes, z_star=1.0):
    """Objective utility U(z; n) with a convex kink at z_star across ability types.

    Left panel shows total objective utility U(z; n) for precomputed ability
    trajectories (low ability, buncher, high ability) with their optimal choices
    marked. Right panel shows the corresponding directional marginal utilities
    U'(z; n), highlighting how the buncher jumps across zero right at z_star.

    Parameters
    ----------
    z : array_like
        Grid of declared taxable income values.
    trajectories : list of dict
        Precomputed curves for each ability type. Each dict must contain:
        ``label``, ``color``, ``utility``, ``marginal_u``, ``z_opt``, ``u_opt``,
        ``u_prime_left``, and ``u_prime_right``.
    fig : matplotlib.figure.Figure
    axes : 1x2 array of matplotlib.axes.Axes
    z_star : float, optional
        Kink threshold. Default 1.0.

    Returns
    -------
    fig, axes
    """
    z = np.asarray(z)
    left_mask = z <= z_star
    right_mask = z > z_star

    # White semi-transparent text box style
    text_box = dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.5)

    for item in trajectories:
        color = item["color"]
        axes[0].plot(z, item["utility"], color=color, lw=1.8, label=item["label"])
        axes[0].scatter([item["z_opt"]], [item["u_opt"]], color=color, zorder=5, s=36)

        axes[1].plot(z[left_mask], item["marginal_u"][left_mask], color=color, lw=1.8)
        axes[1].plot(z[right_mask], item["marginal_u"][right_mask], color=color, lw=1.8)
        axes[1].plot(
            [z_star, z_star],
            [item["u_prime_left"], item["u_prime_right"]],
            color=color,
            ls="--",
            lw=1.0,
            alpha=0.6,
        )
        axes[1].scatter(
            [z_star, z_star],
            [item["u_prime_left"], item["u_prime_right"]],
            color=color,
            s=22,
            zorder=5,
        )

    buncher = trajectories[1]

    all_u = np.concatenate([item["utility"] for item in trajectories])
    u_max = float(np.max(all_u))

    # Panel 0: Objective utility U(z; n)
    axes[0].axvline(z_star, color="black", ls=":", lw=1.0)
    axes[0].annotate(
        "kink optimum\n" + r"at $z^*$",
        xy=(buncher["z_opt"], buncher["u_opt"]),
        xytext=(z_star - 0.58, buncher["u_opt"] + 0.08 * u_max),
        fontsize=FONTSIZES["annotation"],
        ha="center",
        bbox=text_box,
        zorder=6,
        arrowprops=dict(arrowstyle="->", color="0.3"),
    )
    axes[0].set_xlim(z.min(), z.max())
    axes[0].set_ylim(0.0, u_max * 1.15)
    axes[0].set_xlabel(r"Taxable income, $z$", fontsize=FONTSIZES["label"])
    axes[0].set_ylabel(r"Objective utility, $U(z; n)$", fontsize=FONTSIZES["label"])
    axes[0].set_title(r"Objective utility $U(z; n)$", fontsize=FONTSIZES["title"])

    # Panel 1: Directional marginal utility U'(z; n)
    axes[1].axvline(z_star, color="black", ls=":", lw=1.0)
    axes[1].axhline(0.0, color="0.3", ls="--", lw=1.0)
    axes[1].annotate(
        r"$U'_-(z^*) > 0$",
        xy=(z_star, buncher["u_prime_left"]),
        xytext=(z_star - 0.58, buncher["u_prime_left"] + 0.14),
        fontsize=FONTSIZES["annotation"],
        bbox=text_box,
        zorder=6,
        arrowprops=dict(arrowstyle="->", color="0.3"),
    )
    axes[1].annotate(
        r"$U'_+(z^*) < 0$",
        xy=(z_star, buncher["u_prime_right"]),
        xytext=(z_star + 0.12, buncher["u_prime_right"] - 0.14),
        fontsize=FONTSIZES["annotation"],
        bbox=text_box,
        zorder=6,
        arrowprops=dict(arrowstyle="->", color="0.3"),
    )
    axes[1].set_xlim(z.min(), z.max())
    axes[1].set_ylim(-0.55, 0.55)
    axes[1].set_xlabel(r"Taxable income, $z$", fontsize=FONTSIZES["label"])
    axes[1].set_ylabel(r"Marginal utility, $U'(z; n)$", fontsize=FONTSIZES["label"])
    axes[1].set_title(r"Directional marginal utility $U'(z; n)$", fontsize=FONTSIZES["title"])

    for ax in axes.flat:
        ax.tick_params(labelsize=FONTSIZES["tick"])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        fontsize=FONTSIZES["legend"],
        frameon=False,
    )

    fig.tight_layout()
    return fig, axes


import numpy as np

from . import FONTSIZES


def plot_excess_mass_identification(z, g_cf, h_obs, z_star, delta_z, fig, ax):
    """Single-panel visual link between excess mass B and counterfactual width Delta z*.

    Plots the observed density h_obs(z) and the smooth counterfactual g_cf(z)
    on a single axes, showing how the excess mass B piled up above g_cf at z_star
    equals the area under g_cf over [z_star, z_star + delta_z], approximated by
    the rectangle g_cf(z_star) * delta_z.

    Parameters
    ----------
    z : array_like
        Grid of declared taxable income values.
    g_cf : array_like
        Counterfactual density evaluated on ``z``.
    h_obs : array_like
        Observed density (with bunching spike around ``z_star``) evaluated on ``z``.
    z_star : float
        Kink threshold.
    delta_z : float
        Counterfactual response width of the marginal buncher, z_0(n_B) - z_star.
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Returns
    -------
    fig, ax
    """
    z = np.asarray(z)
    g_cf = np.asarray(g_cf)
    h_obs = np.asarray(h_obs)

    text_box = dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5)
    g_at_kink = float(np.interp(z_star, z, g_cf))
    y_max = float(np.max(h_obs)) * 1.15

    # 1. Observed and counterfactual densities
    ax.plot(z, h_obs, color="darkblue", lw=1.8, label=r"Observed density $h(z)$")
    ax.plot(z, g_cf, color="darkred", ls="--", lw=1.8, label=r"Counterfactual $g(z)$")

    # 2. Observed excess mass B (above the counterfactual)
    bunch_mask = h_obs > g_cf
    ax.fill_between(
        z[bunch_mask],
        g_cf[bunch_mask],
        h_obs[bunch_mask],
        color="darkblue",
        alpha=0.22,
        label=r"Observed excess mass $B$",
    )

    # 3. Exact counterfactual area and rectangle g(z*) * Delta z* (below the counterfactual)
    band_mask = (z >= z_star) & (z <= z_star + delta_z)
    ax.fill_between(
        z[band_mask],
        0.0,
        g_cf[band_mask],
        color="darkgreen",
        alpha=0.14,
        label=r"Counterfactual area $\approx g(z^*) \cdot \Delta z^*$",
    )
    ax.plot(
        [z_star, z_star + delta_z, z_star + delta_z],
        [g_at_kink, g_at_kink, 0.0],
        color="darkgreen",
        lw=1.8,
    )

    # 4. Líneas guía y anotaciones
    ax.axvline(z_star, color="black", ls=":", lw=1.0)
    ax.axvline(z_star + delta_z, color="darkgreen", ls=":", lw=1.0)

    ax.annotate(
        "observed spike\n" + r"mass $B$",
        xy=(z_star, float(np.max(h_obs)) * 0.85),
        xytext=(z_star - 0.38, float(np.max(h_obs)) * 0.82),
        fontsize=FONTSIZES["annotation"],
        ha="center",
        bbox=text_box,
        zorder=6,
        arrowprops=dict(arrowstyle="->", color="0.3"),
    )
    ax.annotate(
        r"height $g(z^*)$",
        xy=(z_star, g_at_kink),
        xytext=(z_star - 0.38, g_at_kink * 1.15),
        fontsize=FONTSIZES["annotation"],
        ha="center",
        bbox=text_box,
        zorder=6,
        arrowprops=dict(arrowstyle="->", color="0.3"),
    )

    # Horizontal arrow showing the width Delta z* = B / g(z*) inside the rectangle
    ax.annotate(
        "",
        xy=(z_star, g_at_kink * 0.30),
        xytext=(z_star + delta_z, g_at_kink * 0.30),
        arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=1.4),
    )
    ax.text(
        z_star + 0.5 * delta_z,
        g_at_kink * 0.42,
        r"$\Delta z^* \approx \frac{B}{g(z^*)}$",
        fontsize=FONTSIZES["annotation"],
        color="darkgreen",
        ha="center",
        bbox=text_box,
        zorder=6,
    )

    ax.set_xlim(z.min(), z.max())
    ax.set_ylim(0.0, y_max)
    ax.set_xlabel(r"Taxable income, $z$", fontsize=FONTSIZES["label"])
    ax.set_ylabel(r"Density (taxpayers / peso)", fontsize=FONTSIZES["label"])
    # ax.set_title(r"Identification of $\Delta z^*$ from observed excess mass $B$", fontsize=FONTSIZES["title"])
    ax.tick_params(labelsize=FONTSIZES["tick"])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=FONTSIZES["legend"],
        frameon=False,
    )

    fig.tight_layout()
    return fig, ax