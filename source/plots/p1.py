import matplotlib.ticker as mtick
import numpy as np

from . import save_fig, FIGURES_DIR, FONTSIZES  # noqa: F401
from numpy.polynomial import polynomial as P

def plot_cutoff_shift_grid(panels_data, fig, axes, xlim=(2e6, 14e6)):
    """1x3 grid showing the nominal upward shift of the bunching spike across years.

    Parameters
    ----------
    panels_data : list of dict
        Each dict corresponds to one year and must contain:
        ``year``, ``z``, ``c``, ``peak_z``, and optionally ``threshold_z``.
    fig : matplotlib.figure.Figure
    axes : 1x3 array of matplotlib.axes.Axes
    xlim : tuple of float, optional
        Common horizontal axis limits in CLP. Default ``(2e6, 14e6)``.

    Returns
    -------
    fig, axes
    """
    text_box = dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5)

    for ax, item in zip(axes.flat, panels_data):
        z = np.asarray(item["z"])
        c = np.asarray(item["c"])
        peak_z = item["peak_z"]
        thr_z = item.get("threshold_z")

        mask = (z >= xlim[0]) & (z <= xlim[1])
        ax.step(z[mask], c[mask], where="post", color="darkblue", lw=1.2)
        ax.fill_between(z[mask], c[mask], step="post", color="darkblue", alpha=0.08, lw=0)

        # Empirical peak
        ax.axvline(peak_z, color="crimson", ls="--", lw=1.1, label="Empirical peak")
        if thr_z is not None:
            ax.axvline(thr_z, color="black", ls=":", lw=1.1, label="Statutory threshold")

        y_max = float(c[mask].max()) * 1.15
        ax.annotate(
            f"Peak: {peak_z/1e6:.1f}M",
            xy=(peak_z, float(c[mask].max()) * 0.92),
            xytext=(peak_z + 0.6e6, float(c[mask].max()) * 0.90),
            fontsize=FONTSIZES["annotation"],
            color="crimson",
            bbox=text_box,
            zorder=6,
            arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8),
        )

        ax.set_xlim(*xlim)
        ax.set_ylim(0, y_max)
        ax.set_title(f"{item['year']}", fontsize=FONTSIZES["title"])
        ax.set_xlabel("Taxable income (CLP)", fontsize=FONTSIZES["label"])
        ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
        ax.tick_params(labelsize=FONTSIZES["tick"])

    axes[0].set_ylabel("NREG (taxpayers)", fontsize=FONTSIZES["label"])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        fontsize=FONTSIZES["legend"],
        frameon=False,
    )

    fig.tight_layout()
    return fig, axes

def counterfactual(
    z, c, z_l, z_r, z_L, z_R, degree=4, chetty=True, max_iter=1000, tol=1e-8
) -> np.ndarray:
    r"""Counterfactual counts over ``[z_l, z_r]`` with optional right-tail reweighting.

    Fits a degree-``p`` polynomial in rescaled income ``u_j = z_j / z_r`` outside
    the excluded bunching region ``[z_L, z_R)``:

    .. math::

        \tilde{c}_j = \sum_{i=0}^{p} \beta_i u_j^i + \varepsilon_j,
        \qquad
        \hat{c}_j = \sum_{i=0}^{p} \beta_i u_j^i.

    When ``chetty=True``, counts in the upper tail ``[z_R, z_r]`` are
    iteratively reweighted following Chetty et al. (2011) so that the total
    counterfactual mass over ``[z_l, z_r]`` matches the observed mass.

    Parameters
    ----------
    z : array_like
        Bin lower edges ``z_j`` in CLP.
    c : array_like
        Observed counts ``c_j`` (taxpayers per bin), aligned with ``z``.
    z_l, z_r : float
        Left and right bounds of the fitting window, in CLP.
    z_L, z_R : float
        Lower and upper bounds of the excluded bunching region ``[z_L, z_R)``,
        in CLP. Because ``z_j`` is a bin lower edge, ``z_R`` is exclusive so
        that bins spanning ``[z_L, z_R)`` have total width ``z_R - z_L``.
    degree : int, optional
        Polynomial degree ``p``. Default is 4.
    chetty : bool, optional
        If ``True`` (default), iteratively reweight the right tail ``[z_R, z_r]``
        to absorb the displaced bunching mass. If ``False``, return the naive
        one-step polynomial fit without right-tail adjustment.
    max_iter : int, optional
        Maximum number of reweighting rounds when ``chetty=True``. Default is 1000.
    tol : float, optional
        Convergence tolerance on ``max|Delta c|`` in the right tail
        ``[z_R, z_r]``. Default is ``1e-8``.

    Returns
    -------
    c_hat : numpy.ndarray
        Fitted counterfactual counts ``\hat{c}_j``, same shape as ``z``.
        Entries outside ``[z_l, z_r]`` are ``NaN``.
    """
    z = np.asarray(z, np.float64)
    c = np.asarray(c, np.float64)
    u = z / z_r
    X = P.polyvander(u, degree)

    window = (z >= z_l) & (z <= z_r)
    excluded = (z >= z_L) & (z < z_R)
    right = (z >= z_R) & (z <= z_r)
    fit = window & ~excluded

    c_adj = c.copy()
    if chetty:
        right_mass = float(np.sum(c[right]))
        for _ in range(max_iter):
            beta = np.linalg.lstsq(X[fit], c_adj[fit], rcond=None)[0]
            B_hat = float(np.sum(c[excluded] - P.polyval(u[excluded], beta)))

            c_adj_new = c.copy()
            if right_mass > 0.0:
                c_adj_new[right] = c[right] * (1.0 + B_hat / right_mass)

            delta = (
                float(np.max(np.abs(c_adj_new[right] - c_adj[right])))
                if right.any()
                else 0.0
            )
            c_adj = c_adj_new
            if delta < tol:
                break

    beta = np.linalg.lstsq(X[fit], c_adj[fit], rcond=None)[0]
    c_hat = np.full(z.shape, np.nan)
    c_hat[window] = P.polyval(u[window], beta)
    return c_hat

def plot_observed_distribution(df, fig, axes, year=2023):
    """Plot the observed NREG distribution for a given year.

    Parameters
    ----------
    df : pandas.DataFrame
        Bunch-count table with columns ``year``, ``group_0`` (lower bin edge,
        CLP) and ``NREG`` (taxpayers per bin).
    fig : matplotlib.figure.Figure
        Figure to draw on.
    axes : sequence of matplotlib.axes.Axes
        Two axes: ``axes[0]`` for the full distribution over 0-30M CLP and
        ``axes[1]`` for the 6M-14M zoom, with the excluded 8.5M-10.5M region
        shaded.
    year : int, optional
        Year to plot. Default is 2023.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The input figure.
    axes : sequence of matplotlib.axes.Axes
        The input axes.
    """
    df_y = df[df['year'] == year].sort_values('group_0')

    axes[0].step(df_y['group_0'], df_y['NREG'], where='post', color='darkblue', lw=1.2)
    axes[0].fill_between(df_y['group_0'], df_y['NREG'], step='post', color='darkblue', alpha=0.08, lw=0)
    axes[0].set_xlim(0, 30000000)
    axes[0].set_xlabel('Taxable income (CLP)', fontsize=FONTSIZES["label"])
    axes[0].set_ylabel('NREG (taxpayers)', fontsize=FONTSIZES["label"])
    axes[0].set_title(f'Observed Distribution, {year}', fontsize=FONTSIZES["title"])
    axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))

    df_z = df_y[df_y['group_0'].between(6e6, 14e6)]
    axes[1].step(df_z['group_0'], df_z['NREG'], where='post', color='darkblue', lw=1.2)
    axes[1].fill_between(df_z['group_0'], df_z['NREG'], step='post', color='darkblue', alpha=0.08, lw=0)
    axes[1].axvspan(8.5e6, 10.5e6, color='darkred', alpha=0.15, label='excluded 8.5M-10.5M')
    axes[1].set_xlim(6e6, 14e6)
    axes[1].set_ylim(0, df_z['NREG'].max() * 1.08)
    axes[1].set_xlabel('Taxable income (CLP)', fontsize=FONTSIZES["label"])
    axes[1].set_title(f'Zoom, {year}', fontsize=FONTSIZES["title"])
    axes[1].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.45, 1.4), frameon=False, borderaxespad=0.0,
                   fontsize=FONTSIZES["legend"])
    for ax in axes:
        ax.tick_params(labelsize=FONTSIZES["tick"])
    fig.tight_layout()

    return fig, axes


def plot_counterfactual(z, c, c_hat, ax, z_star=None, z_L=None, z_R=None, xlim=(6e6, 14e6),
                        xlabel="Taxable income (CLP)", tick_div=1e6, tick_fmt="{:.1f}M"):
    """Observed counts vs. estimated counterfactual density.

    Bars are the observed ``c_j``; the red line is the fitted counterfactual
    ``c_hat_j`` (``NaN`` outside the estimation window); the shaded band marks
    the excluded region ``[z_L, z_R]``; the dashed vertical line marks the kink
    ``z_star``.

    Parameters
    ----------
    z : array_like
        Bin lower edges ``z_j`` in CLP.
    c : array_like
        Observed counts ``c_j`` (taxpayers per bin), aligned with ``z``.
    c_hat : array_like
        Counterfactual counts ``\\hat{c}_j``, same shape as ``z``. ``NaN``
        entries are not drawn.
    ax : matplotlib.axes.Axes
        Axis to draw on.
    z_star : float, optional
        Kink point ``z^*`` in CLP. If ``None``, no vertical line is drawn.
    z_L, z_R : float, optional
        Bounds of the excluded region in CLP. If either is ``None``, no shaded
        band is drawn.
    xlim : tuple of float, optional
        Horizontal axis limits in CLP. Default is ``(6e6, 14e6)``.
    xlabel : str, optional
        Horizontal axis label. Default is ``"Taxable income (CLP)"``.
    tick_div : float, optional
        Divisor applied to the tick values. Default is ``1e6``.
    tick_fmt : str, optional
        Format string applied to the scaled tick values. Default is
        ``"{:.1f}M"``.

    Returns
    -------
    matplotlib.axes.Axes
        The input axis.
    """
    z = np.asarray(z)
    c = np.asarray(c)
    c_hat = np.asarray(c_hat)

    ax.step(z, c, where='post', color='darkblue', lw=1.2, label='Observed')
    ax.fill_between(z, c, step='post', color='darkblue', alpha=0.08, lw=0)
    ax.plot(z, c_hat, color='crimson', lw=1.8, label='Counterfactual')
    if z_L is not None and z_R is not None:
        ax.axvspan(z_L, z_R, color='darkred', alpha=0.15, label='Excluded region')
    if z_star is not None:
        ax.axvline(z_star, color='black', ls='--', lw=1.0, label=r'$z^*$')
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=FONTSIZES["label"])
    ax.set_ylabel('NREG (taxpayers)', fontsize=FONTSIZES["label"])
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: tick_fmt.format(x / tick_div)))
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False, borderaxespad=0.0,
              fontsize=FONTSIZES["legend"])
    ax.tick_params(labelsize=FONTSIZES["tick"])
    return ax


def plot_pooled_bunching(z, c, c_hat, ax, z_star=None, z_L=None, z_R=None, xlim=(100, 320)):
    """Bunching pooled across years, with income expressed in annual UTM.

    Same visual language as ``plot_counterfactual`` but on the UTM scale:
    bars are pooled counts, the red line is the counterfactual, the shaded band
    is the excluded region and the dashed line marks the kink.
    """
    return plot_counterfactual(
        z, c, c_hat, ax, z_star=z_star, z_L=z_L, z_R=z_R, xlim=xlim,
        xlabel="Taxable income (multiples of the exempt threshold)", tick_div=1.0, tick_fmt="{:.1f}",
    )


def plot_sensitivity(sens, axes, reference=0.2799):
    dark = ["darkblue", "#c61f1f", "darkgreen", "#773ad4", "#f0a518"]
    marks = ["o", "s", "^", "D", "v", "P"]
    order = list(dict.fromkeys(sens["window"]))
    for i, k in enumerate(sorted(sens["bin_k"].unique())):
        sub = sens[sens.bin_k == k].set_index("window").reindex(order)
        axes[0].plot(order, sub["epsilon"], marker=marks[i % len(marks)],
                     color=dark[i % len(dark)], label=f"{k}k", lw=1.0)
    axes[0].axhline(reference, color="#444444", ls="--", lw=1.0, label=f"paper ({reference})")
    axes[0].set_xlabel("Estimation window (CLP)", fontsize=FONTSIZES["label"])
    axes[0].set_ylabel(r"$\hat{\epsilon}$", fontsize=FONTSIZES["label"])
    axes[0].set_title("Elasticity vs. window", fontsize=FONTSIZES["title"])

    bins = sorted(sens["bin_k"].unique())
    for i, w in enumerate(order):
        sub = sens[sens.window == w].set_index("bin_k").reindex(bins)
        axes[1].plot(bins, sub["epsilon"], marker=marks[i % len(marks)],
                     color=dark[i % len(dark)], label=w)
    axes[1].axhline(reference, color="#444444", ls="--", lw=1.0)
    axes[1].set_xlabel("Bin width (thousand CLP)", fontsize=FONTSIZES["label"])
    axes[1].set_title("Elasticity vs. bin width", fontsize=FONTSIZES["title"])
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
                   borderaxespad=0.0, fontsize=FONTSIZES["legend"])
    for ax in axes:
        ax.tick_params(labelsize=FONTSIZES["tick"], rotation=30)
    return axes


def plot_kink_neighbourhood(z, c, ax, z_star, n_bins=5, xlim=None):
    """Zoom on the kink: observed counts around ``z_star`` with the mean count of
    the excess block and of the deficit block drawn as horizontal segments, 
    along with the statutory threshold line and legend.

    Parameters
    ----------
    z : array_like
        Bin lower edges ``z_j`` in CLP.
    c : array_like
        Observed counts ``c_j`` (taxpayers per bin), aligned with ``z``.
    ax : matplotlib.axes.Axes
        Axis to draw on.
    z_star : float
        Statutory kink point ``z^* = K`` in CLP (e.g., 9,907,434).
    n_bins : int, optional
        Number of bins in each block. Default is 5.
    xlim : tuple of float, optional
        Horizontal axis limits in CLP. Default is the block span.

    Returns
    -------
    ax : matplotlib.axes.Axes, mean_below, mean_above, r_below, r_above
    """
    z = np.asarray(z)
    c = np.asarray(c)

    # Bin that contains z_star: z[j0] <= z_star < z[j0 + 1]
    j0 = int(np.searchsorted(z, z_star, side="right") - 1)
    j0 = int(np.clip(j0, n_bins, len(z) - n_bins - 2))

    below = slice(j0 - n_bins + 1, j0 + 1)
    above = slice(j0 + 1, j0 + 1 + n_bins)

    mean_below = float(c[below].mean())
    mean_above = float(c[above].mean())
    r_below = (float(z[below.start]), float(z[below.stop - 1]))
    r_above = (float(z[above.start]), float(z[above.stop - 1]))

    win = slice(j0 - n_bins, j0 + n_bins + 2)
    ax.step(z, c, where="post", color="darkblue", lw=1.2, label="Observed counts")
    ax.fill_between(z[win], c[win], step="post", color="darkblue", alpha=0.08, lw=0)
    ax.hlines(
        mean_below,
        z[below.start],
        z[j0 + 1],
        color="crimson",
        ls="--",
        lw=1.2,
        label=f"Excess mean ({mean_below:,.0f})",
    )
    ax.hlines(
        mean_above,
        z[j0 + 1],
        z[above.stop],
        color="grey",
        ls="--",
        lw=1.2,
        label=f"Deficit mean ({mean_above:,.0f})",
    )
    ax.axvline(
        z_star,
        color="black",
        ls=":",
        lw=1.4,
        label=rf"Statutory threshold ($K$ = {z_star/1e6:.2f}M)",
    )

    ax.set_xlim(*(xlim if xlim is not None else (z[j0 - n_bins], z[j0 + n_bins + 1])))
    ax.set_xlabel("Taxable income (CLP)", fontsize=FONTSIZES["label"])
    ax.set_ylabel("NREG (taxpayers)", fontsize=FONTSIZES["label"])
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0.0,
        fontsize=FONTSIZES["legend"],
    )
    ax.tick_params(labelsize=FONTSIZES["tick"])

    return ax, mean_below, mean_above, r_below, r_above

def plot_chetty_comparison(z, c, c_hat_chetty, c_hat_naive, z_star, z_L, z_R, fig, axes):
    """Compare counterfactual fits with and without Chetty et al. (2011) reweighting."""
    z = np.asarray(z)
    c = np.asarray(c)
    valid = np.isfinite(c_hat_chetty) & np.isfinite(c_hat_naive)

    # Panel 1: Zoom en la región excluida y cola derecha cercana [8M, 14M]
    ax0 = axes[0]
    m0 = (z >= 8.0e6) & (z <= 14.0e6)
    ax0.step(
        z[m0], c[m0], where="post", color="darkblue", lw=1.0, alpha=0.65, label="Observed"
    )
    ax0.plot(
        z[m0],
        c_hat_chetty[m0],
        color="crimson",
        lw=1.6,
        label="Counterfactual (Chetty reweighted)",
    )
    ax0.plot(
        z[m0],
        c_hat_naive[m0],
        color="darkorange",
        ls="--",
        lw=1.6,
        label="Counterfactual (unadjusted)",
    )
    ax0.fill_between(
        z[m0],
        c_hat_naive[m0],
        c_hat_chetty[m0],
        color="crimson",
        alpha=0.18,
        label="Reweighting shift",
    )
    ax0.axvspan(z_L, z_R, color="grey", alpha=0.12, label="Excluded region")
    ax0.axvline(z_star, color="black", ls=":", lw=1.1, label=rf"$K = {z_star/1e6:.2f}\mathrm{{M}}$")

    ax0.set_xlim(8.0e6, 14.0e6)
    ax0.set_ylim(9500, 17000)
    ax0.set_title("Levels (zoom: 8M--14M)", fontsize=FONTSIZES["title"])
    ax0.set_xlabel("Taxable income (CLP)", fontsize=FONTSIZES["label"])
    ax0.set_ylabel("NREG (taxpayers)", fontsize=FONTSIZES["label"])
    ax0.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax0.tick_params(labelsize=FONTSIZES["tick"])

    # Panel 2: Exact difference (Chetty - Naive) in taxpayers
    ax1 = axes[1]
    diff = c_hat_chetty[valid] - c_hat_naive[valid]
    z_val = z[valid]

    ax1.axhline(0.0, color="darkorange", ls="--", lw=1.1)
    ax1.plot(z_val, diff, color="crimson", lw=1.7)
    ax1.fill_between(z_val, 0.0, diff, color="crimson", alpha=0.18)
    ax1.axvspan(z_L, z_R, color="grey", alpha=0.12)
    ax1.axvline(z_star, color="black", ls=":", lw=1.1)

    ax1.set_xlim(z_val.min(), z_val.max())
    ax1.set_title(r"Counterfactual shift ($\hat{c}_{\mathrm{Chetty}} - \hat{c}_{\mathrm{naive}}$)", fontsize=FONTSIZES["title"])
    ax1.set_xlabel("Taxable income (CLP)", fontsize=FONTSIZES["label"])
    ax1.set_ylabel("Difference in NREG", fontsize=FONTSIZES["label"])
    ax1.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax1.tick_params(labelsize=FONTSIZES["tick"])

    handles, labels = ax0.get_legend_handles_labels()
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

def plot_elasticity_sensitivity(df, ax, z_bench=9_907_434, eps_bench=0.2799):
    """Plot estimated elasticity across kink locations ``z_star`` and
    normalization rules (``mean`` vs ``kink``).

    Parameters
    ----------
    df : pandas.DataFrame
        Table with columns ``z_star_M``, ``normalize``, and ``epsilon``.
    ax : matplotlib.axes.Axes
        Axis to draw on.
    z_bench : float, optional
        Statutory benchmark cutoff in CLP. Default is ``9_907_434``.
    eps_bench : float, optional
        Benchmark elasticity from the paper. Default is ``0.2799``.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    styles = {
        "mean": dict(color="darkblue", marker="o", ls="-", label="Mean normalization"),
        "kink": dict(color="darkred", marker="s", ls="--", label="Kink normalization"),
    }

    for norm, kw in styles.items():
        sub = df[df["normalize"] == norm].sort_values("z_star_M")
        ax.plot(sub["z_star_M"], sub["epsilon"], lw=1.5, markersize=5.5, **kw)

    # Reference to statutory threshold K and benchmark elasticity
    ax.axvline(z_bench / 1e6, color="black", ls=":", lw=1.1, label=rf"Statutory $K = {z_bench/1e6:.2f}\mathrm{{M}}$")
    ax.scatter([z_bench / 1e6], [eps_bench], color="gold", edgecolor="black", s=65, zorder=5,
               label=rf"Benchmark ($\hat{{\epsilon}}_E = {eps_bench:.4f}$)")

    ax.set_xlabel(r"Assumed kink location $z^*$ (million CLP)", fontsize=FONTSIZES["label"])
    ax.set_ylabel(r"Estimated elasticity $\hat{\epsilon}$", fontsize=FONTSIZES["label"])
    # ax.set_title("Sensitivity of elasticity to kink location and normalization", fontsize=FONTSIZES["title"])
    ax.set_ylim(0.255, 0.305)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0.0,
        fontsize=FONTSIZES["legend"],
    )
    ax.tick_params(labelsize=FONTSIZES["tick"])
    return ax