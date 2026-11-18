"""Marginal Value of Public Funds (MVPF) and Pareto calibration for top-bracket tax reforms."""

import numpy as np
import pandas as pd


# Pareto tail indices estimated in Problem Set 1 (EFH 2021 + Forbes / SII tabulations)
PS1_PARETO_ALPHAS = {
    "PS1 Vermeulen MLE (w_min = 380M, baseline)": 1.859,
    "PS1 SII Top Bracket (>310 UTA, AT 2023)": 2.085,
    "PS1 Gabaix-Ibragimov Reg (w_min = 380M)": 2.135,
    "PS1 Vermeulen MLE (w_min = P99 = 812.1M)": 2.531,
}


def mechanical_effect(d_tau: float, f_cdf_star: float, y_bar: float, z_star: float) -> float:
    r"""Mechanical revenue change ``\Delta R^M = d\tau (1 - F(z^*)) (\bar{y} - z^*)``."""
    return float(d_tau * (1.0 - f_cdf_star) * (y_bar - z_star))


def behavioral_effect(delta_r_m: float, tau: float, epsilon: float, alpha: float) -> float:
    r"""Behavioral revenue change ``\Delta R^B = -\Delta R^M \frac{\tau}{1-\tau} \epsilon \alpha``."""
    return float(-delta_r_m * (tau / (1.0 - tau)) * epsilon * alpha)


def behavioral_leakage_share(tau: float, epsilon: float, alpha: float) -> float:
    r"""Fraction of mechanical revenue offset by behavior, ``\frac{\tau}{1-\tau} \epsilon \alpha``."""
    return float((tau / (1.0 - tau)) * epsilon * alpha)


def mvpf_local(tau: float | np.ndarray, epsilon: float | np.ndarray, alpha: float | np.ndarray):
    r"""Local absolute MVPF at marginal tax rate ``\tau``:

    .. math::

        |\mathrm{MVPF}(\tau)| = \frac{1}{1 - \frac{\tau}{1-\tau} \epsilon \alpha}.
    """
    leakage = (tau / (1.0 - tau)) * epsilon * alpha
    return 1.0 / (1.0 - leakage)


def mvpf_average_midpoint(tau0: float, tau1: float, epsilon: float, alpha: float) -> float:
    r"""Approximate average MVPF over ``[\tau_1, \tau_0]`` evaluated at ``\bar{\tau} = (\tau_0 + \tau_1)/2``."""
    tau_bar = 0.5 * (tau0 + tau1)
    return float(mvpf_local(tau_bar, epsilon, alpha))


def mvpf_average_exact(tau0: float, tau1: float, epsilon: float, alpha: float) -> float:
    r"""Exact analytical average of ``|\mathrm{MVPF}(\tau)|`` integrated over ``[\tau_1, \tau_0]``:

    .. math::

        \frac{1}{\tau_0 - \tau_1} \int_{\tau_1}^{\tau_0} \frac{1 - \tau}{1 - (1 + k)\tau}\,d\tau
        = \frac{1}{1+k} + \frac{k}{(1+k)^2(\tau_0 - \tau_1)}
          \ln\!\left(\frac{1 - (1+k)\tau_1}{1 - (1+k)\tau_0}\right),

    where ``k = \epsilon \alpha``.
    """
    k = float(epsilon * alpha)
    if np.isclose(k, 0.0):
        return 1.0
    if np.isclose(tau0, tau1):
        return float(mvpf_local(tau0, epsilon, alpha))
    term1 = 1.0 / (1.0 + k)
    term2 = (k / ((1.0 + k) ** 2 * (tau0 - tau1))) * np.log(
        (1.0 - (1.0 + k) * tau1) / (1.0 - (1.0 + k) * tau0)
    )
    return float(term1 + term2)


def laffer_top_rate(epsilon: float, alpha: float) -> float:
    r"""Revenue-maximizing top marginal rate ``\tau^* = \frac{1}{1 + \epsilon \alpha}``."""
    return float(1.0 / (1.0 + epsilon * alpha))


def estimate_pareto_alpha_binned(z: np.ndarray, c: np.ndarray, z_star: float) -> dict:
    r"""Estimate Pareto tail index ``\alpha`` from binned counts above ``z_star``.

    Returns both the conditional-mean hazard estimator ``\bar{y} / (\bar{y} - z^*)``
    and the binned pseudo-MLE ``\big[\sum w_j \ln(z_j / z^*)\big]^{-1}``.
    """
    z = np.asarray(z, np.float64)
    c = np.asarray(c, np.float64)
    dz = float(np.median(np.diff(np.unique(z))))
    mid = z + 0.5 * dz
    mask = z >= z_star
    w = c[mask]
    y = mid[mask]
    total_n = float(c.sum())
    tail_n = float(w.sum())
    f_cdf_star = 1.0 - tail_n / total_n
    y_bar = float(np.sum(w * y) / tail_n)
    alpha_hazard = float(y_bar / (y_bar - z_star))
    alpha_mle = float(tail_n / np.sum(w * np.log(y / z_star)))
    return {
        "z_star": float(z_star),
        "tail_n": tail_n,
        "f_cdf_star": f_cdf_star,
        "y_bar": y_bar,
        "alpha_hazard": alpha_hazard,
        "alpha_mle": alpha_mle,
    }


def calibrate_mvpf_table(
    epsilon_specs: dict[str, float],
    alpha_specs: dict[str, float] | None = None,
    tau0: float = 0.40,
    tau1: float = 0.35,
) -> pd.DataFrame:
    """Build a calibration table of local and average MVPFs across (epsilon, alpha) pairs."""
    if alpha_specs is None:
        alpha_specs = PS1_PARETO_ALPHAS

    rows = []
    tau_bar = 0.5 * (tau0 + tau1)
    for eps_label, eps in epsilon_specs.items():
        for alpha_label, alpha in alpha_specs.items():
            leak_40 = behavioral_leakage_share(tau0, eps, alpha)
            leak_35 = behavioral_leakage_share(tau1, eps, alpha)
            leak_bar = behavioral_leakage_share(tau_bar, eps, alpha)
            rows.append(
                {
                    "epsilon_label": eps_label,
                    "epsilon": float(eps),
                    "alpha_label": alpha_label,
                    "alpha": float(alpha),
                    "k_eps_alpha": float(eps * alpha),
                    "leakage_tau0": leak_40,
                    "leakage_tau1": leak_35,
                    "leakage_taubar": leak_bar,
                    "mvpf_tau0": float(mvpf_local(tau0, eps, alpha)),
                    "mvpf_tau1": float(mvpf_local(tau1, eps, alpha)),
                    "mvpf_midpoint": mvpf_average_midpoint(tau0, tau1, eps, alpha),
                    "mvpf_exact": mvpf_average_exact(tau0, tau1, eps, alpha),
                    "tau_laffer": laffer_top_rate(eps, alpha),
                }
            )
    return pd.DataFrame(rows)
