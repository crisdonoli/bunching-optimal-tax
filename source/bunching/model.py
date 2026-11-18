import numpy as np
from numpy.polynomial import polynomial as P


def counterfactual(
    z, c, z_l, z_r, z_L, z_R, degree=4, chetty=True, max_iter=1000, tol=1e-8
) -> np.ndarray:
    r"""Counterfactual counts over ``[z_l, z_r]`` with optional right-tail reweighting.

    Fits a degree-``p`` polynomial in rescaled income ``u_j = z_j / z_r`` outside
    the excluded bunching region ``[z_L, z_R)``:

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


def excess_mass(z, c, c_hat, z_L, z_R, normalize="mean") -> dict:
    r"""Excess mass ``B`` and normalized excess (bunching) mass ``b``.

    .. math::

        B = \sum_{z_L \le z_j < z_R} (c_j - \hat{c}_j),
        \qquad
        h_j = \frac{\hat{c}_j}{dz},

    where ``dz`` is the bin width.

    Parameters
    ----------
    z : array_like
        Bin lower edges ``z_j`` in CLP.
    c : array_like
        Observed counts ``c_j`` (taxpayers per bin), aligned with ``z``.
    c_hat : array_like
        Counterfactual counts ``\hat{c}_j``; ``NaN`` entries are ignored.
    z_L, z_R : float
        Lower and upper bounds of the excluded bunching region ``[z_L, z_R)``,
        in CLP.
    normalize : {"mean", "kink"}, optional
        Normalization of the counterfactual density. ``"mean"`` uses
        ``b = B / \bar{h}`` with ``\bar{h}`` the mean of ``h_j`` over
        ``[z_L, z_R)``; ``"kink"`` uses ``b = B / h`` at the bin closest to
        ``z_L``. Default is ``"mean"``.

    Returns
    -------
    dict
        ``"B"`` : float
            Excess mass in taxpayers.
        ``"b"`` : float
            Normalized excess mass in CLP; ``h_j = \hat{c}_j / dz`` is a
            density per peso, matching
            ``\varepsilon = (B / g(z^*)) / (z^* \ln((1 - \tau_0)/(1 - \tau_1)))``
            when ``g(z^*) = \bar{h}`` (or the kink bin density).
        ``"n_bins"`` : int
            Number of bins used in ``[z_L, z_R)``.

    Raises
    ------
    ValueError
        If ``normalize`` is neither ``"mean"`` nor ``"kink"``.
    """
    z = np.asarray(z, np.float64)
    c = np.asarray(c, np.float64)
    c_hat = np.asarray(c_hat, np.float64)
    region = (z >= z_L) & (z < z_R) & np.isfinite(c_hat)
    dz = float(np.median(np.diff(np.unique(z))))

    B = float(np.sum(c[region] - c_hat[region]))
    if normalize == "mean":
        denominator = float(np.mean(c_hat[region])) / dz
    elif normalize == "kink":
        denominator = float(c_hat[np.argmin(np.abs(z - z_L))]) / dz
    else:
        raise ValueError("normalize must be 'mean' or 'kink'")
    return {"B": B, "b": B / denominator, "n_bins": int(region.sum())}


def elasticity(b, z_star, tau0=0.0, tau1=0.04) -> float:
    r"""Bunching-implied elasticity of taxable income.

    .. math::

        \hat{\varepsilon} = \frac{b}{z^* \ln\!\left(\frac{1 - \tau_0}{1 - \tau_1}\right)}.

    Parameters
    ----------
    b : float
        Normalized excess mass in CLP (see :func:`excess_mass`).
    z_star : float
        Kink point ``z^*`` in CLP.
    tau0, tau1 : float, optional
        Marginal tax rates below and above the kink. Defaults are ``0.0``
        and ``0.04``.

    Returns
    -------
    float
        Estimated elasticity ``\hat{\varepsilon}``.
    """
    return float(b / (z_star * np.log((1.0 - tau0) / (1.0 - tau1))))


def estimate(z, c, z_star, z_l, z_r, z_L, z_R, degree=4, tau0=0.0, tau1=0.04, normalize="mean") -> dict:
    r"""Full bunching pipeline: counterfactual, excess mass, and elasticity.

    Parameters
    ----------
    z, c : array_like
        Bin lower edges ``z_j`` (CLP) and observed counts ``c_j`` (taxpayers).
    z_star : float
        Kink point ``z^*`` in CLP.
    z_l, z_r : float
        Left and right bounds of the fitting window, in CLP.
    z_L, z_R : float
        Lower and upper bounds of the excluded bunching region ``[z_L, z_R)``,
        in CLP.
    degree : int, optional
        Polynomial degree. Default is 4.
    tau0, tau1 : float, optional
        Marginal tax rates below and above the kink. Defaults are ``0.0``
        and ``0.04``.
    normalize : {"mean", "kink"}, optional
        Density normalization passed to :func:`excess_mass`. Default is
        ``"mean"``.

    Returns
    -------
    dict
        ``"z_star"``, ``"B"``, ``"b"``, ``"epsilon"``, ``"c_hat"``,
        ``"degree"``, ``"z_l"``, ``"z_r"``, ``"z_L"``, ``"z_R"``,
        ``"normalize"``.

    Raises
    ------
    ValueError
        If ``z_l < z_L < z_R < z_r`` does not hold.
    ValueError
        If ``z_L <= z_star <= z_R`` does not hold.

    Notes
    -----
    Identification assumes bins in ``[z_L, z_R)`` capture the bunching and
    missing-mass response around ``z^*``, and that the counterfactual counts
    follow a polynomial in ``u_j = z_j / z_r`` after right-tail reweighting.
    """
    if not (z_l < z_L < z_R < z_r):
        raise ValueError("Require z_l < z_L < z_R < z_r")
    if not (z_L <= z_star <= z_R):
        raise ValueError("Require z_L <= z_star <= z_R")

    c_hat = counterfactual(z, c, z_l, z_r, z_L, z_R, degree=degree)
    mass = excess_mass(z, c, c_hat, z_L, z_R, normalize=normalize)
    epsilon = elasticity(mass["b"], z_star, tau0=tau0, tau1=tau1)

    return {
        "z_star": z_star,
        "B": mass["B"],
        "b": mass["b"],
        "epsilon": epsilon,
        "c_hat": c_hat,
        "degree": degree,
        "z_l": z_l,
        "z_r": z_r,
        "z_L": z_L,
        "z_R": z_R,
        "normalize": normalize,
    }


def locate_kink(z, c, z_min=None, z_max=None, window=3):
    """Locate the kink as the bin where the density stops falling.

    For each bin ``j`` compute ``r_j = c_j / mean(c_{j-window},...,c_{j-1})``
    and return the argmax of ``r_j`` restricted to ``z_min <= z_j <= z_max``.
    Returns ``(z_kink, r_max)``; ``(nan, nan)`` if no admissible bin.
    """
    z = np.asarray(z, np.float64)
    c = np.asarray(c, np.float64)
    z_min = -np.inf if z_min is None else z_min
    z_max = np.inf if z_max is None else z_max

    r = np.full(z.shape, np.nan)
    for j in range(window, len(c)):
        previous = c[j - window:j]
        if previous.sum() > 0:
            r[j] = c[j] / previous.mean()

    admissible = (z >= z_min) & (z <= z_max) & np.isfinite(r)
    if not admissible.any():
        return np.nan, np.nan
    j = np.flatnonzero(admissible)[np.argmax(r[admissible])]
    return float(z[j]), float(r[j])