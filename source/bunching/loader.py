import numpy as np
import pandas as pd


def load_counts(path: str, year: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load binned counts of taxpayers from a ``;``-separated CSV.

    Parameters
    ----------
    path : str
        Path to the CSV file. Expected columns are ``group_0`` (lower bin
        edge in CLP, i.e. the bin label used as ``z``), ``NREG`` (count of
        taxpayers in the bin, i.e. ``c_j``) and ``year``.
    year : int or None, optional
        Keep only rows for this year. When ``None``, all years are kept and
        mixed together (assumption: one year per call).

    Returns
    -------
    z : numpy.ndarray, shape (n,)
        Lower bin edges ``z_j`` in CLP, sorted ascending.
    c : numpy.ndarray, shape (n,)
        Counts ``c_j`` from ``NREG``, aligned with ``z``.

    Notes
    -----
    ``z`` is ``group_0`` (lower bin edge, CLP) and ``c_j`` is ``NREG``
    (count of taxpayers). Rows are sorted by ``z`` before returning.
    """
    df = pd.read_csv(path, sep=";")
    if year is not None:
        df = df[df["year"] == year]
    df = df.sort_values("group_0")
    return df["group_0"].to_numpy(np.float64), df["NREG"].to_numpy(np.float64)


def load_utm(path) -> dict[int, float]:
    """Load a year-to-UTM table and return ``{year: utm_clp}``.

    The file must be a csv with columns ``year`` and ``utm`` (separator ','
    or ';').
    """
    df = pd.read_csv(path, sep=None, engine="python")
    return {int(y): float(u) for y, u in zip(df["year"], df["utm"])}


def resample_counts(z, c, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate `factor` contiguous bins into one of width factor*dz.

    z is the lower bin edge (CLP) and c_j the count of taxpayers; the new count
    is the sum over the aggregated bins. Trailing incomplete groups are dropped.
    """
    n = len(c) - len(c) % factor
    z2 = z[:n:factor]
    c2 = c[:n].reshape(-1, factor).sum(axis=1)
    return np.asarray(z2, float), np.asarray(c2, float)


def rebin(z, c, width):
    """Aggregate counts onto a grid of bins of fixed ``width`` in the units of ``z``.

    Counts are assigned to the bin that contains the centre of the original bin.
    Returns ``(z_new, c_new)``.
    """
    z = np.asarray(z, np.float64)
    c = np.asarray(c, np.float64)
    dz = float(np.median(np.diff(z)))
    centers = z + dz / 2.0
    idx = np.floor(centers / width).astype(int)
    out = np.bincount(idx, weights=c, minlength=int(idx.max()) + 1)
    z_new = np.arange(len(out), dtype=np.float64) * width
    return z_new, out.astype(np.float64)


def pool_counts(counts, scale, width):
    """Pool several years after normalising income by ``scale``.

    ``counts`` is ``{year: (z, c)}``, ``scale`` is ``{year: float}`` and
    ``width`` the bin width in the normalised units. Each year's income is
    divided by ``scale[year]``, rebinned on the common grid and summed.
    Returns ``(z_pool, c_pool)`` as float arrays.
    """
    pooled = {}
    for year, (z, c) in counts.items():
        z_year, c_year = rebin(np.asarray(z, np.float64) / scale[year], c, width)
        for k, v in zip(np.rint(z_year / width).astype(int), c_year):
            pooled[k] = pooled.get(k, 0.0) + v
    keys = np.array(sorted(pooled))
    return keys * width, np.array([pooled[k] for k in keys], np.float64)


def weighted_median(z, c):
    """Count-weighted median of the income distribution.

    ``z`` are bin lower edges, ``c`` the counts. The median is the midpoint
    ``z[j] + dz/2`` of the bin ``j`` that contains the 50th percentile of
    mass, with ``dz`` the median bin width (no interpolation inside the bin).
    """
    z = np.asarray(z, np.float64)
    c = np.asarray(c, np.float64)
    cum = np.cumsum(c) / c.sum()
    j = int(np.searchsorted(cum, 0.5))
    return float(z[j] + np.median(np.diff(z)) / 2.0)
