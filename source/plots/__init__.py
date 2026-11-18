"""LaTeX typography for all matplotlib figures."""

import matplotlib

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIGURES_DIR = Path(__file__).resolve().parents[2] / "output" / "figures"
DARKBLUE_CMAP = LinearSegmentedColormap.from_list("darkblue", ["white", "darkblue"])
_cmap = plt.get_cmap('tab10')

# Master font sizes shared by every figure in the package.
FONTSIZES = {
    "title": 12,
    "label": 11,
    "tick": 10,
    "legend": 10,
    "annotation": 10,
}


def save_fig(path, fig=None, dpi=300, bbox_inches="tight", **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    fig.savefig(str(path), dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    print(f"saved figure to {path}")
    return path
