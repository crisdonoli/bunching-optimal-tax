import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from source.plots.p1 import plot_sensitivity
CSV = ROOT / "output" / "tables" / "p1_iv_sensitivity.csv"
TEX = ROOT / "report" / "tables" / "p1_iv.tex"
FIG = ROOT / "output" / "figures" / "p1" / "p1_iv_sensitivity.pdf"


def num(x, decimals):
    s = f"{x:,.{decimals}f}"
    return s.translate(str.maketrans({",": ".", ".": ","})).replace(".", "\\,")


def main():
    sens = pd.read_csv(CSV)

    rows = []
    for r in sens.itertuples(index=False):
        rows.append(
            f"{r.bin_k} & {r.window} & {num(r.B, 1)} & {num(r.b, 0)} & {num(r.epsilon, 3)} \\\\"
        )

    body = "\n".join(rows)
    tex = f"""\\begin{{table}}[htbp]
\\centering
\\small
\\begin{{tabular}}{{rrrrr}}
\\toprule
Bin (miles CLP) & Ventana (CLP) & $B$ & $b$ (CLP) & $\\hat{{\\epsilon}}$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Elasticidad estimada seg\\'un ancho de bin y ventana de estimaci\\'on.}}
\\label{{tab:p1_iv}}
\\end{{table}}
"""
    TEX.write_text(tex)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_sensitivity(sens, axes)
    fig.tight_layout()
    fig.savefig(FIG, bbox_inches="tight")


if __name__ == "__main__":
    main()
