# Bunching, Optimization Frictions, and the Marginal Value of Public Funds in Chile

Reproducible codebase and LaTeX report for **Advanced Public Finance — Problem Set 2** (*Facultad de Economía y Negocios, Universidad de Chile*).

- **Full Report (PDF):** [report/main.pdf](report/main.pdf)
- **Code Traceability:** Throughout the report, footnotes of the form `code: <path>:LXX` point directly to the exact file and line number in this repository that implements the corresponding estimator, calibration, table, or figure.

## Overview

1. **Bunching and the Elasticity of Taxable Income in Chile (`report/sections/01.tex`):**
   Replicates the Chilean first-kink bunching analysis on administrative SII wage-earner records (`data/bunch_counts_wages.csv`) using a fourth-degree polynomial counterfactual with Chetty et al. (2011) iterative right-tail reweighting (`source/bunching/model.py`), evaluates sensitivity across bin widths and estimation windows (`scripts/make_p1_iv_table.py`), and extends the analysis to 2005–2023 pooled distributions (`report/sections/appendix.tex`).
2. **Extending Bunching Assumptions (`report/sections/02.tex`):**
   Derives the frictionless uniform-ability benchmark and characterizes attenuation bias under contract-renegotiation frictions ($\phi > 0$), develops two nonparametric counterfactual alternatives (reference/shifted-kink and shape-constrained bounded-curvature bounds), and visualizes the mechanisms (`source/plots/p2.py`, `notebooks/p2/examples.ipynb`).
3. **Marginal Value of Public Funds (`report/sections/03.tex`):**
   Derives the mechanical effect, behavioral Pareto-tail offset, and $|\mathrm{MVPF}|$ for a $40\% \to 35\%$ top-bracket rate cut; estimates upper-tail bunching on `data/synthetic_bunch_counts.csv`; calibrates local and exact-integral MVPFs using Problem Set 1 Pareto indices (`source/bunching/mvpf.py`, `scripts/make_p3_tables.py`, `notebooks/p3/pipeline.ipynb`); and presents a policy brief.

## Repository Layout

```text
.
├── data/
│   ├── bunch_counts_wages.csv         # SII binned wage-earner counts (2005–2023)
│   ├── synthetic_bunch_counts.csv     # Synthetic binned counts for top-bracket analysis
│   └── threshold.csv                  # Annual statutory first-bracket exemption cutoffs (CLP)
├── notebooks/
│   ├── p1/                            # Problem 1 exploration and pipeline
│   ├── p2/examples.ipynb              # Problem 2 visual examples
│   └── p3/pipeline.ipynb              # Problem 3 reproducible pipeline
├── output/
│   ├── figures/                       # Generated PDF figures (examples/, p1/, p2_*, p3/)
│   └── tables/                        # Exported CSV summary tables
├── report/
│   ├── main.tex                       # Master LaTeX document (LuaLaTeX + Biber)
│   ├── main.pdf                       # Compiled report
│   ├── sections/                      # 01.tex, 02.tex, 03.tex, appendix.tex
│   └── tables/                        # Generated LaTeX tables (p1_iv.tex, p3_*.tex)
├── scripts/
│   ├── make_p1_iv_table.py            # Generates Problem 1 sensitivity table & figure
│   └── make_p3_tables.py              # Generates Problem 3 tables & figures
└── source/
    ├── bunching/
    │   ├── loader.py                  # Data loading, rebinning, and multi-year pooling
    │   ├── model.py                   # Polynomial counterfactual, excess mass, ETI
    │   └── mvpf.py                    # Mechanical/behavioral effects, MVPF, Pareto calibration
    └── plots/
        ├── __init__.py                # Shared LaTeX matplotlib styling and save_fig()
        ├── examples.py                # Stylized theory figures
        ├── p1.py                      # Problem 1 figures
        ├── p2.py                      # Problem 2 figures
        └── p3.py                      # Problem 3 figures
```

## Reproducing Results

1. **Generate tables and figures:**
   ```bash
   python3 scripts/make_p1_iv_table.py
   python3 scripts/make_p3_tables.py
   ```
   Or execute the notebooks in `notebooks/p1/pipeline.ipynb`, `notebooks/p2/examples.ipynb`, and `notebooks/p3/pipeline.ipynb`.
2. **Compile the report:**
   ```bash
   cd report
   latexmk -lualatex main.tex
   ```
# bunching-optimal-tax
