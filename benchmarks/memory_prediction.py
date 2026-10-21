#!/usr/bin/env python3
"""
Predict memory requirements from memory_fit_params.csv (output of fit_memory.py).

Model per solver:  m(L) = C + A * L^p   (A, C in MB)

2D solvers use --L. Cubic solvers use --L3d; if --L3d is not given, it is derived
from --L at equal number of sites: L3d = round(L^(2/3)).

Usage:
    python predict_memory.py memory_fit_params.csv
    python predict_memory.py memory_fit_params.csv --L 1024 2048 4096
    python predict_memory.py memory_fit_params.csv --L 1024 2048 --L3d 100 128
    python predict_memory.py memory_fit_params.csv --margin 1.5
    python predict_memory.py memory_fit_params.csv --solver lb_lrim wolff_quadratic
    python predict_memory.py memory_fit_params.csv --budget-GB 64
    python predict_memory.py memory_fit_params.csv --plot mem_pred.png
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REQUIRED = ["solver", "p", "A_MB", "C_MB"]


def is_3d(solver):
    return "cubic" in solver


def model(L, C, A, p):
    return C + A * np.asarray(L, float) ** p


def max_L(budget_MB, C, A, p):
    """Largest L with C + A*L^p <= budget. NaN if impossible."""
    if A <= 0 or budget_MB <= C:
        return np.nan
    return ((budget_MB - C) / A) ** (1.0 / p)


def slurm_mem(mb):
    """Round up to a clean SLURM --mem string."""
    if mb >= 1024:
        return f"{int(np.ceil(mb / 1024))}G"
    return f"{int(np.ceil(mb))}M"


def sizes_for(three_d, args):
    if not three_d:
        return list(args.L)
    if args.L3d:
        return list(args.L3d)
    return [float(round(L ** (2 / 3))) for L in args.L]  # equal number of sites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="memory_fit_params.csv")
    ap.add_argument("--L", type=float, nargs="+", default=[256, 512, 1024, 2048, 4096],
                    help="linear sizes for 2D solvers")
    ap.add_argument("--L3d", type=float, nargs="+", default=None,
                    help="linear sizes for cubic solvers (default: equal sites as --L)")
    ap.add_argument("--solver", nargs="+", default=None,
                    help="only these solvers (default: all)")
    ap.add_argument("--margin", type=float, default=1.25,
                    help="safety factor applied for the requested memory (default 1.25)")
    ap.add_argument("--budget-GB", type=float, default=None,
                    help="also print the max L per solver that fits in this memory")
    ap.add_argument("--plot", default=None, metavar="FILE",
                    help="save a plot of predicted memory vs L (2D and 3D panels)")
    ap.add_argument("--out", default="memory_predictions.csv")
    args = ap.parse_args()

    if args.margin < 1:
        raise SystemExit("--margin should be >= 1")

    df = pd.read_csv(args.csv)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"{args.csv} is missing columns: {missing}")

    if args.solver:
        unknown = set(args.solver) - set(df["solver"])
        if unknown:
            raise SystemExit(f"Unknown solver(s): {sorted(unknown)}. "
                             f"Available: {sorted(df['solver'])}")
        df = df[df["solver"].isin(args.solver)]
    if df.empty:
        raise SystemExit("No solvers left after filtering")

    rows = []
    for r in df.itertuples():
        three_d = is_3d(r.solver)
        dim = 3 if three_d else 2
        if r.A_MB <= 0:
            print(f"[warn] {r.solver}: A <= 0, the fit is unphysical; "
                  "predictions are unreliable")
        if getattr(r, "n_points", 99) <= 3:
            print(f"[warn] {r.solver}: fitted from <= 3 points; extrapolation is uncertain")
        for L in sizes_for(three_d, args):
            est = float(model(L, r.C_MB, r.A_MB, r.p))
            req = est * args.margin
            rows.append(dict(solver=r.solver, dim=dim, L=L, N_sites=L ** dim,
                             est_MB=est, est_GB=est / 1024,
                             requested_MB=req, slurm_mem=slurm_mem(req)))

    res = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:.4g}")
    print(f"\nPredicted memory (margin x{args.margin:g} for requested/slurm_mem):")
    print(res.to_string(index=False))
    res.to_csv(args.out, index=False)
    print(f"\nSaved predictions to {args.out}")

    if args.budget_GB:
        budget_MB = args.budget_GB * 1024 / args.margin
        print(f"\nMax L fitting in {args.budget_GB:g} GB (with margin x{args.margin:g}):")
        for r in df.itertuples():
            Lmax = max_L(budget_MB, r.C_MB, r.A_MB, r.p)
            txt = "n/a" if np.isnan(Lmax) else f"{int(Lmax)}"
            print(f"  {r.solver:<20s} L <= {txt}")

    if args.plot:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        cmap = plt.get_cmap("tab10")
        for ax, three_d, title in ((axes[0], False, "2D lattices"),
                                   (axes[1], True, "3D (cubic)")):
            sub = [r for r in df.itertuples() if is_3d(r.solver) == three_d]
            if not sub:
                ax.set_visible(False)
                continue
            Ls = sizes_for(three_d, args)
            Lf = np.geomspace(min(Ls), max(Ls), 200)
            for i, r in enumerate(sub):
                c = cmap(i % 10)
                ax.plot(Lf, model(Lf, r.C_MB, r.A_MB, r.p) / 1024, color=c,
                        label=f"{r.solver} (p={r.p:.3g})")
                ax.plot(Ls, model(Ls, r.C_MB, r.A_MB, r.p) / 1024, "o", color=c)
            if args.budget_GB:
                ax.axhline(args.budget_GB, color="k", ls=":",
                           label=f"{args.budget_GB:g} GB budget")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("L")
            ax.set_ylabel("predicted memory [GB]")
            ax.set_title(f"{title}: $m = C + A L^p$")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=200)
        print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()