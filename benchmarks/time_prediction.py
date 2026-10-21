#!/usr/bin/env python3
"""
Predict run times from the time fit parameters (output of fit_benchmarks.py).

Model per curve:  t(L) = A * L^p   (t in seconds, per measurement as benchmarked)

2D and 3D curves use separate default L lists, since 3D lattices are much smaller.

Usage:
    python predict_time.py benchmark_fit_params.csv
    python predict_time.py benchmark_fit_params.csv --L 1024 2048 --L3d 100 128
    python predict_time.py benchmark_fit_params.csv --scale 50 --margin 1.5
    python predict_time.py benchmark_fit_params.csv --solver wolff lb --lattice quadratic
    python predict_time.py benchmark_fit_params.csv --budget-hours 24
    python predict_time.py benchmark_fit_params.csv --plot time_pred.png
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REQUIRED = ["solver", "lattice", "temp", "sigma", "p", "A"]


def is_3d(solver, lattice):
    return lattice == "cubic" or solver == "cubic_nnn"


def model(L, A, p):
    return A * np.asarray(L, float) ** p


def max_L(budget_s, A, p):
    """Largest L with A*L^p <= budget."""
    if A <= 0 or budget_s <= 0:
        return np.nan
    return (budget_s / A) ** (1.0 / p)


def human(sec):
    if sec < 1:
        return f"{sec * 1e3:.3g} ms"
    if sec < 60:
        return f"{sec:.3g} s"
    if sec < 3600:
        return f"{sec / 60:.3g} min"
    if sec < 86400:
        return f"{sec / 3600:.3g} h"
    return f"{sec / 86400:.3g} d"


def slurm_time(sec):
    """Round up to SLURM --time format D-HH:MM:SS (minimum 1 minute)."""
    s = int(np.ceil(max(sec, 60)))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{d}-{h:02d}:{m:02d}:{s:02d}"


def curve_label(r):
    lab = f"{r.solver} / {r.lattice}"
    if r.sigma > 0:
        lab += f" (σ={r.sigma:g}, T={r.temp:g})"
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="time fit params csv")
    ap.add_argument("--L", type=float, nargs="+", default=[256, 512, 1024, 2048, 4096],
                    help="linear sizes for 2D curves")
    ap.add_argument("--L3d", type=float, nargs="+", default=[64, 96, 128, 192, 256],
                    help="linear sizes for 3D curves")
    ap.add_argument("--solver", nargs="+", default=None, help="only these solvers")
    ap.add_argument("--lattice", nargs="+", default=None, help="only these lattices")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiplier on the benchmarked time (e.g. number of repeats/"
                         "samples/temperatures in the real job). Default 1")
    ap.add_argument("--margin", type=float, default=1.25,
                    help="safety factor for the requested wall time (default 1.25)")
    ap.add_argument("--budget-hours", type=float, default=None,
                    help="also print the max L per curve that finishes within this wall time")
    ap.add_argument("--plot", default=None, metavar="FILE")
    ap.add_argument("--out", default="time_predictions.csv")
    args = ap.parse_args()

    if args.margin < 1:
        raise SystemExit("--margin should be >= 1")
    if args.scale <= 0:
        raise SystemExit("--scale must be positive")

    df = pd.read_csv(args.csv)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"{args.csv} is missing columns: {missing}")

    for col, vals in (("solver", args.solver), ("lattice", args.lattice)):
        if vals:
            unknown = set(vals) - set(df[col])
            if unknown:
                raise SystemExit(f"Unknown {col}(s): {sorted(unknown)}. "
                                 f"Available: {sorted(set(df[col]))}")
            df = df[df[col].isin(vals)]
    if df.empty:
        raise SystemExit("No curves left after filtering")

    rows = []
    for r in df.itertuples():
        if r.A <= 0:
            print(f"[warn] {curve_label(r)}: A <= 0, unphysical fit")
        if getattr(r, "n_points", 99) <= 3:
            print(f"[warn] {curve_label(r)}: fitted from <= 3 points; "
                  "extrapolation is uncertain")
        err = getattr(r, "mean_rel_err", 0)
        if err > 0.05:
            print(f"[warn] {curve_label(r)}: mean_rel_err={err:.1%}; "
                  "pure power law fits poorly, expect larger prediction error")
        Ls = args.L3d if is_3d(r.solver, r.lattice) else args.L
        for L in Ls:
            est = float(model(L, r.A, r.p)) * args.scale
            req = est * args.margin
            rows.append(dict(solver=r.solver, lattice=r.lattice, temp=r.temp,
                             sigma=r.sigma, L=L, est_s=est, est_human=human(est),
                             requested_s=req, slurm_time=slurm_time(req)))

    res = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:.4g}")
    print(f"\nPredicted time (scale x{args.scale:g}, margin x{args.margin:g} "
          "for requested_s/slurm_time):")
    print(res.to_string(index=False))
    res.to_csv(args.out, index=False)
    print(f"\nSaved predictions to {args.out}")

    if args.budget_hours:
        budget_s = args.budget_hours * 3600 / args.margin / args.scale
        print(f"\nMax L finishing within {args.budget_hours:g} h "
              f"(scale x{args.scale:g}, margin x{args.margin:g}):")
        for r in df.itertuples():
            Lm = max_L(budget_s, r.A, r.p)
            txt = "n/a" if np.isnan(Lm) else f"{int(Lm)}"
            print(f"  {curve_label(r):<42s} L <= {txt}")

    if args.plot:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        cmap = plt.get_cmap("tab10")
        for ax, three_d, Ls, title in ((axes[0], False, args.L, "2D lattices"),
                                       (axes[1], True, args.L3d, "3D (cubic)")):
            sub = [r for r in df.itertuples() if is_3d(r.solver, r.lattice) == three_d]
            ax.set_title(f"{title}: predicted $t = A L^p$")
            if not sub:
                ax.set_visible(False)
                continue
            Lf = np.geomspace(min(Ls), max(Ls), 200)
            for i, r in enumerate(sub):
                c = cmap(i % 10)
                ax.plot(Lf, model(Lf, r.A, r.p) * args.scale, color=c,
                        label=f"{curve_label(r)}, p={r.p:.3g}")
                ax.plot(Ls, model(Ls, r.A, r.p) * args.scale, "o", color=c)
            if args.budget_hours:
                ax.axhline(args.budget_hours * 3600, color="k", ls=":",
                           label=f"{args.budget_hours:g} h budget")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("L")
            ax.set_ylabel("predicted time [s]")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=200)
        print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()