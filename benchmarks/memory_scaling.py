#!/usr/bin/env python3
"""
Plot max memory vs lattice size and fit  m = C + A * L^p.

C is a constant baseline (runtime/allocator overhead), A is the prefactor.
By default p is fixed (2 for 2D, 3 for 3D) and C, A are fitted per solver.
Use --fit-p to fit p as well, --no-offset to drop C.

The exponent can be pinned per solver or dimension with --p:
    --p lb_lrim=2.2          one solver
    --p 3d=3                 all 3D curves   (also: 2d=...)
Lookup order: solver, 2d/3d.

Use --predict to print (and plot out to) estimates at larger L.

Usage:
    python fit_memory.py memory.csv
    python fit_memory.py memory.csv --fit-p
    python fit_memory.py memory.csv --p lb_lrim=2 --predict 1024 2048
    python fit_memory.py memory.csv --linear --out mem.png
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

P_2D, P_3D = 2.0, 3.0


def is_3d(solver):
    return "cubic" in solver


def panel_name(solver):
    return "3D (cubic)" if is_3d(solver) else "2D lattices"


def fit_CA(L, m, p, offset=True):
    """Weighted (relative-error) linear least squares for m = C + A*L^p, p fixed.
    Returns (C, A, sum of squared relative residuals)."""
    w = 1.0 / m
    X = np.column_stack([np.ones_like(L), L**p]) if offset else (L**p)[:, None]
    coef, *_ = np.linalg.lstsq(X * w[:, None], m * w, rcond=None)
    C = coef[0] if offset else 0.0
    A = coef[-1]
    resid = np.sum(((C + A * L**p - m) * w) ** 2)
    return C, A, resid


def fit_CAp(L, m, offset=True, p_range=(0.5, 6.0)):
    """Fit p by 1D search (coarse then fine grid); C, A solved linearly for each p."""
    grid = np.arange(p_range[0], p_range[1] + 1e-9, 0.02)
    best = min(grid, key=lambda p: fit_CA(L, m, p, offset)[2])
    fine = np.arange(best - 0.02, best + 0.02 + 1e-9, 0.0005)
    p = min(fine, key=lambda q: fit_CA(L, m, q, offset)[2])
    C, A, _ = fit_CA(L, m, p, offset)
    return C, A, p


def model(L, C, A, p):
    return C + A * L**p


def mean_rel_err(L, m, C, A, p):
    return np.mean(np.abs(model(L, C, A, p) - m) / m)


def parse_overrides(items):
    out = {}
    for item in items or []:
        try:
            key, val = item.rsplit("=", 1)
            out[key.strip()] = float(val)
        except ValueError:
            raise SystemExit(f"Bad --p '{item}', expected KEY=VALUE, e.g. lb_lrim=2")
    return out


def lookup_p(overrides, solver):
    """Return (key, p) for the most specific matching override, else (None, None)."""
    for key in (solver, "3d" if is_3d(solver) else "2d"):
        if key in overrides:
            return key, overrides[key]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="memory_fits.png")
    ap.add_argument("--linear", action="store_true", help="linear axes instead of log-log")
    ap.add_argument("--p", action="append", metavar="KEY=VALUE",
                    help="pin exponent p (repeatable); see docstring for keys")
    ap.add_argument("--fit-p", action="store_true",
                    help="also fit p (needs >=4 points per curve to be non-exact)")
    ap.add_argument("--no-offset", action="store_true",
                    help="drop the constant baseline C (m = A*L^p)")
    ap.add_argument("--predict", type=float, nargs="+", metavar="L", default=[],
                    help="print estimated memory at these L (and extend the plot)")
    args = ap.parse_args()
    overrides = parse_overrides(args.p)
    offset = not args.no_offset

    df = pd.read_csv(args.csv)
    df["MB"] = df["kbytes_max"] / 1024.0
    df["panel"] = [panel_name(s) for s in df["ising_solver"]]

    panels = sorted(df["panel"].unique(), key=lambda s: (s != "2D lattices", s))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.5), squeeze=False)
    ax_of = dict(zip(panels, axes[0]))

    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    results, predictions = [], []
    counter = {p: 0 for p in panels}
    used_keys = set()

    for solver, g in df.groupby("ising_solver"):
        panel = panel_name(solver)
        ax = ax_of[panel]
        i = counter[panel]
        counter[panel] += 1

        g = g.sort_values("L")
        L = g["L"].to_numpy(float)
        m = g["MB"].to_numpy(float)

        key, p = lookup_p(overrides, solver)
        custom = key is not None
        if custom:
            used_keys.add(key)
            mode = "override"
            C, A, _ = fit_CA(L, m, p, offset)
        elif args.fit_p:
            mode = "fitted"
            C, A, p = fit_CAp(L, m, offset)
        else:
            mode = "default"
            p = P_3D if is_3d(solver) else P_2D
            C, A, _ = fit_CA(L, m, p, offset)

        label = f"{solver}, p={p:.3g}, A={A:.2e} MB"
        if offset:
            label += f", C={C:.2f} MB"

        color = cmap(i % 10)
        ax.plot(L, m, markers[i % len(markers)], color=color, label=label)
        Lmax = max([L.max()] + args.predict)
        Lf = np.geomspace(L.min(), Lmax, 200) if not args.linear else np.linspace(L.min(), Lmax, 200)
        ax.plot(Lf, model(Lf, C, A, p), "--" if mode != "fitted" else "-",
                color=color, alpha=0.7)

        for Lp in args.predict:
            est = model(Lp, C, A, p)
            predictions.append(dict(solver=solver, L=Lp, est_MB=est, est_GB=est / 1024))
            ax.plot([Lp], [est], markers[i % len(markers)], color=color, mfc="none")

        results.append(dict(solver=solver, p=p, p_mode=mode, n_points=len(L),
                            A_MB=A, C_MB=C, mean_rel_err=mean_rel_err(L, m, C, A, p)))

    for key in overrides:
        if key not in used_keys:
            print(f"[warn] --p {key} matched no data (or was shadowed by a more specific key)")

    for panel, ax in ax_of.items():
        ax.set_xlabel("L")
        ax.set_ylabel("max memory [MB]")
        ax.set_title(f"{panel}: $m = C + A L^p$" if offset else f"{panel}: $m = A L^p$")
        if not args.linear:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)

    res = pd.DataFrame(results)
    pd.set_option("display.float_format", lambda x: f"{x:.4e}")
    print("\nResults (solid line = p fitted, dashed = p fixed):")
    print(res.to_string(index=False))
    res.to_csv("memory_fit_params.csv", index=False)
    if predictions:
        print("\nPredictions:")
        print(pd.DataFrame(predictions).to_string(index=False))
    print(f"\nSaved plot to {args.out} and parameters to memory_fit_params.csv")
    plt.show()


if __name__ == "__main__":
    main()