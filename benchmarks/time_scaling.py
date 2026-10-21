#!/usr/bin/env python3
"""
Plot avg time vs lattice size and fit t = A * L^p (no constant term).
By default both A and p are fitted per curve (linear least squares in log-log
space). The exponent can be pinned per topology with --p, in which case only A
is fitted (dashed line):

    --p wolff/quadratic=3.5      solver/lattice
    --p cubic=5                  lattice type (all solvers on it)
    --p lb/quadratic@2=3         one specific sigma only
    --p 3d=5                     all 3D curves   (also: 2d=...)
Lookup order: solver/lattice@sigma, solver/lattice, lattice, 2d/3d.

--default-p skips the free fit and uses P_2D / P_3D wherever there is no override.

Usage:
    python fit_benchmarks.py benchmark_results.csv
    python fit_benchmarks.py benchmark_results.csv --p cubic=5 --p wolff/triangular=3.5
    python fit_benchmarks.py benchmark_results.csv --default-p
    python fit_benchmarks.py benchmark_results.csv --linear --out fits.png
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

P_2D, P_3D = 2.0, 2.8


def is_3d(solver, lattice):
    return lattice == "cubic" or lattice == "cubic_nnn"


def panel_name(solver, lattice):
    return "3D (cubic)" if is_3d(solver, lattice) else "2D lattices"


def fit_Ap(L, t):
    """Fit t = A*L^p by linear least squares in log-log space. Returns (A, p)."""
    p, logA = np.polyfit(np.log(L), np.log(t), 1)
    return np.exp(logA), p


def fit_A(L, t, p):
    """Least squares for t = A*L^p with p fixed. Closed form: A = sum(x*t) / sum(x*x)."""
    x = L**p
    return np.sum(x * t) / np.sum(x * x)


def mean_rel_err(L, t, p, A):
    return np.mean(np.abs(A * L**p - t) / t)


def parse_overrides(items):
    out = {}
    for item in items or []:
        try:
            key, val = item.rsplit("=", 1)
            out[key.strip()] = float(val)
        except ValueError:
            raise SystemExit(f"Bad --p '{item}', expected KEY=VALUE, e.g. cubic=5")
    return out


def lookup_p(overrides, solver, lattice, sigma):
    """Return (key, p) for the most specific matching override, else (None, None)."""
    for key in (f"{solver}/{lattice}@{sigma:g}", f"{solver}/{lattice}", lattice,
                "3d" if is_3d(solver, lattice) else "2d"):
        if key in overrides:
            return key, overrides[key]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="time_fits.png")
    ap.add_argument("--linear", action="store_true", help="linear axes instead of log-log")
    ap.add_argument("--p", action="append", metavar="KEY=VALUE",
                    help="pin exponent p (repeatable); see docstring for keys")
    ap.add_argument("--default-p", action="store_true",
                    help="don't fit p; use P_2D/P_3D unless pinned with --p")
    ap.add_argument("--sweeps", type=float, default=None,
                help="number of sweeps per measurement; draws error bars of t/sqrt(sweeps)")
    args = ap.parse_args()
    overrides = parse_overrides(args.p)

    if args.sweeps is not None and args.sweeps <= 0:
        raise SystemExit("--sweeps must be positive")

    df = pd.read_csv(args.csv)
    df["L"] = df["size"].astype(str).str.split("x").str[0].astype(int)
    df["panel"] = [panel_name(s, l) for s, l in zip(df["ising_solver"], df["lattice_type"])]

    panels = sorted(df["panel"].unique(), key=lambda s: (s != "2D lattices", s))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.5), squeeze=False)
    ax_of = dict(zip(panels, axes[0]))

    cmap = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    keys = ["ising_solver", "lattice_type", "temp", "sigma"]
    results = []
    counter = {p: 0 for p in panels}
    used_keys = set()

    for (solver, lattice, temp, sigma), g in df.groupby(keys):
        panel = panel_name(solver, lattice)
        ax = ax_of[panel]
        i = counter[panel]
        counter[panel] += 1

        g = g.sort_values("L")
        L = g["L"].to_numpy(float)
        t = g["avg_time_s"].to_numpy(float)

        key, p = lookup_p(overrides, solver, lattice, sigma)
        custom = key is not None
        if custom:
            used_keys.add(key)
            mode = "override"
            A = fit_A(L, t, p)
        elif args.default_p or len(L) < 2:
            mode = "default"
            p = P_3D if is_3d(solver, lattice) else P_2D
            A = fit_A(L, t, p)
        else:
            mode = "fitted"
            A, p = fit_Ap(L, t)

        label = f"{solver} / {lattice}"
        if sigma > 0:
            label += f" (σ={sigma:g}, T={temp:g})"
        label += f", p={p:.3g}, A={A:.2e}"

        color = cmap(i % 10)
        # ax.plot(L, t, markers[i % len(markers)], color=color, label=label)
        if args.sweeps:
            yerr = t / np.sqrt(args.sweeps)
            ax.errorbar(L, t, yerr=yerr, fmt=markers[i % len(markers)], color=color,
                        label=label, capsize=3, elinewidth=1)
        else:
            ax.plot(L, t, markers[i % len(markers)], color=color, label=label)
        Lf = np.linspace(L.min(), L.max(), 200)
        ax.plot(Lf, A * Lf**p, "--" if custom else "-", color=color, alpha=0.7)

        results.append(dict(solver=solver, lattice=lattice, temp=temp, sigma=sigma,
                            p=p, p_mode=mode, n_points=len(L), A=A,
                            mean_rel_err=mean_rel_err(L, t, p, A)))

    for key in overrides:
        if key not in used_keys:
            print(f"[warn] --p {key} matched no data (or was shadowed by a more specific key)")

    for panel, ax in ax_of.items():
        ax.set_xlabel("L")
        ax.set_ylabel("avg time [s]")
        ax.set_title(f"{panel}: $t = A L^p$")
        if not args.linear:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)

    res = pd.DataFrame(results)
    pd.set_option("display.float_format", lambda x: f"{x:.4e}")
    print("\nResults (dashed line = p fixed by --p):")
    print(res.to_string(index=False))
    res.to_csv("time_fit_params.csv", index=False)
    print(f"\nSaved plot to {args.out} and parameters to time_fit_params.csv")
    plt.show()


if __name__ == "__main__":
    main()