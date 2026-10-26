import argparse
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import FSS
import pandas as pd

def block_jackknife(x, stat, num_bins=20, return_replicates=False):
    """x: (n_samples, n_T). stat: function of moment dict -> (n_T,) array."""
    n = (x.shape[0] // num_bins) * num_bins
    b = n // num_bins
    x = np.abs(x[:n])
    S = {k: (x**k).reshape(num_bins, b, -1).sum(axis=1) for k in (1, 2, 4)}
    full = {k: S[k].sum(axis=0) / n for k in S}
    loo  = {k: (S[k].sum(axis=0) - S[k]) / (n - b) for k in S}
    theta, theta_i = stat(full), stat(loo)
    err = np.sqrt((num_bins - 1) * np.var(theta_i, axis=0, ddof=0))
    if return_replicates:
        return theta, err, theta_i
    return theta, err

def crossing_point(T, U1, U2, window, deg=3):
    """Crossing of two curves: polynomial fit in window, then roots of the difference."""
    m = (T >= window[0]) & (T <= window[1])
    if m.sum() <= deg + 1:
        return np.nan
    Tw = T[m]
    d = np.polyfit(Tw, U1[m], deg) - np.polyfit(Tw, U2[m], deg)
    r = np.roots(d)
    r = r[np.abs(r.imag) < 1e-9].real
    r = r[(r >= Tw.min()) & (r <= Tw.max())]
    if r.size == 0:
        return np.nan
    return r[np.argmin(np.abs(r - Tw.mean()))]

def jackknife_crossing(T, th1, rep1, th2, rep2, window, deg=3):
    """Crossing temperature and jackknife error for two independent data sets."""
    Tx = crossing_point(T, th1, th2, window, deg)
    B1, B2 = rep1.shape[0], rep2.shape[0]
    c1 = np.array([crossing_point(T, r, th2, window, deg) for r in rep1])
    c2 = np.array([crossing_point(T, th1, r, window, deg) for r in rep2])
    var = (B1 - 1) * np.var(c1) + (B2 - 1) * np.var(c2)
    return Tx, np.sqrt(var)

def fit_loglog_line(ax, lnL, lnchi, dlnchi, color="#d62728"):
    """Weighted least-squares line y = slope * x + intercept, drawn on ax.
    Returns slope, d_slope, intercept, d_intercept."""
    x, y = np.asarray(lnL), np.asarray(lnchi)
    w = 1.0 / np.asarray(dlnchi)**2

    S, Sx, Sy = w.sum(), (w * x).sum(), (w * y).sum()
    Sxx, Sxy = (w * x * x).sum(), (w * x * y).sum()
    D = S * Sxx - Sx**2

    slope = (S * Sxy - Sx * Sy) / D
    intercept = (Sxx * Sy - Sx * Sxy) / D
    d_slope = np.sqrt(S / D)
    d_intercept = np.sqrt(Sxx / D)

    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(
        xs, slope * xs + intercept,
        linestyle="--", lw=1.5, color=color, zorder=2,
        label=fr"Fit: $\eta= {-slope:.3f} \pm {d_slope:.3f}$"
    )
    return slope, d_slope, intercept, d_intercept

def interpolate_at(T, y, t_c, npts=4, deg=2):
    """Local polynomial fit through the npts grid points nearest t_c, evaluated at t_c."""
    T, y = np.asarray(T), np.asarray(y)
    idx = np.sort(np.argsort(np.abs(T - t_c))[:npts])
    p = np.polyfit(T[idx] - t_c, y[idx], deg)  # centered for conditioning
    return p[-1]  # polynomial value at T - t_c = 0


def jackknife_interpolate(T, theta, reps, t_c, npts=4, deg=2):
    """Value and jackknife error of a curve interpolated at t_c.
    theta: (n_T,) full-sample estimate, reps: (n_bins, n_T) leave-one-out replicates."""
    if not (np.min(T) <= t_c <= np.max(T)):
        raise ValueError(f"t_c={t_c} lies outside the temperature range [{np.min(T)}, {np.max(T)}]")
    if npts <= deg:
        raise ValueError("npts must be larger than deg")

    val = interpolate_at(T, theta, t_c, npts, deg)
    vals_i = np.array([interpolate_at(T, r, t_c, npts, deg) for r in reps])
    err = np.sqrt((len(reps) - 1) * np.var(vals_i, ddof=0))
    return val, err

def read_temperatures(temp_file):
    temps = []
    with open(temp_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                temps.append(float(parts[1]))
    return np.array(temps)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simple_ann(args, df): 
    return sigmoid(args.a * np.abs(df.values) + args.bias)

def plot_metric(args, parameter_data, parameter_error, temps, Tmean=None, Terr=None):

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    colors = cm.viridis(np.linspace(0, 0.9, len(args.L)))

    for i, L in enumerate(args.L):
        ax.errorbar(
            temps, parameter_data[i], yerr=parameter_error[i],
            linestyle="-", marker="o", markersize=3,
            markerfacecolor="white", markeredgecolor=colors[i], markeredgewidth=1.2,
            lw=1.4, color=colors[i], label=f"$L = {L}$",
            capsize=3, capthick=1.2, elinewidth=1.2, zorder=3,
        )

    # Plot Critical Temperature line
    if getattr(args, "t_c", 0.0) != 0.0:
        ax.axvline(
            args.t_c,
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
            label=f"$T_c = {args.t_c}$",
            zorder=1,
        )

    if Tmean is not None and Terr is not None:
        ax.axvline(
            Tmean,
            color="#1f77b4",
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
            label=fr"$T^* = {Tmean:.4f} \pm {Terr:.4f}$",
            zorder=2,
        )
        ax.fill_between(
            [Tmean - Terr, Tmean + Terr],
            ax.get_ylim()[0], ax.get_ylim()[1],
            color="#1f77b4",
            alpha=0.2,
            zorder=2,
        )

    # Labels and Aesthetics
    ax.set_xlabel(r"Temperature ($T$)", fontsize=12, labelpad=8)
    ax.set_ylabel(args.metric, fontsize=12, labelpad=8)

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.legend(frameon=False, fontsize=10)

    ax.set_xlim(args.x_min, args.x_max)
    ax.set_ylim(args.y_min, args.y_max)

    plt.tight_layout()
    plt.savefig(args.save_path, bbox_inches="tight")
    plt.show()
    # plt.close(fig)

def scale_suceptibility(args, suceptibility_at_Tc, suceptibility_error):

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    colors = cm.viridis(np.linspace(0, 0.9, len(args.L)))

    chi = np.asarray(suceptibility_at_Tc) 
    dchi = np.asarray(suceptibility_error)

    for i in range(len(chi)): 
        chi[i] = chi[i] / args.N[i]
        dchi[i] = dchi[i] / args.N[i]

    lnL = np.log(np.asarray(args.L, dtype=float))
    lnchi = np.log(chi)
    dlnchi = dchi / chi  # error propagation: d(ln chi) = d(chi) / chi

    ax.errorbar(
        lnL, lnchi, yerr=dlnchi,
        linestyle="none", marker="o", markersize=4,
        markerfacecolor="white", markeredgecolor=colors[0], markeredgewidth=1.2,
        color=colors[0], label=r"Data",
        capsize=3, capthick=1.2, elinewidth=1.2, zorder=3,
    )

    slope, d_slope, intercept, d_intercept = fit_loglog_line(ax, lnL, lnchi, dlnchi)
    print(f"slope = {slope:.4f} +- {d_slope:.4f}, intercept = {intercept:.4f} +- {d_intercept:.4f}")
    print(f"eta = {-slope:.4f} +- {d_slope:.4f}")

    # Labels and Aesthetics
    ax.set_xlabel(r"$\ln L$", fontsize=12, labelpad=8)
    ax.set_ylabel(r"$\ln (\chi_{T_c}/L^2)$", fontsize=12, labelpad=8)

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.legend(frameon=False, fontsize=10)

    ax.set_xlim(args.x_min, args.x_max)
    ax.set_ylim(args.y_min, args.y_max)

    plt.tight_layout()
    plt.savefig(args.save_path, bbox_inches="tight")
    plt.show()

def plot_fss_order_parameter(args, parameter_data, parameter_error, temps): 
    if args.upper == 0: 
        args.upper = len(temps) - 1

    ret, x, y, dy, x_true, y_true, dy_true = FSS.finite_size_scaling(
        args.L, np.array(temps), np.array(parameter_data), np.array(parameter_error),
        args.lower, args.upper, args.t_c, args.nu, args.beta
    )

    # ---------- Plotting ----------
    plt.rcParams.update({
        "font.size": 12,
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })

    n_L = len(args.L)
    colors = cm.viridis(np.linspace(0, 0.9, n_L))

    def plot_collapse(ax, x_data, y_data, dy_data, title):
        for i, L in enumerate(args.L):
            xi, yi, dyi = np.asarray(x_data[i]), np.asarray(y_data[i]), np.asarray(dy_data[i])
            order = np.argsort(xi)
            xi, yi, dyi = xi[order], yi[order], dyi[order]

            ax.errorbar(
                xi, yi, yerr=dyi,
                linestyle="-", marker="o", markersize=3,
                markerfacecolor="white", markeredgecolor=colors[i], markeredgewidth=1.2,
                lw=1.4, color=colors[i], label=f"L = {L}",
                capsize=3, capthick=1.2, elinewidth=1.2, zorder=3,
            )

        ax.set_xlabel(r"$(T - T_c)\, L^{1/\nu}$")
        ax.set_ylabel(r"$O(T, L) \, L^{\beta/\nu}$")
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)

    ncols = 2 if args.ts else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5.5))
    if ncols == 1:
        axes = [axes]

    plot_collapse(
        axes[0], x, y, dy,
        rf"FSS collapse (fitted): $T_c={ret['rho']:.4f}({ret['drho']:.4f})$, "
        rf"$\nu={ret['nu']:.3f}({ret['dnu']:.3f})$, $\beta={-ret['zeta']:.3f}({ret['dzeta']:.3f})$"
    )

    if args.ts:
        plot_collapse(
            axes[1], x_true, y_true, dy_true,
            rf"FSS collapse (reference): $T_c={args.t_c:.4f}$, "
            rf"$\nu={args.nu:.3f}$, $\beta={args.beta:.3f}$"
        )

    fig.suptitle(f"Finite-size scaling — {args.ising_solver}", fontsize=14, y=1.02)
    fig.tight_layout()

    plt.tight_layout()
    plt.savefig(args.save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)  # Close the figure to free memory

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot average magnetization vs temperature for a given lattice size and solver."
    )
    parser.add_argument(
        "-f", "--file-path",
        dest="file_path",
        type=str,
        nargs='+',
        required=True,
        help="Path to the CSV file(s) containing the data for each lattice size"
    )
    parser.add_argument(
        "-t", "--temp-file",
        dest="temp_file",
        type=str,
        required=True,
        help="Path to the CSV file containing the temperature data"
    )
    parser.add_argument(
        "-s", "--save_path",
        dest="save_path",
        type=str,
        default="plot.pdf",
        help="Path to save the plots"
    )
    parser.add_argument(
        "-L", "--lattice-sizes",
        dest="L",
        type=int,
        nargs='+',
        required=True,
        help="Size of the lattices (e.g., 8 16 32 48)"
    )
    parser.add_argument(
        "-N", "--num-spins",
        dest="N",
        type=int,
        nargs='+',
        required=True,
        help="Number of spins in each lattice (e.g., 64 256 1024 2304)"
    )
    parser.add_argument(
        "-op", "--order-parameter",
        dest="order_parameter",
        type=str,
        required=True,
        help="Order parameter to plot"
    )
    parser.add_argument(
        "-m", "--metric", 
        dest="metric",
        type=str,
        required=False, 
        help="Metric to plot (e.g., magnetization, simple_ann, susceptibility, etc.)"
    )
    parser.add_argument(
        "-tc", "--critical-temperature",
        dest="t_c",
        type=float,
        default=0,
        required=False,
        help="Critical temperature for the Ising model"
    )
    parser.add_argument(
        "-a", "--a",
        dest="a",
        type=float,
        default=1.0,
        required=False,
        help="Minimum value for the y-axis range"
    )
    parser.add_argument(
        "-b", "--bias",
        dest="bias",
        type=float,
        default=0.0,
        required=False,
        help="Offset for the y-axis range"
    )
    parser.add_argument(
        "-lower", "--lower-bound",
        dest="lower",
        type=int,
        default=0,
        required=False,
        help="Lower bound for the temperature range in finite-size scaling"
    )
    parser.add_argument(
        "-upper", "--upper-bound",
        dest="upper",
        type=int,
        default=0,
        required=False,
        help="Upper bound for the temperature range in finite-size scaling"
    )
    parser.add_argument(
        "-nu", "--nu-value",
        dest="nu",
        type=float,
        default=1.0,
        required=False,
        help="Critical exponent nu for finite-size scaling"
    )
    parser.add_argument(
        "-beta", "--beta-value",
        dest="beta",
        type=float,
        default=0.125,
        required=False,
        help="Critical exponent beta for finite-size scaling"
    )
    parser.add_argument(
        "-ts", "--true-scaling",
        dest="ts",
        type=bool,
        default=False,
        required=False,
        help="If True, plot the true scaling function and the finite-size scaling collapse"
    )
    parser.add_argument(
        "-bins", "--num-bins",
        dest="bins",
        type=int,
        default=20,
        required=False,
        help="Number of bins for Jackknife resampling"
    )
    parser.add_argument(
        "-cw", "--cross-window",
        dest="cross_window",
        type=float,
        nargs=2,
        default=None,
        help="T window (lo hi) around the Binder crossing, e.g. 2.2 2.35"
    )
    parser.add_argument(
        "-cd", "--cross-deg",
        dest="cross_deg",
        type=int,
        default=3,
        help="Polynomial degree for the crossing fits"
    )
    parser.add_argument(
        "-xmax", "--x-max",
        dest="x_max",
        type=float,
        default=None,
        help="Maximum x-axis value for the plot"
    )
    parser.add_argument(
        "-xmin", "--x-min",
        dest="x_min",
        type=float,
        default=None,
        help="Minimum x-axis value for the plot"
    )
    parser.add_argument(
        "-ymax", "--y-max",
        dest="y_max",
        type=float,
        default=None,
        help="Maximum y-axis value for the plot"
    )
    parser.add_argument(
        "-ymin", "--y-min",
        dest="y_min",
        type=float,
        default=None,
        help="Minimum y-axis value for the plot"
    )
    parser.add_argument(
        "-fss", "--finite-size-scaling",
        dest="fss",
        action="store_true",
        default=False,
        help="Plot finite-size scaling"
    )
    parser.add_argument(
        "-npts", "--interpolation-npts",
        dest="interp_npts",
        type=int,
        default=4,
        help="Number of points to use for interpolation at T_c"
    )
    parser.add_argument(
        "-deg", "--interpolation-deg",
        dest="interp_deg",
        type=int,
        default=2,
        help="Degree of polynomial for interpolation at T_c"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    parameter_data = []
    parameter_errors = []
    replicates = []

    temps = read_temperatures(args.temp_file)

    for i, L in enumerate(args.L):

        file_path = args.file_path[i]

        df = pd.read_csv(file_path, index_col=0)
        if args.order_parameter == "simple_ann":
            input_array = simple_ann(args, df)
        else:
            input_array = df.abs().to_numpy()

        if args.metric == "susceptibility":
            stat = lambda M: (M[2] - M[1]**2) * args.N[i] / temps
        elif args.metric == "binder":
            stat = lambda M: 1.0 - M[4] / (3.0 * M[2]**2)
        else:
            stat = lambda M: M[1]

        parameter, parameter_error, theta_i = block_jackknife(input_array, stat, num_bins=args.bins, return_replicates=True)

        parameter_data.append(parameter)
        parameter_errors.append(parameter_error)
        replicates.append(theta_i)

    if args.metric == "binder" and args.cross_window is not None:
        Ts, dTs = [], []
        for i, j in itertools.combinations(range(len(args.L)), 2):
            Tx, dTx = jackknife_crossing(
                temps, parameter_data[i], replicates[i],
                parameter_data[j], replicates[j],
                args.cross_window, args.cross_deg,
            )
            Ts.append(Tx); dTs.append(dTx)
            print(f"L=({args.L[i]}, {args.L[j]}):  T* = {Tx:.5f} +- {dTx:.5f}")

        Ts, dTs = np.array(Ts), np.array(dTs)
        ok = np.isfinite(Ts) & np.isfinite(dTs) & (dTs > 0)
        if ok.any():
            w = 1 / dTs[ok]**2
            Tmean = np.sum(w * Ts[ok]) / w.sum()
            Terr = 1 / np.sqrt(w.sum())
            print(f"Weighted mean T* = {Tmean:.5f} +- {Terr:.5f}")

        plot_metric(args, parameter_data, parameter_errors, temps, Tmean, Terr)

    elif args.metric == "susceptibility":
        suceptibility_at_Tc = []
        suceptibility_error = []
        for i, L in enumerate(args.L):
            val, err = jackknife_interpolate(
                temps, parameter_data[i], replicates[i], args.t_c,
                npts=args.interp_npts, deg=args.interp_deg,
            )
            suceptibility_at_Tc.append(val)
            suceptibility_error.append(err)
        scale_suceptibility(args, suceptibility_at_Tc, suceptibility_error)
    
    else:
        plot_metric(args, parameter_data, parameter_errors, temps)

    if args.fss:
        plot_fss_order_parameter(args, parameter_data, parameter_errors, temps)


