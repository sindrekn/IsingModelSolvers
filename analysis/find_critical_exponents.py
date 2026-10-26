import argparse
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from plot_raw_data import read_temperatures, simple_ann, get_metric, block_jackknife

def fit_loglog_line(ax, lnL, lnchi, dlnchi, color="#d62728"):
    """
    Weighted least-squares line y = slope * x + intercept, drawn on ax.
    Returns slope, d_slope, intercept, d_intercept.
    ax: matplotlib Axes object to draw the line on.
    lnL: array of ln(L) values.
    lnchi: array of ln(chi) values.
    dlnchi: array of uncertainties in ln(chi) values.
    color: color of the fitted line.
    
    LLM-assisted
    ------------
    Tool: Claude (October 2026)
    Role: Made the plotting part
    """
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
        linestyle="--", lw=1.5, color=color, zorder=2
    )
    return slope, d_slope, intercept, d_intercept

def interpolate_at(T, y, t_c, npts=4, deg=2):
    """
    Local polynomial fit through the npts grid points nearest t_c, evaluated at t_c.
    T: array of temperature values.
    y: array of observable values corresponding to T.
    t_c: critical temperature at which to evaluate the polynomial.
    npts: number of nearest points to use for the polynomial fit.
    deg: degree of the polynomial to fit.
    Returns the value of the polynomial at t_c.
    """
    T, y = np.asarray(T), np.asarray(y)
    idx = np.sort(np.argsort(np.abs(T - t_c))[:npts])
    p = np.polyfit(T[idx] - t_c, y[idx], deg)  # centered for conditioning
    return p[-1]  # polynomial value at T - t_c = 0

def jackknife_interpolate(T, theta, reps, t_c, npts=4, deg=2):
    """
    Value and jackknife error of a curve interpolated at t_c.
    T: (n_T,) array of temperatures
    theta: (n_T,) array of full-sample observable values
    reps: (n_bins, n_T) array of leave-one-out replicates of theta
    t_c: temperature at which to interpolate
    npts: number of nearest points to use for the polynomial fit
    deg: degree of the polynomial to fit
    Returns:
    val: interpolated value at t_c
    err: jackknife error estimate of the interpolated value
    """
    if not (np.min(T) <= t_c <= np.max(T)):
        raise ValueError(f"t_c={t_c} lies outside the temperature range [{np.min(T)}, {np.max(T)}]")
    if npts <= deg:
        raise ValueError("npts must be larger than deg")

    val = interpolate_at(T, theta, t_c, npts, deg)
    vals_i = np.array([interpolate_at(T, r, t_c, npts, deg) for r in reps])
    err = np.sqrt((len(reps) - 1) * np.var(vals_i, ddof=0))
    return val, err

def log_scaling(args, parameter_at_tc, error_at_tc):
    """
    Fit a log-log line to the observable at t_c vs L.
    args: command-line arguments containing L values.
    parameter_at_tc: size: (L, sigma) array of observable at
        t_c for each lattice size L.
    error_at_tc: size: (L, sigma) array of jackknife errors
        for the observable at t_c.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    lnL = np.log(np.asarray(args.L, dtype=float))
    n_sigma = parameter_at_tc.shape[1]

    for num in range(n_sigma):
        color = cm.viridis(num / n_sigma)
        lnchi = np.log(parameter_at_tc[:, num] / args.N)
        dlnchi = error_at_tc[:, num] / parameter_at_tc[:, num]

        slope, d_slope, intercept, d_intercept = fit_loglog_line(
            ax, lnL, lnchi, dlnchi, color=color
        )
        ax.errorbar(
            lnL, lnchi, yerr=dlnchi, fmt="o", color=color,
            markersize=6, capsize=3, zorder=3,
            label=f"σ = {args.sigmas[num]}, slope = {slope:.2f} ± {d_slope:.2f}",
        )

    ax.set_xlabel(r"$\ln(L)$", fontsize=12)
    ax.set_ylabel(r"$\ln(\theta(T^*))\,/\,a$", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.legend(frameon=False, fontsize=10)

    plt.tight_layout()
    plt.savefig(args.save_path, bbox_inches="tight")
    if args.show:
        plt.show()

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot a metric vs temperature for given lattice sizes and solver."
    )
    p.add_argument("--show", action="store_true",
                   help="Display the plot")

    # Required inputs
    p.add_argument("-f", "--file-path", required=True,
                   help="CSV file(s) with data, one per lattice size")
    p.add_argument("-L", "--lattice-sizes", dest="L", type=int, nargs="+", required=True,
                   help="Lattice sizes (e.g. 8 16 32 48)")
    p.add_argument("--sigmas", type=float, nargs='+',
                   help="Number of sigmas to plot")
    p.add_argument("--sigma_index", type=int, nargs='+',
                   help="The sigmas indexes")
    p.add_argument("-N", "--num-spins", dest="N", type=int, nargs="+", required=True,
                   help="Spins per lattice (e.g. 64 256 1024 2304)")
    p.add_argument("--op", required=True,
                   help="Order parameter to plot (e.g. raw, simple_ann)")

    # Output / analysis
    p.add_argument("-s", "--save-path", default="plot.pdf",
                   help="Path to save the plot")
    p.add_argument("-m", "--metric",
                   help="Metric to plot (e.g. direct, binder, susceptibility)")
    p.add_argument("--tc", dest="t_c", type=float, nargs="+", required=True,
                   help="Critical temperature for the Ising model")
    p.add_argument("--bins", dest="bins", type=int, default=20,
                   help="Number of bins for jackknife resampling")
    p.add_argument("--npts", type=int, default=4,
                   help="Number of points to use for interpolation at T_c")
    p.add_argument("--deg", type=int, default=2,
                   help="Degree of polynomial for interpolation at T_c")

    # Axis limits
    p.add_argument("-a", "--a", type=float, default=1.0,
                   help="Scaling for the simple ANN order parameter")
    p.add_argument("-b", "--bias", type=float, default=0.0,
                   help="Bias for the simple ANN order parameter")
    
    names = {"min": "Minimum", "max": "Maximum"}
    for axis in ("x", "y"):
        for lim, word in names.items():
            p.add_argument(f"--{axis}{lim}", type=float,
                           help=f"{word} {axis}-axis value for the plot")

    args = p.parse_args()
    if not (len(args.L) == len(args.N)):
        p.error("--lattice-sizes and --num-spins must have the same length")
    if not (len(args.sigmas) == len(args.t_c) == len(args.sigma_index)):
        p.error("--sigmas, --sigma_index and --tc must have the same length")
    return args

if __name__ == "__main__":
    args = parse_args()

    parameter_data = []
    parameter_errors = []
    replicates = []

    for i, L in enumerate(args.L):
        parameters_L = []
        parameter_errors_L = []

        for j, sigma in enumerate(args.sigma_index):

            temps = read_temperatures(args.file_path + f"/temp_si-{sigma}.txt")

            file_path = args.file_path + f"/magnetization_lb_sigma-{sigma}_L{L}.csv"

            df = pd.read_csv(file_path, index_col=0)
            if args.op == "simple_ann":
                input_array = simple_ann(args, df)
            else:
                input_array = df.abs().to_numpy()

            stat = get_metric(args.metric, args.N[i], temps)

            parameter, parameter_error, theta_i = block_jackknife(input_array, stat, num_bins=args.bins, return_replicates=True)
            
            val, err = jackknife_interpolate(temps, parameter, theta_i, args.t_c[j], npts=args.npts, deg=args.deg)
            parameters_L.append(val)
            parameter_errors_L.append(err)


        parameter_data.append(parameters_L)
        parameter_errors.append(parameter_errors_L)

    parameter_data = np.array(parameter_data)  # shape: (num_L, num_sigmas)
    parameter_errors = np.array(parameter_errors)  # shape: (num_L, num_sigmas)

    print(f"Num L's: {parameter_data.shape[0]}, Num sigmas: {parameter_data.shape[1]}")

    log_scaling(args, parameter_data, parameter_errors)





