import argparse
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

def block_jackknife(x, stat, num_bins=20, return_replicates=False):
    """
    Returns the jackknife estimate of the statistic and its error.

    x: data of size (n_samples, n_T) 
    stat: function of moment dict -> (n_T,) array 
    (Ex of stat: stat = lambda M: 1.0 - M[4] / (3.0 * M[2]**2), 
    the Binder cumulant)
    
    
    LLM-assisted
    ------------
    Tool: Claude (October 2026)
    Role: Generated the block jackknife function 
    for computing statistics and errors from data.
    """
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
    """
    Crossing of two curves: polynomial fit in window, then roots of the difference.
    T: (n_T,) array of temperatures
    U1, U2: (n_T,) arrays of observables
    window: (T_min, T_max) tuple for the fitting range
    deg: degree of the polynomial fit
    Returns the crossing temperature or np.nan if no crossing is found.

    LLM-assisted
    ------------
    Tool: Claude (October 2026)
    Role: Generated the crossing_point function for finding the intersection of two curves.
    """
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
    """
    Crossing temperature and jackknife error for two independent data sets.
    T: (n_T,) array of temperatures
    th1, th2: (n_T,) arrays of observables for the two data sets
    rep1, rep2: (n_bins, n_T) arrays of jackknife replicates for the two data sets
    window: (T_min, T_max) tuple for the fitting range
    deg: degree of the polynomial fit
    Returns the crossing temperature and its jackknife error.

    LLM-assisted
    ------------
    Tool: Claude (October 2026)
    Role: Generated the jackknife_crossing function for 
    estimating the crossing temperature and its error.
    """
    Tx = crossing_point(T, th1, th2, window, deg)
    B1, B2 = rep1.shape[0], rep2.shape[0]
    c1 = np.array([crossing_point(T, r, th2, window, deg) for r in rep1])
    c2 = np.array([crossing_point(T, th1, r, window, deg) for r in rep2])
    var = (B1 - 1) * np.var(c1) + (B2 - 1) * np.var(c2)
    return Tx, np.sqrt(var)

def read_temperatures(temp_file):
    """
    Reads temperatures from a file and returns them as a numpy array.
    """
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
    """Applies a simple artificial neural network transformation to the data."""
    return sigmoid(args.a * np.abs(df.values) + args.bias)

def get_metric(name, N, T):
    metrics = {
        "direct":         lambda M: M[1],
        "susceptibility": lambda M: (M[2] - M[1]**2) * N / T,
        "binder":         lambda M: 1 - M[4] / (3 * M[2]**2),
    }
    return metrics[name]

def plot_metric(args, parameter_data, parameter_error, temps, Tmean=None, Terr=None):
    """
    Plots a given metric (e.g., magnetization, susceptibility) against temperature 
    for different lattice sizes.

    Parameters:
    - args: Command-line arguments containing plot settings.
    - parameter_data: List of arrays containing the metric data for each lattice size.
    - parameter_error: List of arrays containing the error for each metric data point.
    - temps: Array of temperature values.
    - Tmean: Optional vertical line for the theoretical or estimated critical temperature.
    - Terr: Optional error in the estimated critical temperature for a shaded region.

    LLM-assisted
    ------------
    Tool: Claude (October 2026)
    Role: Helped with the aesthetics of the plot.    
    """
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

    if Tmean is not None and Terr is not None:
        label_=fr"$T^* = {Tmean:.4f} \pm {Terr:.4f}$"
    elif Tmean is not None:
        label_=fr"$T^* = {Tmean:.4f}$"

    if Tmean is not None:
        ax.axvline(
            Tmean,
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
            label=label_,
            zorder=2,
        )
    if Tmean is not None and Terr is not None:
        ax.fill_between(
            [Tmean - Terr, Tmean + Terr],
            ax.get_ylim()[0], ax.get_ylim()[1],
            color="#d62728",
            alpha=0.2,
            zorder=2,
        )

    # Labels and Aesthetics
    ax.set_xlabel(r"Temperature ($T$)", fontsize=12, labelpad=8)
    ax.set_ylabel(args.metric, fontsize=12, labelpad=8)

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.legend(frameon=False, fontsize=10)

    ax.set_xlim(args.xmin, args.xmax)
    ax.set_ylim(args.ymin, args.ymax)

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
    p.add_argument("-f", "--file-path", nargs="+", required=True,
                   help="CSV file(s) with data, one per lattice size")
    p.add_argument("-t", "--temp-file", required=True,
                   help="CSV file with temperature data")
    p.add_argument("-L", "--lattice-sizes", dest="L", type=int, nargs="+", required=True,
                   help="Lattice sizes (e.g. 8 16 32 48)")
    p.add_argument("-N", "--num-spins", dest="N", type=int, nargs="+", required=True,
                   help="Spins per lattice (e.g. 64 256 1024 2304)")
    p.add_argument("--op", required=True,
                   help="Order parameter to plot (e.g. direct, simple_ann)")

    # Output / analysis
    p.add_argument("-s", "--save-path", default="plot.pdf",
                   help="Path to save the plot")
    p.add_argument("-m", "--metric",
                   help="Metric to plot (e.g. direct, binder, susceptibility)")
    p.add_argument("--tc", dest="t_c", type=float,
                   help="Critical temperature for the Ising model")
    p.add_argument("--bins", dest="bins", type=int, default=20,
                   help="Number of bins for jackknife resampling")
    p.add_argument("--cw", type=float, nargs=2, metavar=("LO", "HI"),
                   help="T window around the Binder crossing, e.g. 2.2 2.35")
    p.add_argument("--cd", type=int, default=3,
                   help="Polynomial degree for the crossing fits")

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
    if not (len(args.file_path) == len(args.L) == len(args.N)):
        p.error("--file-path, --lattice-sizes and --num-spins must have the same length")
    return args

if __name__ == "__main__":
    args = parse_args()

    parameter_data = []
    parameter_errors = []
    replicates = []

    temps = read_temperatures(args.temp_file)

    for i, L in enumerate(args.L):

        file_path = args.file_path[i]

        df = pd.read_csv(file_path, index_col=0)
        if args.op == "simple_ann":
            input_array = simple_ann(args, df)
        else:
            input_array = df.abs().to_numpy()

        stat = get_metric(args.metric, args.N[i], temps)

        parameter, parameter_error, theta_i = block_jackknife(input_array, stat, num_bins=args.bins, return_replicates=True)

        parameter_data.append(parameter)
        parameter_errors.append(parameter_error)
        replicates.append(theta_i)

    if args.metric == "binder" and args.cw is not None:
        Ts, dTs = [], []
        for i, j in itertools.combinations(range(len(args.L)), 2):
            Tx, dTx = jackknife_crossing(
                temps, parameter_data[i], replicates[i],
                parameter_data[j], replicates[j],
                args.cw, args.cd,
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
    else: 
        plot_metric(args, parameter_data, parameter_errors, temps, Tmean=args.t_c, Terr=None)




