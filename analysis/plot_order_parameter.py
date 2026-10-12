import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import file_reader
import FSS
import pandas as pd

def block_jackknife(x, stat, num_bins=20):
    """x: (n_samples, n_T). stat: function of moment dict -> (n_T,) array."""
    n = (x.shape[0] // num_bins) * num_bins
    b = n // num_bins
    x = np.abs(x[:n])
    S = {k: (x**k).reshape(num_bins, b, -1).sum(axis=1) for k in (1, 2, 4)}
    full = {k: S[k].sum(axis=0) / n for k in S}
    loo  = {k: (S[k].sum(axis=0) - S[k]) / (n - b) for k in S}
    theta, theta_i = stat(full), stat(loo)
    err = np.sqrt((num_bins - 1) * np.var(theta_i, axis=0, ddof=0))
    return theta, err

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

def plot_metric(args, parameter_data, parameter_error, temps, plot_dir):

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

    # Labels and Aesthetics
    ax.set_xlabel(r"Temperature ($T$)", fontsize=12, labelpad=8)
    ax.set_ylabel(args.metric, fontsize=12, labelpad=8)
    ax.set_title(
        f"{args.metric} vs Temperature ({args.ising_solver.upper()})",
        fontsize=13,
        pad=12,
        weight="semibold",
    )

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.legend(frameon=False, fontsize=10)

    # Save and show
    if "lrim" in args.ising_solver or "lb" in args.ising_solver:
        save_path = plot_dir + f"/{args.ising_solver}_si_{args.sigma_index}.pdf"
    else:
        save_path = plot_dir + f"/{args.ising_solver}.pdf"

    plt.tight_layout()
    # plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def plot_fss_order_parameter(args, parameter_data, parameter_error, temps, plot_dir): 
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

    # Save and show
    if "lrim" in args.ising_solver:
        save_path = plot_dir + f"/fss_{args.ising_solver}_si_{args.sigma_index}.pdf"
    elif "lb" in args.ising_solver:
        save_path = plot_dir + f"/fss_{args.ising_solver}_si_{args.sigma_index}.pdf"
    else: 
        save_path = plot_dir + f"/fss_{args.ising_solver}.pdf"

    plt.tight_layout()
    # plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)  # Close the figure to free memory

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot average magnetization vs temperature for a given lattice size and solver."
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
        "-I", "--ising-solver",
        dest="ising_solver",
        type=str,
        required=True,
        help="Name of the Ising solver (used to build the output directory path)"
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
        default=2.27,
        required=False,
        help="Critical temperature for the Ising model"
    )
    parser.add_argument(
        "-si", "--sigma-index",
        dest="sigma_index",
        type=str,
        required=False,
        help="Index for sigma value (only for lrim solver)"
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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dir_path = "/home/sindrekampennesheim/Documents/PhD/Optimizing/IsingModelSolver/benchmarks/phase_detection/"
    plot_dir = dir_path + f"plots/test2/{args.order_parameter}"

    if "wolff" in args.ising_solver: 
        temp_file = dir_path + f"cluster/{args.ising_solver}/test2/L{args.L[0]}/temp_index.txt"
    elif "lb" in args.ising_solver: 
        temp_file = dir_path + f"cluster/{args.ising_solver}/test2/L{args.L[0]}/temp_index.txt"
    else: 
        temp_file = dir_path + f"metropolis/{args.ising_solver}/test2/L{args.L[0]}/temp_index.txt"

    if "cubic" in args.ising_solver:
        num_spins_list = [L**3 for L in args.L]
    else:
        num_spins_list = [L**2 for L in args.L]

    parameter_data = []
    parameter_errors = []

    if args.order_parameter == "simple_ann":
        target_observable = "magnetization"
    else: 
        target_observable = args.order_parameter

    temps = read_temperatures(temp_file)   # read once, not per L

    for i, L in enumerate(args.L):

        if "lrim" in args.ising_solver:
            file_path = dir_path + f"/observables/test2/{target_observable}_{args.ising_solver}_L{L}_sigma_{args.sigma_index}.csv"
        elif "lb" in args.ising_solver:
            file_path = dir_path + f"/observables/test2/{target_observable}_{args.ising_solver}_L{L}_sigma_{args.sigma_index}.csv"
        else:
            file_path = dir_path + f"/observables/test2/{target_observable}_{args.ising_solver}_L{L}.csv"

        df = pd.read_csv(file_path, index_col=0)
        if args.order_parameter == "simple_ann":
            input_array = simple_ann(args, df)
        else:
            input_array = df.abs().to_numpy()

        if args.metric == "susceptibility":
            stat = lambda M: (M[2] - M[1]**2) * num_spins_list[i] / temps
        elif args.metric == "binder":
            stat = lambda M: 1.0 - M[4] / (3.0 * M[2]**2)
        else:
            stat = lambda M: M[1]

        parameter, parameter_error = block_jackknife(input_array, stat, num_bins=args.bins)

        parameter_data.append(parameter)
        parameter_errors.append(parameter_error)

    plot_metric(args, parameter_data, parameter_errors, temps, plot_dir)

    plot_fss_order_parameter(args, parameter_data, parameter_errors, temps, plot_dir)


