import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import file_reader
import FSS
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def simple_ann(args, df): 
    return sigmoid(args.a * np.abs(df.values) + args.bias)

def mag_suceptibility(input_array, num_spins, temp_file):
    temperatures = []
    with open(temp_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                temperatures.append(float(parts[1]))

    avg_mag = np.mean(np.abs(input_array), axis=0)
    avg_mag_sq = np.mean(input_array**2, axis=0)
    susceptibility = (avg_mag_sq - avg_mag**2) * num_spins / np.array(temperatures)

    return susceptibility

def binder_cumulant(input_array):
    avg_mag_quad = np.mean(input_array**4, axis=0)
    avg_mag_sq = np.mean(input_array**2, axis=0)
    binder = 1 - avg_mag_quad / (3 * avg_mag_sq**2)

    return binder

def mag_susceptibility_with_error(input_array, num_spins, temp_file, num_bins=20):
    """
    Computes magnetic susceptibility and its uncertainty using Jackknife binning.
    
    input_array: 2D numpy array of shape (num_snapshots, num_temperatures)
    """
    # Load temperatures
    temperatures = []
    with open(temp_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                temperatures.append(float(parts[1]))
    
    temps = np.array(temperatures)
    n_samples = input_array.shape[0]
    
    avg_mag = np.mean(np.abs(input_array), axis=0)
    avg_mag_sq = np.mean(input_array**2, axis=0)
    susceptibility = (avg_mag_sq - avg_mag**2) * num_spins / temps
    
    bin_size = n_samples // num_bins
    jackknife_chi = np.zeros((num_bins, input_array.shape[1]))
    
    for i in range(num_bins):
        mask = np.ones(n_samples, dtype=bool)
        mask[i * bin_size : (i + 1) * bin_size] = False
        
        m_subset = input_array[mask]
        
        m_avg_sub = np.mean(np.abs(m_subset), axis=0)
        m_sq_avg_sub = np.mean(m_subset**2, axis=0)
        
        jackknife_chi[i] = (m_sq_avg_sub - m_avg_sub**2) * num_spins / temps
        
    chi_error = np.sqrt((num_bins - 1) * np.var(jackknife_chi, axis=0, ddof=0))

    return susceptibility, chi_error

def binder_cumulant_with_error(input_array, num_bins=20):
    """
    Computes the Binder cumulant and its uncertainty using Jackknife binning.
    
    input_array: 2D numpy array of shape (num_snapshots, num_temperatures)
    num_bins: Number of Jackknife blocks (default: 20)
    """
    n_samples = input_array.shape[0]
    
    # 1. Full dataset estimation
    avg_mag_quad = np.mean(input_array**4, axis=0)
    avg_mag_sq = np.mean(input_array**2, axis=0)
    binder = 1.0 - avg_mag_quad / (3.0 * (avg_mag_sq**2))
    
    # 2. Jackknife Resampling
    bin_size = n_samples // num_bins
    jackknife_binder = np.zeros((num_bins, input_array.shape[1]))
    
    for i in range(num_bins):
        # Leave out the i-th block
        mask = np.ones(n_samples, dtype=bool)
        mask[i * bin_size : (i + 1) * bin_size] = False
        
        m_subset = input_array[mask]
        
        m4_sub = np.mean(m_subset**4, axis=0)
        m2_sub = np.mean(m_subset**2, axis=0)
        
        jackknife_binder[i] = 1.0 - m4_sub / (3.0 * (m2_sub**2))
        
    # 3. Calculate Jackknife standard error
    binder_error = np.sqrt((num_bins - 1) * np.var(jackknife_binder, axis=0, ddof=0))

    return binder, binder_error

def plot_metric(args, parameter_data, parameter_std_data, temp_file, plot_dir):

    plt.style.use("seaborn-b_8-whitegrid" if "seaborn-b_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    for i, L in enumerate(args.L):
        # Read temperatures using context manager
        temperatures = []
        with open(temp_file, "r") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    temperatures.append(float(parts[1]))

        # Enhanced plotting styling
        line = ax.plot(
            temperatures,
            parameter_data[i],
            marker="o",
            markersize=5,
            linewidth=1.8,
            linestyle="-",
            label=f"$L = {L}$",
        )
        color = line[0].get_color()

        # Error shading matching the line color
        ax.fill_between(
            temperatures,
            parameter_data[i] - parameter_std_data[i],
            parameter_data[i] + parameter_std_data[i],
            color=color,
            alpha=0.18,
            linewidth=0,
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

    # Fine-tune limits and grid
    # ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.legend(frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=10)

    # Save and show
    if "lrim" in args.ising_solver:
        save_path = plot_dir + f"/{args.ising_solver}_si_{args.sigma_index}.pdf"
    elif "lb" in args.ising_solver:
        save_path = plot_dir + f"/{args.ising_solver}_si_{args.sigma_index}.pdf"
    else: 
        save_path = plot_dir + f"/{args.ising_solver}.pdf"

    plt.tight_layout()
    # plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)  # Close the figure to free memory

def plot_fss_order_parameter(args, parameter_data, parameter_std_data, temp_file, plot_dir): 
    
    # Read temperatures using context manager
    temperatures = []
    with open(temp_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                temperatures.append(float(parts[1]))

    if args.upper == 0: 
        args.upper = len(temperatures) - 1

    ret, x, y, dy, x_true, y_true, dy_true = FSS.finite_size_scaling(
        args.L, np.array(temperatures), np.array(parameter_data), np.array(parameter_std_data),
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

            ax.plot(
                xi, yi,
                linestyle="-", marker="o", markersize=3,
                markerfacecolor="white", markeredgecolor=colors[i], markeredgewidth=1.2,
                lw=1.4, color=colors[i], label=f"L = {L}", zorder=3
            )
            ax.fill_between(
                xi, yi - dyi, yi + dyi,
                color=colors[i], alpha=0.25, linewidth=0, zorder=2
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
        default=0.0,
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
    parameter_std_data = []

    if args.order_parameter == "simple_ann":
        target_observable = "magnetization"
    else: 
        target_observable = args.order_parameter

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
            parameter, parameter_std = mag_susceptibility_with_error(input_array, num_spins_list[i], temp_file)
        elif args.metric == "binder":
            parameter, parameter_std = binder_cumulant_with_error(input_array)
        else: 
            parameter = input_array.mean(axis=0)
            parameter_std = input_array.std(axis=0)

        parameter_data.append(parameter)
        parameter_std_data.append(parameter_std)

    plot_metric(args, parameter_data, parameter_std_data, temp_file, plot_dir)

    # plot_fss_order_parameter(args, parameter_data, parameter_std_data, temp_file, plot_dir)


