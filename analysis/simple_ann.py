import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import file_reader
import FSS
from pathlib import Path


def plot_simple_ann(args, base_dir, plot_dir):
    """Plots the simple avg magnitazation based ann with error bars for a given lattice size L,

    number of temperatures, and number of runs.
    """
    # Set modern style, font scaling, and ideal dimensions for papers/reports (8x5 or 7x5)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    base_path = Path(base_dir)

    for L, num_spins in zip(args.L, args.num_spins):
        if args.ising_solver in ["lrim", "lb"]:
            dir_path = base_path / args.ising_solver / "test1" / f"L{L}" / f"sigma_{args.sigma_index}"
        else:
            dir_path = base_path / args.ising_solver / "test1" / f"L{L}"

        parameter = np.zeros(args.num_temps)
        parameter_std = np.zeros(args.num_temps)

        for temp_index in range(args.num_temps):
            avg_para = []
            for run_index in range(args.num_runs):
                filename = dir_path / f"temp_{temp_index}" / f"Snapshots_{run_index}.dat"
                filename = str(filename)  # Convert Path object to string for compatibility
                avg_para.append(file_reader.return_simple_ann(filename, num_spins, args.ymin, args.ymax))

            parameter[temp_index] = np.mean(avg_para)
            parameter_std[temp_index] = np.std(avg_para)

        # Read temperatures using context manager
        temp_file = dir_path / "temp_index.txt"
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
            parameter,
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
            parameter - parameter_std,
            parameter + parameter_std,
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
    ax.set_ylabel(r"Simple ANN", fontsize=12, labelpad=8)
    ax.set_title(
        f"Simple ANN vs Temperature ({args.ising_solver.upper()})",
        fontsize=13,
        pad=12,
        weight="semibold",
    )

    # Fine-tune limits and grid
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.legend(frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=10)

    # Save and show
    if args.ising_solver in ["lrim", "lb"]:
        save_path = plot_dir + f"/{args.ising_solver}_si_{args.sigma_index}.pdf"
    else: 
        save_path = plot_dir + f"/{args.ising_solver}.pdf"

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    # plt.show()
    plt.close(fig)  # Close the figure to free memory

def plot_fss_simple_ann(args, base_dir, plot_dir): 
    base_path = Path(base_dir)
    parameter_data = []
    parameter_std_data = []

    for L, num_spins in zip(args.L, args.num_spins):
        if args.ising_solver in ["lrim", "lb"]:
            dir_path = base_path / args.ising_solver / "test1" / f"L{L}" / f"sigma_{args.sigma_index}"
        else:
            dir_path = base_path / args.ising_solver / "test1" / f"L{L}"

        parameter = np.zeros(args.num_temps)
        parameter_std = np.zeros(args.num_temps)

        for temp_index in range(args.num_temps):
            avg_para = []
            for run_index in range(args.num_runs):
                filename = dir_path / f"temp_{temp_index}" / f"Snapshots_{run_index}.dat"
                filename = str(filename)  # Convert Path object to string for compatibility
                avg_para.append(file_reader.return_simple_ann(filename, num_spins, args.ymin, args.ymax))

            parameter[temp_index] = np.mean(avg_para)
            parameter_std[temp_index] = np.std(avg_para)

        parameter_data.append(parameter)
        parameter_std_data.append(parameter_std)

    # Read temperatures using context manager
    temp_file = dir_path / "temp_index.txt"
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
        ax.set_ylabel(r"$m \, L^{\beta/\nu}$")
        ax.set_title(title)
        ax.legend(frameon=False)
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)

    ncols = 2 if args.ts else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5.5))
    if ncols == 1:
        axes = [axes]

    plot_collapse(
        axes[0], x, y, dy,
        rf"FSS collapse (fitted): $T_c={ret['rho']:.4f}$, "
        rf"$\nu={ret['nu']:.3f}$, $\beta={ret['zeta']:.3f}$"
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
    if args.ising_solver in ["lrim", "lb"]:
        save_path = plot_dir + f"/fss_{args.ising_solver}_si_{args.sigma_index}.pdf"
    else: 
        save_path = plot_dir + f"/fss_{args.ising_solver}.pdf"

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    # plt.show()
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
        "-N", "--number-of-spins",
        dest="num_spins",
        type=int,
        nargs='+',
        required=True,
        help="Number of spins in the lattices (e.g., 64 256 1024 4096)"
    )
    parser.add_argument(
        "-I", "--ising-solver",
        dest="ising_solver",
        type=str,
        required=True,
        help="Name of the Ising solver (used to build the output directory path)"
    )
    parser.add_argument(
        "-t", "--num-temps",
        dest="num_temps",
        type=int,
        required=True,
        help="Number of temperature values to consider"
    )
    parser.add_argument(
        "-r", "--num-runs",
        dest="num_runs",
        type=int,
        required=True,
        help="Number of runs to average over for each temperature"
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
        "-ymin", "--ymin",
        dest="ymin",
        type=float,
        required=True,
        help="Minimum value for the y-axis range"
    )
    parser.add_argument(
        "-ymax", "--ymax",
        dest="ymax",
        type=float,
        required=True,
        help="Maximum value for the y-axis range"
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
    if "wolff" in args.ising_solver: 
        base_dir = "/home/sindrekampennesheim/Documents/PhD/Optimizing/IsingModelSolver/benchmarks/phase_detection/cluster"
    elif "lrim" in args.ising_solver:
        base_dir = "/home/sindrekampennesheim/Documents/PhD/Optimizing/IsingModelSolver/benchmarks/phase_detection/cluster"
    else: 
        base_dir = "/home/sindrekampennesheim/Documents/PhD/Optimizing/IsingModelSolver/benchmarks/phase_detection/metropolis"
    plot_dir = "/home/sindrekampennesheim/Documents/PhD/Optimizing/IsingModelSolver/benchmarks/phase_detection/plots/test1/simple_ann"

    plot_simple_ann(args, base_dir, plot_dir)

    plot_fss_simple_ann(args, base_dir, plot_dir)

    print(f"Simple ann analysis of {args.ising_solver} completed successfully.")


