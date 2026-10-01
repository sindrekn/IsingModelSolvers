import file_reader
import argparse
import pandas as pd

def save_magnetization(args, dir_path, save_path ,num_spins):
    temp_data = {}
    for temp in range(args.num_temps):
        temp_values = []
        for run_index in range(args.num_runs):
            filename = f"{dir_path}/temp_{temp}/Snapshots_{run_index}.dat"
            temp_values.extend(file_reader.return_mag(filename, num_spins))
        temp_data[f"temp_{temp}"] = temp_values

    df = pd.DataFrame(temp_data)
    df.index = [f"snapshot_{i}" for i in range(len(df))]

    df.to_csv(save_path if save_path else f"{dir_path}/avg_magnetization.csv", index=True)

def save_regmag(args, dir_path, save_path, num_spins, A):
    if num_spins % (A ** 2) != 0:
        raise ValueError(f"num_spins ({num_spins}) must be divisible by A^2 ({A ** 2}) for regmag calculation.")
    temp_data = {}
    for temp in range(args.num_temps):
        temp_values = []
        for run_index in range(args.num_runs):
            filename = f"{dir_path}/temp_{temp}/Snapshots_{run_index}.dat"
            temp_values.extend(file_reader.return_regmag(filename, num_spins, A, args.v0, args.w, args.b))
        temp_data[f"temp_{temp}"] = temp_values

    df = pd.DataFrame(temp_data)
    df.index = [f"snapshot_{i}" for i in range(len(df))]

    df.to_csv(save_path if save_path else f"{dir_path}/avg_regmag.csv", index=True)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot average magnetization vs temperature for a given lattice size and solver."
    )
    parser.add_argument(
        "-op", "--order-parameter",
        dest="order_parameter",
        type=str,
        required=True,
        help="Order parameter to analyze (e.g., magnetization, regmag)"
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
        "-si", "--sigma-index",
        dest="sigma_index",
        type=str,
        required=False,
        help="Index for sigma value (only for lrim solver)"
    )
    parser.add_argument(
        "-A", "--A",
        dest="A",
        type=int,
        nargs='+',
        required=False,
        help="Side length of regions in regmag calculation"
    )
    parser.add_argument(
        "-v0", "--v0",
        dest="v0",
        type=float,
        default=0.0,
        required=False,
        help="Bias in sigmoid function for regmag calculation"
    )
    parser.add_argument(
        "-w", "--w",
        dest="w",
        type=float,
        default=1.0,
        required=False,
        help="Scaling of sigmoid function for regmag calculation"
    )
    parser.add_argument(
        "-b", "--b",
        dest="b",
        type=float,
        default=0.0,
        required=False,
        help="Bias in ReLu function for regmag calculation"
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

    if "cubic" in args.ising_solver:
        num_spins_list = [L**3 for L in args.L]
    else:
        num_spins_list = [L**2 for L in args.L]

    target_observable = args.order_parameter

    for L, num_spins, A in zip(args.L, num_spins_list, args.A):
        if "lrim" in args.ising_solver:
            dir_path = base_dir + "/" + args.ising_solver + "/test2/L" + str(L) + "/sigma_" + str(args.sigma_index)
            save_path = f"/home/sindrekampennesheim/Documents/PhD/Optimizing/IsingModelSolver/benchmarks/phase_detection/observables/test2/{target_observable}_{args.ising_solver}_L{L}_sigma_{args.sigma_index}.csv"
        elif "lb" in args.ising_solver:
            dir_path = base_dir + "/" + args.ising_solver + "/test2/L" + str(L) + "/sigma_" + str(args.sigma_index)
            save_path = f"/home/sindrekampennesheim/Documents/PhD/Optimizing/IsingModelSolver/benchmarks/phase_detection/observables/test2/{target_observable}_{args.ising_solver}_L{L}_sigma_{args.sigma_index}.csv"
        else:
            dir_path = base_dir + "/" + args.ising_solver + "/test2/L" + str(L)
            save_path = f"/home/sindrekampennesheim/Documents/PhD/Optimizing/IsingModelSolver/benchmarks/phase_detection/observables/test2/{target_observable}_{args.ising_solver}_L{L}.csv"

        if args.order_parameter == "regmag":
            save_regmag(args, dir_path, save_path=save_path, num_spins=num_spins, A=A)
        elif args.order_parameter == "magnetization":
            save_magnetization(args, dir_path, save_path=save_path, num_spins=num_spins)
        else: 
            raise ValueError(f"Unknown order parameter: {args.order_parameter}")

        print(f"Saved average {target_observable} data for L={L} to {save_path}")

    


