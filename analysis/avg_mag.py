import argparse
import matplotlib.pyplot as plt
import numpy as np
import file_reader


def plot_avg_magnetization(num_spins, num_temps, num_runs, base_dir, t_c=None):
    """
    Plots the average magnetization with error bars for a given lattice size L,
    number of temperatures, and number of runs.

    Parameters:
    - L: int, the lattice size
    - num_temps: int, the number of temperature values to consider
    - num_runs: int, the number of runs to average over for each temperature
    - base_dir: str, the base directory containing the snapshot files
    """
    magnetization = np.zeros(num_temps)
    magnetization_std = np.zeros(num_temps)

    for temp_index in range(num_temps):
        avg_mags = []
        for run_index in range(num_runs): 
            filename = base_dir + f"/temp_{temp_index}/Snapshots_{run_index}.dat"
            avg_mags.append(file_reader.return_avg_mag(filename, num_spins))
        magnetization[temp_index] = np.mean(np.abs(avg_mags))
        magnetization_std[temp_index] = np.std(np.abs(avg_mags))    

    temp_file = base_dir + "/temp_index.txt"
    temperatures = []

    with open(temp_file, 'r') as f:
        # Skip the first line (header)
        next(f)
        for line in f.readlines():
            temperatures.append(float(line.strip().split()[1]))  # Extract the first column (temperature values)
        f.close()

    if t_c is not None:
        plt.axvline(t_c, color='r', linestyle='--', label=f'Tc = {t_c}')

    # Plot the average magnetization with error bars
    plt.errorbar(temperatures, magnetization, yerr=magnetization_std, fmt='o', capsize=5, label='Average Magnetization')
    plt.xlabel('Temperature')
    plt.ylabel('Average Magnetization')
    plt.title('Average Magnetization vs Temperature')
    plt.legend()
    plt.grid()
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot average magnetization vs temperature for a given lattice size and solver."
    )
    parser.add_argument(
        "-L", "--lattice-size",
        dest="L",
        type=int,
        required=True,
        help="Size of the lattice (L)"
    )
    parser.add_argument(
        "-N", "--number-of-spins",
        dest="num_spins",
        type=int,
        required=True,
        help="Number of spins in the lattice"
    )
    parser.add_argument(
        "-s", "--ising-solver",
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
        required=False,
        help="Critical temperature for the Ising model"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    base_dir = "/home/sindrekampennesheim/Documents/PhD/Optimizing/Output/IsingSolver/" + args.ising_solver + "/test1/L" + str(args.L);

    plot_avg_magnetization(args.num_spins, args.num_temps, args.num_runs, base_dir, args.t_c)


