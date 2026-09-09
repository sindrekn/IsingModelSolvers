import matplotlib.pyplot as plt
import numpy as np
import file_reader


def plot_avg_magnetization(L, num_temps, num_runs, base_dir):
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
            avg_mags.append(file_reader.return_avg_mag(filename, L))
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

    # Plot the average magnetization with error bars
    plt.errorbar(temperatures, magnetization, yerr=magnetization_std, fmt='o', capsize=5, label='Average Magnetization')
    plt.xlabel('Temperature')
    plt.ylabel('Average Magnetization')
    plt.title('Average Magnetization vs Temperature')
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
if __name__ == "__main__":
    base_dir = '/home/sindrekampennesheim/Documents/PhD/Optimizing/Output/SimulatedAnnealing/2D_nn/ferromagnetic/metropolis/L16'

    plot_avg_magnetization(L=16, num_temps=10, num_runs=50, base_dir=base_dir)


