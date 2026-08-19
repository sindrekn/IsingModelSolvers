# Long-Range Ising Model (LRIM) Simulation

## Overview

The long-range Ising model (LRIM) extends the nearest-neighbor 2D Ising model by coupling every pair of spins with an interaction strength that decays as a power law with distance, rather than restricting interactions to nearest neighbors. Spin configurations are sampled using the Metropolis algorithm, where individual spins are flipped according to the standard Metropolis acceptance criterion based on the resulting energy change. To efficiently cover a range of temperatures, simulations are run as embarrassingly parallel jobs, with each independent run assigned to a temperature in the schedule.

A cluster update with Luijten-Blote updates are being constructed. 

## Building and Running

The project uses CMake for building.

```bash
# Create the build directory and configure, if it already exsists first delete it
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build (repeat this after any code change)
cmake --build build -j$(nproc)

# Run
./build/run --solver metropolis --L 16 --sigma 1.5 --sigma_index 0
```

### Command-line arguments

| Argument | Description |
|---|---|
| `--solver` | Update algorithm to use. Either `metropolis` or `LB` (cluster update — under construction). |
| `--L` | Lattice size. The system is an `L x L` grid of spins. |
| `--sigma` | Strength of the long-range interaction, entering the coupling as `1/r^(sigma + d)`, with `d = 2`. |
| `--sigma_index` | Index of the directory that this `sigma` value's output should be stored under (see [Storage layout](#storage-layout)). |

## Configuration (`config/params.hpp`)

Simulation parameters that are not passed via the command line are set in `config/params.hpp`:

```cpp
// --- Lattice ---
int    L;               // lattice side length (LxL spins)
double J;                // coupling constant (positive for ferromagnetic)

// --- Temp Schedule ---
double t_c;              // critical temperature for 2D Ising (for reference)
double t_min;            // lowest temperature in schedule
double t_max;            // highest temperature in schedule
double sharpness;        // power-law warp around T_c (higher = more points near T_c)

// --- Run control ---
int    temp_updates;          // number of temperature steps
int    num_runs;              // independent parallel runs
int    metro_sweeps_per_temp; // Metropolis sweeps per temperature step
int    store_step;            // store state every N sweeps/updates
```

## Storage Layout

Output is organized in a nested directory structure, rooted at a folder you create before running the program:

1. Create a top-level folder named after the lattice size, e.g. `L16` (for `L = 16`).
2. Inside it, create a subfolder named after the sigma index, e.g. `sigma0` (for `--sigma_index 0`).
3. Pass this path into `main.cpp` as the output directory.

On startup, the program generates a `temp_index.txt` file inside the `sigma*` folder, mapping an integer index to each temperature in the schedule. For each index, a corresponding `temp_<index>` folder is created to hold the output of the runs at that temperature.

Runs are then solved in parallel, distributed across the available threads, and state snapshots are written to their respective `temp_<index>` folders on the fly (frequency controlled by `store_step`).

Example layout:

```
L16/
└── sigma0/
    ├── temp_index.txt
    ├── temp_0/
    |   ├── Snapshots_0.dat
    |   ├── Snapshots_1.dat
    |    .
    |    .
    ├── temp_1/
    └── ...
```

## Analysis Tools

Analysis scripts live alongside the simulation output and use custom-built C++ extensions via `pybind11` for fast file reading.

### Setup

```bash
pip install pybind11

g++ -O3 -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) file_reader.cpp -o file_reader$(python3-config --extension-suffix)
```

This compiles `file_reader.cpp` into a Python-importable extension module used by the analysis scripts.

### `avg_mag.py`

Currently the only available analysis tool. It reads the stored simulation snapshots and plots the average magnetization of the system as a function of temperature.

```bash
python3 avg_mag.py
```