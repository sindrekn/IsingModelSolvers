#include <iostream>
#include <algorithm>
#include <thread>

#include "backend/runner.hpp"

#include "core/io/storage.hpp"
#include "core/schedule/schedule.hpp"

#include "config/params.hpp"

// ----------------------------------------------------------
// Main
// ----------------------------------------------------------
int main()
{
    int L = 16; // Lattice size 16x16
    auto p = params::defaults(L); // Lattice size 16x16

    std::string base_dir = "/home/sindrekampennesheim/Documents/PhD/Optimizing/Output/SimulatedAnnealing/2D_nn/ferromagnetic/metropolis/L" + std::to_string(L);

    // -- Test the path --
    if (!std::filesystem::exists(base_dir)) {
        std::cerr << "Error: Base directory does not exist: " << base_dir << std::endl;
        return 1;
    }

    std::vector<double> temp_schedule = peak_temperature_schedule(
    p.t_min, p.t_max, p.t_c, p.temp_updates, p.sharpness);

    // --- Determine number of threads to use ---
    int num_threads = static_cast<int>(
        std::min(static_cast<unsigned>(p.num_runs),
                std::max(1u, std::thread::hardware_concurrency())));

    // --- Prepare storage for all runs ---
    std::vector<std::string> temp_dirs = setup_temperature_directories(temp_schedule, base_dir);

    std::cout << "\n-------------------------------\n";

    run_ising_solver solver(L, num_threads, 0, p.num_runs, temp_schedule, temp_dirs);

    auto t_start = std::chrono::high_resolution_clock::now();

    solver.nn_2d_solver();

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n"
            << "Simulation took " << elapsed << " s\n"
            << "-------------------------------\n";

    return 0;
}


