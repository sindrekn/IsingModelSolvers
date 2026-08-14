#include <iostream>
#include <algorithm>
#include <thread>
#include <CLI/CLI.hpp>

#include "backend/runner.hpp"

#include "core/io/storage.hpp"
#include "core/schedule/schedule.hpp"

#include "config/params.hpp"

// ----------------------------------------------------------
// Main
// ----------------------------------------------------------
int main(int argc, char *argv[])
{
    CLI::App app{"My solver"};

    std::string solver; 
    app.add_option("--solver", solver, "Solver to use (metropolis, LB)")->required();

    int L;
    app.add_option("--L", L, "Lattice size")->required();

    double sigma;
    app.add_option("--sigma", sigma, "Sigma for LRIM solver")->required();

    int sigma_index; 
    app.add_option("--sigma_index", sigma_index, "Sigma index for LRIM solver")->required();

    CLI11_PARSE(app, argc, argv);

    if (solver != "metropolis" && solver != "LB") {
        std::cerr << "Error: Invalid solver specified. Must be one of: metropolis, LB." << std::endl;
        return 1;
    }

    if (solver == "metropolis") {
        std::string base_dir = "/home/sindrekampennesheim/Documents/PhD/Optimizing/Output/SimulatedAnnealing/2D_LRIM/metropolis/L" + std::to_string(L) + "/sigma" + std::to_string(sigma_index);

        // -- Test the path --
        if (!std::filesystem::exists(base_dir)) {
            std::cerr << "Error: Base directory does not exist: " << base_dir << std::endl;
            return 1;
        }

        auto p = LRIM_2d_params::defaults(L);

        std::vector<double> temp_schedule = peak_temperature_schedule(
        p.t_min, p.t_max, p.t_c, p.temp_updates, p.sharpness);

        // --- Determine number of threads to use ---
        int num_threads = static_cast<int>(
            std::min(static_cast<unsigned>(p.num_runs),
                    std::max(1u, std::thread::hardware_concurrency())));

        // --- Prepare storage for all runs ---
        std::vector<std::string> temp_dirs = setup_temperature_directories(temp_schedule, base_dir);

        std::cout << "\n-------------------------------\n";
        std::cout << "SA: 2D LRIM simulation with L=" << L << "\n";

        auto t_start = std::chrono::high_resolution_clock::now();

        LRIM_metropolis(L, sigma, num_threads, 0, p.num_runs, temp_schedule, temp_dirs);

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();

        std::cout << "\n"
                << "Simulation took " << elapsed << " s\n"
                << "\n";
    }

    if (solver == "LB") {
        std::string base_dir = "/home/sindrekampennesheim/Documents/PhD/Optimizing/Output/SimulatedAnnealing/2D_LRIM/LuijtenBloteCluster/L" + std::to_string(L) + "/sigma" + std::to_string(sigma_index);

        // -- Test the path --
        if (!std::filesystem::exists(base_dir)) {
            std::cerr << "Error: Base directory does not exist: " << base_dir << std::endl;
            return 1;
        }

        auto p = LRIM_2d_params::defaults(L);

        std::vector<double> temp_schedule = peak_temperature_schedule(
        p.t_min, p.t_max, p.t_c, p.temp_updates, p.sharpness);

        // --- Determine number of threads to use ---
        int num_threads = static_cast<int>(
            std::min(static_cast<unsigned>(p.num_runs),
                    std::max(1u, std::thread::hardware_concurrency())));

        // --- Prepare storage for all runs ---
        std::vector<std::string> temp_dirs = setup_temperature_directories(temp_schedule, base_dir);

        std::cout << "\n-------------------------------\n";
        std::cout << "SA: 2D LRIM with Luijten-Blote cluster updates, L=" << L << "\n";

        auto t_start = std::chrono::high_resolution_clock::now();

        LRIM_LB(L, sigma, num_threads, 0, p.num_runs, temp_schedule, temp_dirs);

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();

        std::cout << "\n"
                << "Simulation took " << elapsed << " s\n"
                << "\n";
    }

    return 0;
}


