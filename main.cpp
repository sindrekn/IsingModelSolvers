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
    CLI::App app{"Ising Model Solver"};

    std::string ising_solver; 
    app.add_option("--ising_solver", ising_solver, "Ising solver to use (quadratic, wolff, quadratic_nnn, triangular, honeycomb, cubic, cubic_nnn)")
        ->required()
        ->check(CLI::IsMember({"quadratic", "wolff", "quadratic_nnn", "triangular", "honeycomb", "cubic", "cubic_nnn"}));

    int L = 0; 
    app.add_option("--L", L, "Lattice size (LxL for 2D, LxLxL for 3D)")
        ->check(CLI::PositiveNumber);

    int Lx = 0; 
    app.add_option("--Lx", Lx, "Lattice size in x-direction (for triangular and honeycomb lattices)")
        ->check(CLI::PositiveNumber);
    
    int Ly = 0; 
    app.add_option("--Ly", Ly, "Lattice size in y-direction (for triangular and honeycomb lattices)")
        ->check(CLI::PositiveNumber);

    double t_c;
    app.add_option("--t_c", t_c, "Critical temperature for the Ising model")
        ->required()
        ->check(CLI::PositiveNumber);

    double t_min;
    app.add_option("--t_min", t_min, "Minimum temperature for the simulation")
        ->required()
        ->check(CLI::PositiveNumber);
    
    double t_max;
    app.add_option("--t_max", t_max, "Maximum temperature for the simulation")
        ->required()
        ->check(CLI::PositiveNumber);

    int run_start = 0;
    app.add_option("--run_start", run_start, "Starting run index")
        ->check(CLI::NonNegativeNumber);

    int run_end; 
    app.add_option("--run_end", run_end, "Ending run index")
        ->required()
        ->check(CLI::NonNegativeNumber);

    CLI11_PARSE(app, argc, argv);

    if (ising_solver == "triangular" || ising_solver == "honeycomb") {
        if (Lx > 0 && Ly > 0) {
            std::cout << "Using specified Lx and Ly for lattice size: " << Lx << "x" << Ly << std::endl;
        } else if (L > 0) {
            std::cout << "Using specified L for lattice size: " << L << "x" << L << std::endl;
            Lx = Ly = L;
        } else {
            std::cerr << "Error: For triangular and honeycomb lattices, either --L, or both --Lx and --Ly, must be specified and greater than 0." << std::endl;
            return 1;
        }
    } else {
        if (L <= 0) {
            std::cerr << "Error: For other lattices, --L must be specified and greater than 0." << std::endl;
            return 1;
        }
    }

    if (t_min <= 0 || t_max <= 0 || t_min >= t_max) {
        std::cerr << "Error: Invalid temperature range. Ensure that 0 < t_min < t_max." << std::endl;
        return 1;
    }

    if (run_start < 0 || run_end < 0 || run_start > run_end) {
        std::cerr << "Error: Invalid run indices. Ensure that 0 <= run_start <= run_end." << std::endl;
        return 1;
    }

    auto p = params::defaults(L);

    std::vector<double> temp_schedule = peak_temperature_schedule(
    t_min, t_max, t_c, p.temp_updates, p.sharpness);

    // --- Determine number of threads to use ---
    int num_threads = static_cast<int>(
        std::min(static_cast<unsigned>(run_end - run_start),
                std::max(1u, std::thread::hardware_concurrency())));

    std::string base_dir = "/home/sindrekampennesheim/Documents/PhD/Optimizing/Output/IsingSolver/" + ising_solver + "/test1/L" + std::to_string(L);

    // -- Test the path --
    if (!std::filesystem::exists(base_dir)) {
        std::cerr << "Error: Base directory does not exist: " << base_dir << std::endl;
        return 1;
    }

    // --- Prepare storage for all runs ---
    std::vector<std::string> temp_dirs = setup_temperature_directories(temp_schedule, base_dir, run_start, run_end);

    std::cout << "\n-------------------------------\n";
    std::cout << "Total number of runs: " << (run_end - run_start) << std::endl;

    run_ising_solver solver(L, num_threads, run_start, run_end, temp_schedule, temp_dirs);

    auto t_start = std::chrono::high_resolution_clock::now();

    if (ising_solver == "quadratic") {
        solver.quadratic_solver();
    } else if (ising_solver == "wolff") {
        solver.wolff_solver();
    } else if (ising_solver == "quadratic_nnn") {
        solver.quadratic_nnn_solver();
    } else if (ising_solver == "triangular") {
        solver.triangular_solver(Lx, Ly);
    } else if (ising_solver == "honeycomb") {
        solver.honeycomb_solver(Lx, Ly);
    } else if (ising_solver == "cubic") {
        solver.cubic_solver();
    } else if (ising_solver == "cubic_nnn") {
        solver.cubic_nnn_solver();
    } else {
        std::cerr << "Error: Unknown Ising solver specified." << std::endl;
        return 1;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n"
            << "Simulation took " << elapsed << " s\n"
            << "-------------------------------\n";

    return 0;
}


