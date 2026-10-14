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
    app.add_option("--ising_solver", ising_solver, 
        "Ising solver to use (quadratic, wolff_quadratic, quadratic_nnn, triangular, wolff_triangular, honeycomb, wolff_honeycomb, cubic, wolff_cubic, cubic_nnn, lrim, lb_lrim)")
        ->required()
        ->check(CLI::IsMember({"quadratic", "wolff_quadratic", 
            "quadratic_nnn", "triangular", "wolff_triangular", 
            "honeycomb", "wolff_honeycomb", "cubic", "wolff_cubic", "cubic_nnn", 
            "lrim", "lb_lrim"}));

    int L = 0; 
    app.add_option("--L", L, "Lattice size (LxL for 2D, LxLxL for 3D)")
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
    app.add_option("--run_end", run_end, "Ending run index (run_end - run_start = number of runs)")
        ->required()
        ->check(CLI::NonNegativeNumber);

    double sigma = 0.0;
    app.add_option("--sigma", sigma, "Sigma value for long-range interactions (only for lrim solver)")
        ->check(CLI::PositiveNumber);

    std::string base_dir;
    app.add_option("--base_dir", base_dir, "Base directory for storing simulation results")
        ->check(CLI::ExistingDirectory);

    CLI11_PARSE(app, argc, argv);

    if (t_min <= 0 || t_max <= 0 || t_min >= t_max) {
        std::cerr << "Error: Invalid temperature range. Ensure that 0 < t_min < t_max." 
        << std::endl;
        return 1;
    }

    if (run_start < 0 || run_end < 0 || run_start > run_end) {
        std::cerr << "Error: Invalid run indices. Ensure that 0 <= run_start <= run_end." 
        << std::endl;
        return 1;
    }

    if (sigma < 0) {
        std::cerr << "Error: Invalid sigma. Ensure that it is non-negative." 
        << std::endl;
        return 1;
    }

    auto p = params::defaults(L);

    std::vector<double> temp_schedule = peak_temperature_schedule(
    t_min, t_max, t_c, p.temp_updates, p.sharpness);

    // --- Determine number of threads to use ---
    int num_threads = static_cast<int>(
        std::min(static_cast<unsigned>(run_end - run_start),
                std::max(1u, std::thread::hardware_concurrency())));

    // --- Prepare storage for all runs ---
    std::vector<std::string> temp_dirs = setup_temperature_directories(temp_schedule, base_dir, run_start, run_end);

    std::cout << "\n-------------------------------\n";
    std::cout << "Total number of runs: " << (run_end - run_start) << std::endl;
    std::cout << "Number of temperature steps: " << p.temp_updates << std::endl;
    std::cout << "Number of threads: " << num_threads << std::endl;

    run_ising_solver solver(L, num_threads, run_start, run_end, temp_schedule, temp_dirs);

    auto t_start = std::chrono::high_resolution_clock::now();

    if (ising_solver == "quadratic") {
        solver.quadratic_solver();
    } else if (ising_solver == "wolff_quadratic") {
        solver.wolff_quadratic_solver();
    } else if (ising_solver == "quadratic_nnn") {
        solver.quadratic_nnn_solver();
    } else if (ising_solver == "triangular") {
        solver.triangular_solver(L, L);
    } else if (ising_solver == "wolff_triangular") {
        solver.wolff_triangular_solver(L, L);
    } else if (ising_solver == "honeycomb") {
        solver.honeycomb_solver(L, L);
    } else if (ising_solver == "wolff_honeycomb") {
        solver.wolff_honeycomb_solver(L, L);
    } else if (ising_solver == "cubic") {
        solver.cubic_solver();
    } else if (ising_solver == "wolff_cubic") {
        solver.wolff_cubic_solver();
    } else if (ising_solver == "cubic_nnn") {
        solver.cubic_nnn_solver();
    } else if (ising_solver == "lrim") {
        solver.lrim(sigma);
    } else if (ising_solver == "lb_lrim") {
        solver.lb(sigma);
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

