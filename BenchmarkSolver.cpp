#include <iostream>
#include <vector>
#include <CLI/CLI.hpp>

#include "core/lattice/lattice.hpp"
#include "core/schedule/schedule.hpp"

#include "solvers/solvers.hpp"

class BenchmarkSolver {
public:
    BenchmarkSolver(
        int L, 
        double temp, 
        int num_sweeps
    ); 

    // void BenchmarkQuadratic();
    void BenchmarkWolff();
    // void BenchmarkQuadraticNNN();
    // void BenchmarkTriangular(int Lx, int Ly);
    // void BenchmarkHoneycomb(int Lx, int Ly);
    // void BenchmarkCubic();
    // void BenchmarkCubicNNN();
    void BenchmarkLRIM(double sigma);
    void BenchmarkLuijtenBloteCluster(double sigma);

private:
    int L; 
    int N;
    double beta; 
    int num_sweeps;
    std::vector<uint64_t> state;
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> udist{0.0, 1.0};
    std::uniform_int_distribution<int>    sitedist;
};

BenchmarkSolver::BenchmarkSolver(
    int L, 
    double temp, 
    int num_sweeps
) : L(L), 
    N(L*L), 
    beta(1.0 / temp), 
    num_sweeps(num_sweeps), 
    state((N + 63) / 64, 0), 
    sitedist(0, N - 1) 
{}

void BenchmarkSolver::BenchmarkWolff() {
    std::cout << "SA: 2D Ising simulation with L=" << L << " and Wolff cluster updates\n";
    std::cout << "Beta: " << beta << " and num_sweeps: " << num_sweeps;
    std::cout << "\n";

    std::vector<int> neighbors = neighbors_quadratic(L);
    int num_neighbors = 4;
    double P_add = 1.0 - std::exp(-2.0 * beta);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int sweep = 0; sweep < num_sweeps; ++sweep) {
        std::vector<int> stack;
        wolff_cluster(
            num_neighbors,
            neighbors.data(),
            P_add,
            state.data(),
            stack,
            udist,
            sitedist,
            rng);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n"
            << "Simulation took \t \t" << elapsed << " s\n"
            << "Avg time per sweep: \t \t" << elapsed / num_sweeps << " s\n";
}

void BenchmarkSolver::BenchmarkLRIM(double sigma) {
    std::cout << "SA: 2D LRIM simulation with L=" << L << " and Metropolis spin updates\n";
    std::cout << "Sigma: " << sigma << ", Beta: " << beta << " and num_sweeps: " << num_sweeps;
    std::cout << "\n";

    LongRange_DistanceResult distanceResult = LongRange_DistancePrecompute(L, sigma); 

    std::vector<double> h_local = construct_h(N, state.data(), distanceResult);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int sweep = 0; sweep < num_sweeps; ++sweep) {

        FieldUpdate(
            distanceResult,
            h_local,
            N,
            L,
            state.data(),
            beta,
            udist,
            rng);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n"
            << "Simulation took \t \t" << elapsed << " s\n"
            << "Avg time per sweep: \t \t" << elapsed / num_sweeps << " s\n";
}

void BenchmarkSolver::BenchmarkLuijtenBloteCluster(double sigma) {
    std::cout << "SA: 2D LRIM simulation with L=" << L << " and Luijten Blote cluster updates\n";
    std::cout << "Sigma: " << sigma << ", Beta: " << beta << " and num_sweeps: " << num_sweeps;
    std::cout << "\n";

    std::vector<double> beta_schedule;
    beta_schedule.push_back(beta);

    LongRange_DistanceResult distanceResult = LongRange_DistancePrecompute(L, sigma); 
    LB_ClusterPrecomputeResult clusterResult = LB_ClusterPrecompute(distanceResult, beta_schedule, L);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int sweep = 0; sweep < num_sweeps; ++sweep) {

        LuijtenBloteCluster(
            clusterResult,
            distanceResult,
            N,
            L,
            state.data(),
            0,
            udist,
            rng);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n"
            << "Simulation took \t \t" << elapsed << " s\n"
            << "Avg time per sweep: \t \t" << elapsed / num_sweeps << " s\n";
}

// ----------------------------------------------------------
// Main
// ----------------------------------------------------------
int main(int argc, char *argv[])
{
    CLI::App app{"My solver"};

    std::string ising_solver; 
    app.add_option("--ising_solver", ising_solver, "Ising solver to use (quadratic, wolff, quadratic_nnn, triangular, honeycomb, cubic, cubic_nnn, lrim, lb)")
        ->required()
        ->check(CLI::IsMember({"quadratic", "wolff", "quadratic_nnn", "triangular", "honeycomb", "cubic", "cubic_nnn", "lrim", "lb"}));

    int L = 0; 
    app.add_option("--L", L, "Lattice size (LxL for 2D, LxLxL for 3D)")
        ->check(CLI::PositiveNumber);

    int Lx = 0; 
    app.add_option("--Lx", Lx, "Lattice size in x-direction (for triangular and honeycomb lattices)")
        ->check(CLI::PositiveNumber);
    
    int Ly = 0; 
    app.add_option("--Ly", Ly, "Lattice size in y-direction (for triangular and honeycomb lattices)")
        ->check(CLI::PositiveNumber);

    double sigma;
    app.add_option("--sigma", sigma, "Sigma for FullyConnected solver")
        ->check(CLI::PositiveNumber);

    double temp;
    app.add_option("--T", temp, "Temperature")
        ->required()
        ->check(CLI::PositiveNumber);

    int num_sweeps;
    app.add_option("--num_sweeps", num_sweeps, "Number of sweeps")
        ->required()
        ->check(CLI::PositiveNumber);

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

    BenchmarkSolver benchmark(L, temp, num_sweeps);

    // if (ising_solver == "quadratic") {
    //     benchmark.BenchmarkQuadratic();
    // } else if (ising_solver == "wolff") {
    //     benchmark.BenchmarkWolff();
    // } else if (ising_solver == "quadratic_nnn") {
    //     benchmark.BenchmarkQuadraticNNN();
    // } else if (ising_solver == "triangular") {
    //     benchmark.BenchmarkTriangular(Lx, Ly);
    // } else if (ising_solver == "honeycomb") {
    //     benchmark.BenchmarkHoneycomb(Lx, Ly);
    // } else if (ising_solver == "cubic") {
    //     benchmark.BenchmarkCubic();
    // } else if (ising_solver == "cubic_nnn") {
    //     benchmark.BenchmarkCubicNNN();
    // } else if (ising_solver == "lrim") {
    //     benchmark.BenchmarkLRIM(sigma);
    // } else {
    //     std::cerr << "Error: Unknown Ising solver specified." << std::endl;
    //     return 1;
    // }
    if (ising_solver == "lrim") {
        benchmark.BenchmarkLRIM(sigma);
    } else if (ising_solver == "lb") {
        benchmark.BenchmarkLuijtenBloteCluster(sigma);
    } else if (ising_solver == "wolff") {
        benchmark.BenchmarkWolff();
    } else {
        std::cerr << "Error: Unknown Ising solver specified." << std::endl;
        return 1;
    }

    std::cout << "\n-------------------------------\n";

}
