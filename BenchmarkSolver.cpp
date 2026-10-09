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
    void BenchmarkWolff(std::string lattice_type);
    void BenchmarkQuadraticNNN(double J1, double J2);
    // void BenchmarkTriangular(int Lx, int Ly);
    // void BenchmarkHoneycomb(int Lx, int Ly);
    // void BenchmarkCubic();
    void BenchmarkCubicNNN(double J1, double J2);
    void BenchmarkLRIM(double sigma);
    void BenchmarkLB(double sigma);

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

void BenchmarkSolver::BenchmarkWolff(std::string lattice_type) {
    std::cout << "SA: " << lattice_type << " Ising simulation with L=" << L << " and Wolff cluster updates\n";
    std::cout << "Beta: " << beta << " and num_sweeps: " << num_sweeps;
    std::cout << "\n";

    std::vector<int> neighbors;
    int num_neighbors;
    double P_add;

    if (lattice_type == "quadratic") {
        neighbors = neighbors_quadratic(L);
        num_neighbors = 4;
        P_add = 1.0 - std::exp(-2.0 * beta);
    } else if (lattice_type == "triangular") {
        neighbors = neighbors_triangular(L, L);
        num_neighbors = 6;
        P_add = 1.0 - std::exp(-2.0 * beta);
    } else if (lattice_type == "honeycomb") {
        neighbors = neighbors_honeycomb(L, L);
        num_neighbors = 3;
        P_add = 1.0 - std::exp(-2.0 * beta);
    } else if (lattice_type == "cubic") {
        neighbors = neighbors_cubic(L);
        num_neighbors = 6;
        P_add = 1.0 - std::exp(-2.0 * beta);
    } else {
        std::cerr << "Error: Unknown lattice type specified for Wolff benchmark." << std::endl;
        return;
    }

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
            << "Simulation of " << lattice_type << " took \t \t" << elapsed << " s\n"
            << "Avg time per sweep: \t \t" << elapsed / num_sweeps << " s\n";
}

void BenchmarkSolver::BenchmarkQuadraticNNN(double J1, double J2) {
    std::cout << "SA: 2D Ising simulation with L=" << L << " and Quadratic Next-Nearest Neighbor updates\n";
    std::cout << "Beta: " << beta << " and num_sweeps: " << num_sweeps;
    std::cout << "\n";

    std::vector<int> nn_neighbors = neighbors_quadratic(L);
    std::vector<int> nnn_neighbors = neighbors_quadratic_nnn(L);

    auto t_start = std::chrono::high_resolution_clock::now();
                    
    for (int sweep = 0; sweep < num_sweeps; sweep++) {
        quadratic_nnn(
            N,
            J1,
            J2,
            nn_neighbors.data(),
            nnn_neighbors.data(),
            beta,
            state.data(),
            udist,
            rng);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n"
            << "Simulation took \t \t" << elapsed << " s\n"
            << "Avg time per sweep: \t \t" << elapsed / num_sweeps << " s\n";

}

void BenchmarkSolver::BenchmarkCubicNNN(double J1, double J2) {
    std::cout << "SA: 3D Ising simulation with L=" << L << " and Cubic Next-Nearest Neighbor updates\n";
    std::cout << "Beta: " << beta << " and num_sweeps: " << num_sweeps;
    std::cout << "\n";

    std::vector<int> nn_neighbors = neighbors_cubic(L);
    std::vector<int> nnn_neighbors = neighbors_cubic_nnn(L);
    
    for (int sweep = 0; sweep < num_sweeps; sweep++) {
        cubic_nnn(
            N,
            J1,
            J2,
            nn_neighbors.data(),
            nnn_neighbors.data(),
            beta,
            state.data(),
            udist,
            rng);
    }

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

void BenchmarkSolver::BenchmarkLB(double sigma) {
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
    app.add_option("--ising_solver", ising_solver, "Ising solver to use (wolff, quadratic_nnn, cubic_nnn, lrim, lb)")
        ->required()
        ->check(CLI::IsMember({"quadratic", "wolff", "quadratic_nnn", "triangular", "honeycomb", "cubic", "cubic_nnn", "lrim", "lb"}));

    std::string lattice_type;
    app.add_option("--lattice_type", lattice_type, "Lattice type to use (quadratic, triangular, honeycomb, cubic)")
        ->check(CLI::IsMember({"quadratic", "triangular", "honeycomb", "cubic"}));

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

    double J1;
    app.add_option("--J1", J1, "Nearest neighbor interaction strength")
        ->check(CLI::PositiveNumber);  

    double J2;
    app.add_option("--J2", J2, "Next-nearest neighbor interaction strength")
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

    if (ising_solver == "lrim") {
        benchmark.BenchmarkLRIM(sigma);
    } else if (ising_solver == "lb") {
        benchmark.BenchmarkLB(sigma);
    } else if (ising_solver == "wolff") {
        benchmark.BenchmarkWolff(lattice_type);
    } else if (ising_solver == "quadratic_nnn") {
        benchmark.BenchmarkQuadraticNNN(J1, J2);
    } else if (ising_solver == "cubic_nnn") {
        benchmark.BenchmarkCubicNNN(J1, J2);
    } else {
        std::cerr << "Error: Unknown Ising solver specified." << std::endl;
        return 1;
    }

    std::cout << "\n-------------------------------\n";

}
