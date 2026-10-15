#include <iostream>
#include <vector>
#include <CLI/CLI.hpp>
#include <fstream>
#include <filesystem>

#include "core/lattice/lattice.hpp"
#include "core/schedule/schedule.hpp"

#include "solvers/solvers.hpp"

class BenchmarkSolver {
public:
    BenchmarkSolver(
        int L, 
        int N,
        double temp, 
        int num_sweeps
    ); 

    double BenchmarkQuadratic();
    double BenchmarkWolff(std::string lattice_type);
    double BenchmarkQuadraticNNN(double J1, double J2);
    double BenchmarkTriangular();
    double BenchmarkHoneycomb();
    double BenchmarkCubic();
    double BenchmarkCubicNNN(double J1, double J2);
    double BenchmarkLRIM(double sigma);
    double BenchmarkLB(double sigma);

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
    int N,
    double temp, 
    int num_sweeps
) : L(L), 
    N(N), 
    beta(1.0 / temp), 
    num_sweeps(num_sweeps), 
    state((N + 63) / 64, 0), 
    sitedist(0, N - 1) 
{}

double BenchmarkSolver::BenchmarkWolff(std::string lattice_type) {
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
        return -1.0;
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

    return elapsed / num_sweeps;
}

double BenchmarkSolver::BenchmarkQuadratic() {
    std::cout << "SA: 2D Ising simulation with L=" << L << " and Quadratic Nearest Neighbor updates\n";
    std::cout << "Beta: " << beta << " and num_sweeps: " << num_sweeps;
    std::cout << "\n";

    std::vector<double> temp_schedule(1, 1.0 / beta);

    std::vector<int> nn_neighbors = neighbors_quadratic(L);
    std::vector<double> BetaJS = BetaJS_quadratic(temp_schedule, 1.0);

    std::vector<int> even_sites, odd_sites;
    even_sites.reserve(N / 2);
    odd_sites.reserve(N / 2);
    for (int id = 0; id < N; id++) {
        if (((id / L) + (id % L)) % 2 == 0) even_sites.push_back(id);
        else odd_sites.push_back(id);
    }

    auto t_start = std::chrono::high_resolution_clock::now();
                    
    for (int sweep = 0; sweep < num_sweeps; sweep++) {
        for (int parity = 0; parity < 2; parity++) {
            quadratic(
                nn_neighbors.data(),
                BetaJS.data(),
                even_sites,
                odd_sites,
                parity,
                state.data(),
                udist,
                rng);
            }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n"
            << "Simulation took \t \t" << elapsed << " s\n"
            << "Avg time per sweep: \t \t" << elapsed / num_sweeps << " s\n";
    
    return elapsed / num_sweeps;
}

double BenchmarkSolver::BenchmarkQuadraticNNN(double J1, double J2) {
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
    
    return elapsed / num_sweeps;
}

double BenchmarkSolver::BenchmarkHoneycomb() {
    std::cout << "SA: 2D Ising simulation with L=" << L << " and Honeycomb updates\n";
    std::cout << "Beta: " << beta << " and num_sweeps: " << num_sweeps;
    std::cout << "\n";

    std::vector<int> neighbors = neighbors_honeycomb(L, L);

    std::vector<double> temp_schedule(1, 1.0 / beta);
    std::vector<double> BetaJS = BetaJS_honeycomb(temp_schedule, 1.0);

    auto t_start = std::chrono::high_resolution_clock::now();
                    
    for (int sweep = 0; sweep < num_sweeps; sweep++) {
        honeycomb(
            N,
            neighbors.data(),
            BetaJS.data(),
            state.data(),
            udist,
            rng);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n"
            << "Simulation took \t \t" << elapsed << " s\n"
            << "Avg time per sweep: \t \t" << elapsed / num_sweeps << " s\n";
    
    return elapsed / num_sweeps;
}

double BenchmarkSolver::BenchmarkTriangular() {
    std::cout << "SA: 2D Ising simulation with L=" << L << " and Triangular updates\n";
    std::cout << "Beta: " << beta << " and num_sweeps: " << num_sweeps;
    std::cout << "\n";

    std::vector<int> neighbors = neighbors_triangular(L, L);

    std::vector<double> temp_schedule(1, 1.0 / beta);
    std::vector<double> BetaJS = BetaJS_triangular(temp_schedule, 1.0);

    auto t_start = std::chrono::high_resolution_clock::now();
                    
    for (int sweep = 0; sweep < num_sweeps; sweep++) {
        triangular(
            N,
            neighbors.data(),
            BetaJS.data(),
            state.data(),
            udist,
            rng);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n"
            << "Simulation took \t \t" << elapsed << " s\n"
            << "Avg time per sweep: \t \t" << elapsed / num_sweeps << " s\n";
    
    return elapsed / num_sweeps;
}

double BenchmarkSolver::BenchmarkCubic() {
    std::cout << "SA: 3D Ising simulation with L=" << L << " and Cubic updates\n";
    std::cout << "Beta: " << beta << " and num_sweeps: " << num_sweeps;
    std::cout << "\n";

    std::vector<int> neighbors = neighbors_cubic(L);

    std::vector<double> temp_schedule(1, 1.0 / beta);
    std::vector<double> BetaJS = BetaJS_cubic(temp_schedule, 1.0);

    std::vector<int> even_sites, odd_sites;
    even_sites.reserve(N / 2);
    odd_sites.reserve(N / 2);
    for (int id = 0; id < N; id++) {
        int i = id / (L * L);
        int j = (id / L) % L;
        int k = id % L;

        if ((i + j + k) % 2 == 0)
            even_sites.push_back(id);
        else
            odd_sites.push_back(id);
    }

    auto t_start = std::chrono::high_resolution_clock::now();
                    
    for (int sweep = 0; sweep < num_sweeps; sweep++) {
        for (int parity = 0; parity < 2; parity++) {
            cubic(
                neighbors.data(),
                BetaJS.data(),
                even_sites,
                odd_sites,
                parity,
                state.data(),
                udist,
                rng);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n"
            << "Simulation took \t \t" << elapsed << " s\n"
            << "Avg time per sweep: \t \t" << elapsed / num_sweeps << " s\n";
    
    return elapsed / num_sweeps;
}

double BenchmarkSolver::BenchmarkCubicNNN(double J1, double J2) {
    std::cout << "SA: 3D Ising simulation with L=" << L << " and Cubic Next-Nearest Neighbor updates\n";
    std::cout << "Beta: " << beta << " and num_sweeps: " << num_sweeps;
    std::cout << "\n";

    std::vector<int> nn_neighbors = neighbors_cubic(L);
    std::vector<int> nnn_neighbors = neighbors_cubic_nnn(L);

    auto t_start = std::chrono::high_resolution_clock::now();
    
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

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n"
        << "Simulation took \t \t" << elapsed << " s\n"
        << "Avg time per sweep: \t \t" << elapsed / num_sweeps << " s\n";
    
    return elapsed / num_sweeps;
    
}

double BenchmarkSolver::BenchmarkLRIM(double sigma) {
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
    
    return elapsed / num_sweeps;
}

double BenchmarkSolver::BenchmarkLB(double sigma) {
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
    
    return elapsed / num_sweeps;
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
        ->check(CLI::IsMember({"lrim", "lb", "wolff", "quadratic_nnn", "cubic_nnn", "quadratic", "triangular", "honeycomb", "cubic"}));

    std::string lattice_type;
    app.add_option("--lattice_type", lattice_type, "Lattice type to use (quadratic, triangular, honeycomb, cubic)")
        ->required()
        ->check(CLI::IsMember({"quadratic", "triangular", "honeycomb", "cubic"}));

    int L = 0; 
    app.add_option("--L", L, "Lattice size (LxL for 2D, LxLxL for 3D)")
        ->required()
        ->check(CLI::PositiveNumber);

    double sigma = 0.0;
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

    const bool is_3d = (ising_solver == "cubic_nnn") ||
                    (ising_solver == "wolff" && lattice_type == "cubic");
    int N = is_3d ? L * L * L : L * L;

    BenchmarkSolver benchmark(L, N, temp, num_sweeps);

    double avg_time = 0.0;

    if (ising_solver == "lrim") {
        avg_time = benchmark.BenchmarkLRIM(sigma);
    } else if (ising_solver == "lb") {
        avg_time = benchmark.BenchmarkLB(sigma);
    } else if (ising_solver == "wolff") {
        avg_time = benchmark.BenchmarkWolff(lattice_type);
    } else if (ising_solver == "quadratic_nnn") {
        avg_time = benchmark.BenchmarkQuadraticNNN(J1, J2);
    } else if (ising_solver == "cubic_nnn") {
        avg_time = benchmark.BenchmarkCubicNNN(J1, J2);
    } else if (ising_solver == "quadratic") {
        avg_time = benchmark.BenchmarkQuadratic();
    } else if (ising_solver == "triangular") {
        avg_time = benchmark.BenchmarkTriangular();
    } else if (ising_solver == "honeycomb") {
        avg_time = benchmark.BenchmarkHoneycomb();
    } else if (ising_solver == "cubic") {
        avg_time = benchmark.BenchmarkCubic();
    } else {
        std::cerr << "Error: Unknown Ising solver specified." << std::endl;
        return 1;
    }

    // Open the CSV file in append mode
    const std::string csv_path = "benchmark_results.csv";
    const bool write_header = !std::filesystem::exists(csv_path) ||
                            std::filesystem::file_size(csv_path) == 0;

    std::ofstream outfile(csv_path, std::ios::app);
    if (!outfile) {
        std::cerr << "Error: Could not open " << csv_path << " for writing." << std::endl;
        return 1;
    }

    // Header (only once)
    if (write_header) {
        outfile << "ising_solver,lattice_type,size,temp,num_sweeps,sigma,avg_time_s\n";
    }

    // One row per benchmark run
    if (is_3d) {
        outfile << ising_solver << ","
                << lattice_type << ","
                << L << "x" << L << "x" << L << ","
                << temp << ","
                << num_sweeps << ","
                << sigma << ","
                << avg_time << "\n";
    } else {
        outfile << ising_solver << ","
                << lattice_type << ","
                << L << "x" << L << ","
                << temp << ","
                << num_sweeps << ","
                << sigma << ","
                << avg_time << "\n";
    }

    std::cout << "\n-------------------------------\n";

}
