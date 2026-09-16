#include <iostream>
#include <vector>

#include "core/lattice/lattice.hpp"
#include "core/schedule/schedule.hpp"

#include "solvers/solvers.hpp"

// ----------------------------------------------------------
// Main
// ----------------------------------------------------------
int main(int argc, char *argv[])
{

    std::string ising_solver; 
    app.add_option("--ising_solver", ising_solver, "Ising solver to use (nn_2d, wolff, nnn_2d, triangular, honeycomb, nn_3d, nnn_3d)")
        ->required()
        ->check(CLI::IsMember({"nn_2d", "wolff", "nnn_2d", "triangular", "honeycomb", "nn_3d", "nnn_3d"}));

    int L; 
    
    int Lx; 
    
    int Ly; 
    
    float beta;

    int num_sweeps;

    double elapsed = 0.0;

    std::cout << "\n-------------------------------\n";

    if (solver == "nn_sa") {

        if (algo != "Metropolis" && algo != "Wolff") {
            std::cerr << "Error: Invalid algorithm specified. Must be one of: Metropolis, Wolff." << std::endl;
            return 1;
        }

        bool update_type = (algo == "Metropolis") ? true : false;

        int N = L * L;

        auto neighbors_vec = precompute_2d_nn(L);

        std::vector<double> temp_schedule(1); 
        temp_schedule[0] = 1.0 / beta;

        std::uniform_real_distribution<double> udist(0.0, 1.0);
        std::mt19937 rng(std::random_device{}());

        // Working state for this run — lives on the thread's stack
        std::vector<uint64_t> state((L * L + 63) / 64, 0);

        // --- Precompute checkerboard site lists once, shared across all threads ---
        std::vector<int> even_sites;
        std::vector<int> odd_sites;
        even_sites.reserve(N / 2);
        odd_sites.reserve(N / 2);
        for (int id = 0; id < N; id++) {
            if (((id / L) + (id % L)) % 2 == 0)
                even_sites.push_back(id);
            else
                odd_sites.push_back(id);
        }

        if (update_type) {

            std::cout << "SA: 2D NN Ising simulation with L=" << L << ", Metropolis updates";
            std::cout << ", Beta: " << beta << " and num_sweeps: " << num_sweeps;
            std::cout << "\n";

            auto BetaJS = precomputed_BetaJS_2d_nn(temp_schedule, J);

            const double* BetaJS_row = &BetaJS[0];

            auto t_start = std::chrono::high_resolution_clock::now();

            for (int sweep = 0; sweep < num_sweeps; sweep++) {
                for (int parity = 0; parity <= 1; parity ++) {
                    nn_metropolis(
                        neighbors_vec.data(), 
                        BetaJS_row,
                        even_sites,
                        odd_sites,
                        parity,
                        state.data(),
                        udist,
                        rng);
                }
            }

            auto t_end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration<double>(t_end - t_start).count();
        } else {

            std::cout << "SA: 2D NN Ising simulation with L=" << L << ", Wolff updates";
            std::cout << ", Beta: " << beta << " and num_sweeps: " << num_sweeps;
            std::cout << "\n";

            auto P_add = precomputed_Padd_2d_nn(temp_schedule, J);

            std::uniform_int_distribution<int> site_dist(0, N - 1);

            std::vector<int> stack;
            stack.reserve(N);

            auto t_start = std::chrono::high_resolution_clock::now();

            for (int sweep = 0; sweep < num_sweeps; sweep++) {
                nn_wolff(
                    neighbors_vec.data(),
                    P_add[0],
                    state.data(),
                    stack,
                    udist,
                    site_dist,
                    rng);
            }

            auto t_end = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration<double>(t_end - t_start).count();
        }

        std::cout << "\n"
                << "Simulation took " << elapsed << " s\n"
                << "Avg time per sweep: " << elapsed / num_sweeps << " s\n";
    }

    if (solver == "2D_FullyConnected") {

        std::cout << "SA: 2D LRIM simulation with L=" << L << " and Metropolis spin updates\n";
        std::cout << "Sigma: " << sigma << ", Beta: " << beta << " and num_sweeps: " << num_sweeps;
        std::cout << "\n";

        int N = L * L;
        std::vector<uint64_t> state((L * L + 63) / 64, 0);

        LongRange_DistanceResult distanceResult = LongRange_DistancePrecompute(L, sigma); 

        std::vector<float> h_local = construct_h(N, state.data(), distanceResult);

        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> udist(0.0, 1.0);

        auto t_start = std::chrono::high_resolution_clock::now();

        for (int sweep = 0; sweep < num_sweeps; ++sweep) {

            FieldUpdate_solver(
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
        elapsed = std::chrono::duration<double>(t_end - t_start).count();

        std::cout << "\n"
                << "Simulation took \t \t" << elapsed << " s\n"
                << "Avg time per sweep: \t \t" << elapsed / num_sweeps << " s\n";
    }

    if (solver == "2D_LB") {

        std::cout << "SA: 2D Fully Connected Ising simulation with L=" << L << " and Luijten-Blote cluster updates\n";
        std::cout << "Sigma: " << sigma << ", Beta: " << beta << " and num_sweeps: " << num_sweeps;
        std::cout << "\n";

        LongRange_DistanceResult distanceResult = LongRange_DistancePrecompute(L, sigma);         
        
        std::vector<float> beta_schedule(1);
        beta_schedule[0] = beta;
        
        LB_ClusterPrecomputeResult clusterResult  = LB_ClusterPrecompute(distanceResult, beta_schedule, L);

        std::vector<uint64_t> state((L * L + 63) / 64, 0);
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> udist(0.0, 1.0);

        auto t_start = std::chrono::high_resolution_clock::now();

        for (int sweep = 0; sweep < num_sweeps; ++sweep) {

            LuijtenBloteCluster(
                clusterResult,
                distanceResult, 
                L, 
                state.data(), 
                0, 
                udist, 
                rng);
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(t_end - t_start).count();

        std::cout << "\n"
                << "Simulation took " << elapsed << " s\n"
                << "Avg time per sweep: " << elapsed / num_sweeps << " s\n";
    }

    std::cout << "-------------------------------\n";

    return 0;
}


