#include "runner.hpp"
#include <iostream>

#include "core/bit_ops.hpp"
#include "core/schedule/schedule.hpp"

#include "core/lattice/lattice.hpp"

#include "core/io/storage.hpp"

#include "solvers/LuijtenBlote.hpp"
#include "solvers/metropolis.hpp"

#include "config/params.hpp"

// ----------------------------------------------------------
// Run multiple independent simulations in parallel and store 
// the resulting snapshots in separate files for each 
// temperature step.
// ----------------------------------------------------------
void LRIM_metropolis(
    int L,
    double sigma,
    int num_threads,
    int run_start,
    int run_end,
    std::vector<double> temp_schedule,
    std::vector<std::string> temp_dirs
)
{
    // --- Load parameters and precompute intercations ---
    auto p = LRIM_2d_params::defaults(L);
    int N            = L * L;
    int W            = num_words(N);

    LongRange_DistanceResult distanceResult = LongRange_DistancePrecompute(L, sigma); 

    // --- Create threads and distribute runs ---
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int num_runs = run_end - run_start;

    int runs_per_thread = num_runs / num_threads;
    int extra           = num_runs % num_threads;

    std::cout << "Runs starts on " + std::to_string(num_threads) + " threads" << std::endl;

    // --- Run the simulations in parallel ---
    int start = run_start;
    for (int t = 0; t < num_threads; t++) {
        int end = start + runs_per_thread + (t < extra ? 1 : 0);

        threads.emplace_back([&, start, end]() {
            std::uniform_real_distribution <double> udist(0.0, 1.0);

            for (int run = start; run < end; run++) {
                std::mt19937 rng(std::random_device{}() + run);

                // Working state for this run — lives on the thread's stack
                std::vector<uint64_t> state(W);
                random_binary_state(state.data(), N, rng);

                std::vector<float> h_local = construct_h(N, state.data(), distanceResult);

                for (int temp = 0; temp < p.temp_updates; temp++) {
                    StateWriter writer(temp_dirs[temp], run);
                    
                    for (int sweep = 0; sweep < p.metro_sweeps_per_temp; sweep++) {
                        FieldUpdate_solver(
                            distanceResult,
                            h_local,
                            N,
                            L,
                            state.data(),
                            1.0 / temp_schedule[temp],
                            udist,
                            rng);
                        if (sweep % p.store_step == 0) {
                            writer.add_snapshot(state);
                        }
                    }
                    // Always store the final state of each temperature step
                    writer.add_snapshot(state);
                    writer.close();
                }
            }
        });
        start = end;
    }
    for (auto& t : threads)
        t.join();

    return;
}


void LRIM_LB(
    int L,
    double sigma,
    int num_threads,
    int run_start,
    int run_end,
    std::vector<double> temp_schedule,
    std::vector<std::string> temp_dirs
)   
{
    // --- Load parameters and precompute the cumulative distrubution function ---
    auto p = LRIM_2d_params::defaults(L);
    int N            = L * L;
    int W            = num_words(N);

    std::vector<float> beta_schedule(p.temp_updates);
    for (int i = 0; i < p.temp_updates; ++i)
        beta_schedule[i] = 1.0f / static_cast<float>(temp_schedule[i]);

    LongRange_DistanceResult distanceResult = LongRange_DistancePrecompute(L, sigma); 
    LB_ClusterPrecomputeResult clusterResult  = LB_ClusterPrecompute(distanceResult, beta_schedule, L);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int num_runs = run_end - run_start;

    int runs_per_thread = num_runs / num_threads;
    int extra           = num_runs % num_threads;

    std::cout << "Runs starts on " + std::to_string(num_threads) + " threads" << std::endl;

    int start = run_start;
    for (int t = 0; t < num_threads; t++) {
        int end = start + runs_per_thread + (t < extra ? 1 : 0);

        threads.emplace_back([&, start, end]() {
            std::uniform_real_distribution <double> udist(0.0, 1.0);

            for (int run = start; run < end; run++) {
                std::mt19937 rng(std::random_device{}() + run);
                // std::mt19937 rng(run); // Use run index as seed for reproducibility

                // Working state for this run — lives on the thread's stack
                std::vector<uint64_t> state(W);
                random_binary_state(state.data(), N, rng);

                for (int beta_id = 0; beta_id < p.temp_updates; beta_id++) {
                    StateWriter writer(temp_dirs[beta_id], run);
                    
                    for (int sweep = 0; sweep < p.metro_sweeps_per_temp; sweep++) {
                        
                        LuijtenBloteCluster(
                            clusterResult,
                            distanceResult,
                            L,
                            state.data(),
                            beta_id,
                            udist,
                            rng);
                        if (sweep % p.store_step == 0) {
                            writer.add_snapshot(state);
                        }
                    }
                    // Always store the final state of each temperature step
                    writer.add_snapshot(state);
                    writer.close();
                }
            }
        });
        start = end;
    }

    for (auto& t : threads)
        t.join();

    return;
}



