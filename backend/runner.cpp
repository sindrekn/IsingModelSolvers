#include "runner.hpp"
#include <iostream>

#include "core/bit_ops.hpp"
#include "core/schedule/schedule.hpp"

#include "core/lattice/lattice.hpp"

#include "core/io/storage.hpp"

#include "solvers/honeycomb.hpp"
#include "solvers/nn_2d.hpp"
#include "solvers/nnn_2d.hpp"
#include "solvers/nn_3d.hpp"

#include "config/params.hpp"

// ----------------------------------------------------------
// Run multiple independent simulations in parallel and store 
// the resulting snapshots in separate files for each 
// temperature step.
// ----------------------------------------------------------
void nn_2d_solver(
    int L,
    int num_threads,
    int run_start,
    int run_end,
    std::vector<double> temp_schedule,
    std::vector<std::string> temp_dirs
)
{
    // --- Load parameters and precompute intercations ---
    auto p = params::defaults(L);
    int N            = L * L;
    int W            = num_words(N);

    std::vector<int> nn_2d_neigh = precompute_2d_nn(L);

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





