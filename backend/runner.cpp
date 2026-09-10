#include "runner.hpp"

#include "core/bit_ops.hpp"
#include "core/schedule/schedule.hpp"
#include "core/lattice/lattice.hpp"
#include "core/io/storage.hpp"

#include "solvers/solvers.hpp"

#include "config/params.hpp"

run_ising_solver::run_ising_solver(
    int L,
    int num_threads,
    int run_start, 
    int run_end,
    const std::vector<double>& temp_schedule,
    const std::vector<std::string>& temp_dirs
)
    : L(L), num_threads(num_threads), run_start(run_start), run_end(run_end),
      temp_schedule(temp_schedule), temp_dirs(temp_dirs)
{
    for (double T : temp_schedule) {
        beta_schedule.push_back(1.0 / T);
    }
}

void run_ising_solver::nn_2d_solver() {
    int N = L * L;
    int W = num_words(N);
    auto p = params::defaults(L);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    std::vector<int> neighbors = neighbors_2d_nn(L);
    std::vector<double> BetaJS = BetaJS_2d_nn(temp_schedule, p.J);

    std::vector<int> even_sites, odd_sites;
    even_sites.reserve(N / 2);
    odd_sites.reserve(N / 2);
    for (int id = 0; id < N; id++) {
        if (((id / L) + (id % L)) % 2 == 0) even_sites.push_back(id);
        else odd_sites.push_back(id);
    }

    std::cout << "Runs start on " << num_threads << " threads" << std::endl;

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

                for (int temp_index = 0; temp_index < p.temp_updates; temp_index++) {
                    const double* BetaJS_row = &BetaJS[temp_index * 5];
                    StateWriter writer(temp_dirs[temp_index], run);
                    
                    for (int sweep = 0; sweep < p.metro_sweeps_per_temp; sweep++) {
                        for (int parity = 0; parity < 2; parity++) {
                            nn_2d(
                                neighbors.data(),
                                BetaJS_row,
                                even_sites,
                                odd_sites,
                                parity,
                                state.data(),
                                udist,
                                rng);
                        }

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

void run_ising_solver::wolff_solver() {
    int N = L * L;
    int W = num_words(N);
    auto p = params::defaults(L);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    std::vector<int> neighbors = neighbors_2d_nn(L);
    std::vector<double> Padd = Padd_2d_nn(temp_schedule, p.J);

    std::cout << "Runs start on " << num_threads << " threads" << std::endl;

    int start = run_start;
    for (int t = 0; t < num_threads; t++) {
        int end = start + runs_per_thread + (t < extra ? 1 : 0);

        threads.emplace_back([&, start, end]() {
            std::vector<int> stack;
            stack.reserve(N);          // allocated once per thread
            std::uniform_real_distribution<double> udist(0.0, 1.0);
            std::uniform_int_distribution<int>    site_dist(0, N - 1);

            for (int run = start; run < end; run++) {
                std::mt19937 rng(std::random_device{}() + run);

                // Working state for this run — lives on the thread's stack
                std::vector<uint64_t> state(W);
                random_binary_state(state.data(), N, rng);

                for (int temp_index = 0; temp_index < p.temp_updates; temp_index++) {
                    StateWriter writer(temp_dirs[temp_index], run);
                    
                    for (int sweep = 0; sweep < p.wolff_sweeps_per_temp; sweep++) {
                        wolff_cluster(
                            neighbors.data(),
                            Padd[temp_index],
                            state.data(),
                            stack,
                            udist,
                            site_dist,
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

void run_ising_solver::nnn_2d_solver() {
    int N = L * L;
    int W = num_words(N);
    auto p = params::defaults(L);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    std::vector<int> nn_neighbors = neighbors_2d_nn(L);
    std::vector<int> nnn_neighbors = neighbors_2d_nnn(L);
    
    std::cout << "Runs start on " << num_threads << " threads" << std::endl;

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

                for (int temp_index = 0; temp_index < p.temp_updates; temp_index++) {
                    StateWriter writer(temp_dirs[temp_index], run);
                    
                    for (int sweep = 0; sweep < p.metro_sweeps_per_temp; sweep++) {
                        nnn_2d(
                            N,
                            p.J1,
                            p.J2,
                            nn_neighbors.data(),
                            nnn_neighbors.data(),
                            beta_schedule[temp_index],
                            state.data(),
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

void run_ising_solver::triangular_solver(int Lx, int Ly) {
    int N = Lx * Ly;
    int W = num_words(N);
    auto p = params::defaults(Lx);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    std::vector<int> neighbors = neighbors_triangular(Lx, Ly);
    std::vector<double> BetaJS = BetaJS_triangular(temp_schedule, p.J);

    std::cout << "Runs start on " << num_threads << " threads" << std::endl;

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

                for (int temp_index = 0; temp_index < p.temp_updates; temp_index++) {
                    const double* BetaJS_row = &BetaJS[temp_index * 7];
                    StateWriter writer(temp_dirs[temp_index], run);
                    
                    for (int sweep = 0; sweep < p.metro_sweeps_per_temp; sweep++) {
                        triangular(
                            N, 
                            neighbors.data(),
                            BetaJS_row,
                            state.data(),
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

void run_ising_solver::honeycomb_solver(int Lx, int Ly) {
    int N = Lx * Ly;
    int W = num_words(N);
    auto p = params::defaults(Lx);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    std::vector<int> neighbors = neighbors_honeycomb(Lx, Ly);
    std::vector<double> BetaJS = BetaJS_honeycomb(temp_schedule, p.J);

    std::cout << "Runs start on " << num_threads << " threads" << std::endl;

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

                for (int temp_index = 0; temp_index < p.temp_updates; temp_index++) {
                    const double* BetaJS_row = &BetaJS[temp_index * 4];
                    StateWriter writer(temp_dirs[temp_index], run);
                    
                    for (int sweep = 0; sweep < p.metro_sweeps_per_temp; sweep++) {
                        honeycomb(
                            N, 
                            neighbors.data(),
                            BetaJS_row,
                            state.data(),
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

void run_ising_solver::nn_3d_solver() {
    int N = L * L * L;
    int W = num_words(N);
    auto p = params::defaults(L);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    std::vector<int> neighbors = neighbors_3d_nn(L);
    std::vector<double> BetaJS = BetaJS_3d_nn(temp_schedule, params::defaults(L).J);

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

    std::cout << "Runs start on " << num_threads << " threads" << std::endl;

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

                for (int temp_index = 0; temp_index < p.temp_updates; temp_index++) {
                    const double* BetaJS_row = &BetaJS[temp_index * 7];
                    StateWriter writer(temp_dirs[temp_index], run);
                    
                    for (int sweep = 0; sweep < p.metro_sweeps_per_temp; sweep++) {
                        for (int parity = 0; parity < 2; parity++) {
                            nn_3d(
                                neighbors.data(),
                                BetaJS_row,
                                even_sites,
                                odd_sites,
                                parity,
                                state.data(),
                                udist,
                                rng);
                        }

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

void run_ising_solver::nnn_3d_solver() {
    int N = L * L;
    int W = num_words(N);
    auto p = params::defaults(L);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    std::vector<int> nn_neighbors = neighbors_3d_nn(L);
    std::vector<int> nnn_neighbors = neighbors_3d_nnn(L);
    
    std::cout << "Runs start on " << num_threads << " threads" << std::endl;

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

                for (int temp_index = 0; temp_index < p.temp_updates; temp_index++) {
                    StateWriter writer(temp_dirs[temp_index], run);
                    
                    for (int sweep = 0; sweep < p.metro_sweeps_per_temp; sweep++) {
                        nnn_3d(
                            N,
                            p.J1,
                            p.J2,
                            nn_neighbors.data(),
                            nnn_neighbors.data(),
                            beta_schedule[temp_index],
                            state.data(),
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




