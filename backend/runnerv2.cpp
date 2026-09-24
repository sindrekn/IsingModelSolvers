#include "runnerv2.hpp"

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

// ---------------------------------------------------------------------
// Shared scaffolding: thread partitioning, per-run RNG + state init,
// the temperature loop, and periodic snapshotting. See runner.hpp for
// the rationale.
// ---------------------------------------------------------------------
void run_ising_solver::run_generic(
    int N,
    int temp_updates,
    int store_step,
    int sweeps_per_temp,
    const ThreadFactory& make_worker)
{
    int W = num_words(N);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int start = run_start;
    for (int t = 0; t < num_threads; t++) {
        int end = start + runs_per_thread + (t < extra ? 1 : 0);

        threads.emplace_back([&, start, end]() {
            std::uniform_real_distribution<double> udist(0.0, 1.0);
            auto [run_init, sweep] = make_worker(udist);

            for (int run = start; run < end; run++) {
                std::mt19937 rng(std::random_device{}() + run);

                // Working state for this run — lives on the thread's stack
                std::vector<uint64_t> state(W);
                random_binary_state(state.data(), N, rng);
                if (run_init) run_init(state.data());

                for (int temp_index = 0; temp_index < temp_updates; temp_index++) {
                    StateWriter writer(temp_dirs[temp_index], run);

                    for (int s = 0; s < sweeps_per_temp; s++) {
                        sweep(temp_index, state.data(), rng);

                        if (s % store_step == 0) {
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
}

// ---------------------------------------------------------------------
// 2D quadratic lattice, nearest-neighbor, checkerboard local updates
// ---------------------------------------------------------------------
void run_ising_solver::quadratic_solver() {
    int N = L * L;
    auto p = params::defaults(L);

    std::vector<int> neighbors = neighbors_quadratic(L);
    std::vector<double> BetaJS = BetaJS_quadratic(temp_schedule, p.J);

    std::vector<int> even_sites, odd_sites;
    even_sites.reserve(N / 2);
    odd_sites.reserve(N / 2);
    for (int id = 0; id < N; id++) {
        if (((id / L) + (id % L)) % 2 == 0) even_sites.push_back(id);
        else odd_sites.push_back(id);
    }

    std::cout << "2D Ising solver with nearest-neighbor interactions on a quadratic lattice of size " << L << "x" << L << std::endl;
    std::cout << "Runs number " << run_start << " to " << run_end << " on " << num_threads << " threads" << std::endl;

    run_generic(N, p.temp_updates, p.store_step, p.sweeps_per_temp,
        [&](std::uniform_real_distribution<double>& udist) {
            SweepFn sweep = [&](int temp_index, uint64_t* state, std::mt19937& rng) {
                const double* row = &BetaJS[temp_index * 5];
                for (int parity = 0; parity < 2; parity++) {
                    quadratic(neighbors.data(), row, even_sites, odd_sites, parity, state, udist, rng);
                }
            };
            return std::make_pair(RunInit{}, sweep);
        });
}

// ---------------------------------------------------------------------
// 2D quadratic lattice, nearest-neighbor, Wolff cluster updates
// ---------------------------------------------------------------------
void run_ising_solver::wolff_quadratic_solver() {
    int N = L * L;
    auto p = params::defaults(L);

    std::vector<int> neighbors = neighbors_quadratic(L);
    int num_neighbors = 4;
    std::vector<double> Padd = Precompute_Padd(temp_schedule, p.J);

    std::cout << "2D Ising solver with nearest-neighbor interactions on a quadratic lattice of size " << L << "x" << L << " with wolff algorithm" << std::endl;
    std::cout << "Runs number " << run_start << " to " << run_end << " on " << num_threads << " threads" << std::endl;

    run_generic(N, p.temp_updates, p.store_step, p.sweeps_per_temp,
        [&](std::uniform_real_distribution<double>& udist) {
            // Allocated once per thread, reused across every run/sweep it handles.
            std::vector<int> stack;
            stack.reserve(N);
            std::uniform_int_distribution<int> sitedist(0, N - 1);

            SweepFn sweep = [&, stack, sitedist](int temp_index, uint64_t* state, std::mt19937& rng) mutable {
                wolff_cluster(num_neighbors, neighbors.data(), Padd[temp_index], state, stack, udist, sitedist, rng);
            };
            return std::make_pair(RunInit{}, sweep);
        });
}

// ---------------------------------------------------------------------
// 2D quadratic lattice, next-nearest-neighbor
// ---------------------------------------------------------------------
void run_ising_solver::quadratic_nnn_solver() {
    int N = L * L;
    auto p = params::defaults(L);

    std::vector<int> nn_neighbors  = neighbors_quadratic(L);
    std::vector<int> nnn_neighbors = neighbors_quadratic_nnn(L);

    std::cout << "2D Ising solver with next-nearest-neighbor interactions on a quadratic lattice of size " << L << "x" << L << std::endl;
    std::cout << "Runs number " << run_start << " to " << run_end << " on " << num_threads << " threads" << std::endl;

    run_generic(N, p.temp_updates, p.store_step, p.sweeps_per_temp,
        [&](std::uniform_real_distribution<double>& udist) {
            SweepFn sweep = [&](int temp_index, uint64_t* state, std::mt19937& rng) {
                quadratic_nnn(N, p.J1, p.J2, nn_neighbors.data(), nnn_neighbors.data(),
                              beta_schedule[temp_index], state, udist, rng);
            };
            return std::make_pair(RunInit{}, sweep);
        });
}

// ---------------------------------------------------------------------
// 2D triangular lattice, nearest-neighbor, local updates
// ---------------------------------------------------------------------
void run_ising_solver::triangular_solver(int Lx, int Ly) {
    int N = Lx * Ly;
    auto p = params::defaults(Lx);

    std::vector<int> neighbors = neighbors_triangular(Lx, Ly);
    std::vector<double> BetaJS = BetaJS_triangular(temp_schedule, p.J);

    std::cout << "2D Ising solver with nearest-neighbor interactions on a triangular lattice of size " << Lx << "x" << Ly << std::endl;
    std::cout << "Runs number " << run_start << " to " << run_end << " on " << num_threads << " threads" << std::endl;

    run_generic(N, p.temp_updates, p.store_step, p.sweeps_per_temp,
        [&](std::uniform_real_distribution<double>& udist) {
            SweepFn sweep = [&](int temp_index, uint64_t* state, std::mt19937& rng) {
                const double* row = &BetaJS[temp_index * 7];
                triangular(N, neighbors.data(), row, state, udist, rng);
            };
            return std::make_pair(RunInit{}, sweep);
        });
}

// ---------------------------------------------------------------------
// 2D triangular lattice, nearest-neighbor, Wolff cluster updates
// ---------------------------------------------------------------------
void run_ising_solver::wolff_triangular_solver(int Lx, int Ly) {
    int N = Lx * Ly;
    auto p = params::defaults(Lx);

    std::vector<int> neighbors = neighbors_triangular(Lx, Ly);
    int num_neighbors = 6;
    std::vector<double> Padd = Precompute_Padd(temp_schedule, p.J);

    std::cout << "2D Ising solver with nearest-neighbor interactions on a triangular lattice of size " << Lx << "x" << Ly << " with wolff algorithm" << std::endl;
    std::cout << "Runs number " << run_start << " to " << run_end << " on " << num_threads << " threads" << std::endl;

    run_generic(N, p.temp_updates, p.store_step, p.sweeps_per_temp,
        [&](std::uniform_real_distribution<double>& udist) {
            std::vector<int> stack;
            stack.reserve(N);
            std::uniform_int_distribution<int> sitedist(0, N - 1);

            SweepFn sweep = [&, stack, sitedist](int temp_index, uint64_t* state, std::mt19937& rng) mutable {
                wolff_cluster(num_neighbors, neighbors.data(), Padd[temp_index], state, stack, udist, sitedist, rng);
            };
            return std::make_pair(RunInit{}, sweep);
        });
}

// ---------------------------------------------------------------------
// 2D honeycomb lattice, nearest-neighbor, local updates
// ---------------------------------------------------------------------
void run_ising_solver::honeycomb_solver(int Lx, int Ly) {
    int N = Lx * Ly;
    auto p = params::defaults(Lx);

    std::vector<int> neighbors = neighbors_honeycomb(Lx, Ly);
    std::vector<double> BetaJS = BetaJS_honeycomb(temp_schedule, p.J);

    std::cout << "2D Ising solver with nearest-neighbor interactions on a honeycomb lattice of size " << Lx << "x" << Ly << std::endl;
    std::cout << "Runs number " << run_start << " to " << run_end << " on " << num_threads << " threads" << std::endl;

    run_generic(N, p.temp_updates, p.store_step, p.sweeps_per_temp,
        [&](std::uniform_real_distribution<double>& udist) {
            SweepFn sweep = [&](int temp_index, uint64_t* state, std::mt19937& rng) {
                const double* row = &BetaJS[temp_index * 4];
                honeycomb(N, neighbors.data(), row, state, udist, rng);
            };
            return std::make_pair(RunInit{}, sweep);
        });
}

// ---------------------------------------------------------------------
// 2D honeycomb lattice, nearest-neighbor, Wolff cluster updates
// ---------------------------------------------------------------------
void run_ising_solver::wolff_honeycomb_solver(int Lx, int Ly) {
    int N = Lx * Ly;
    auto p = params::defaults(Lx);

    std::vector<int> neighbors = neighbors_honeycomb(Lx, Ly);
    int num_neighbors = 3;
    std::vector<double> Padd = Precompute_Padd(temp_schedule, p.J);

    std::cout << "2D Ising solver with nearest-neighbor interactions on a honeycomb lattice of size " << Lx << "x" << Ly << " with wolff algorithm" << std::endl;
    std::cout << "Runs number " << run_start << " to " << run_end << " on " << num_threads << " threads" << std::endl;

    run_generic(N, p.temp_updates, p.store_step, p.sweeps_per_temp,
        [&](std::uniform_real_distribution<double>& udist) {
            std::vector<int> stack;
            stack.reserve(N);
            std::uniform_int_distribution<int> sitedist(0, N - 1);

            SweepFn sweep = [&, stack, sitedist](int temp_index, uint64_t* state, std::mt19937& rng) mutable {
                wolff_cluster(num_neighbors, neighbors.data(), Padd[temp_index], state, stack, udist, sitedist, rng);
            };
            return std::make_pair(RunInit{}, sweep);
        });
}

// ---------------------------------------------------------------------
// 3D cubic lattice, nearest-neighbor, checkerboard local updates
// ---------------------------------------------------------------------
void run_ising_solver::cubic_solver() {
    int N = L * L * L;
    auto p = params::defaults(L);

    std::vector<int> neighbors = neighbors_cubic(L);
    std::vector<double> BetaJS = BetaJS_cubic(temp_schedule, p.J);

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

    std::cout << "3D Ising solver with nearest-neighbor interactions on a cubic lattice of size " << L << "x" << L << "x" << L << std::endl;
    std::cout << "Runs number " << run_start << " to " << run_end << " on " << num_threads << " threads" << std::endl;

    run_generic(N, p.temp_updates, p.store_step, p.sweeps_per_temp_3d,
        [&](std::uniform_real_distribution<double>& udist) {
            SweepFn sweep = [&](int temp_index, uint64_t* state, std::mt19937& rng) {
                const double* row = &BetaJS[temp_index * 7];
                for (int parity = 0; parity < 2; parity++) {
                    cubic(neighbors.data(), row, even_sites, odd_sites, parity, state, udist, rng);
                }
            };
            return std::make_pair(RunInit{}, sweep);
        });
}

// ---------------------------------------------------------------------
// 3D cubic lattice, nearest-neighbor, Wolff cluster updates
// ---------------------------------------------------------------------
void run_ising_solver::wolff_cubic_solver() {
    int N = L * L * L;
    auto p = params::defaults(L);

    std::vector<int> neighbors = neighbors_cubic(L);
    int num_neighbors = 6;
    std::vector<double> Padd = Precompute_Padd(temp_schedule, p.J);

    std::cout << "3D Ising solver with nearest-neighbor interactions on a cubic lattice of size " << L << "x" << L << "x" << L << " with wolff algorithm" << std::endl;
    std::cout << "Runs number " << run_start << " to " << run_end << " on " << num_threads << " threads" << std::endl;

    // NOTE: kept at p.sweeps_per_temp (not p.sweeps_per_temp_3d), matching
    // the original code — worth double-checking whether that was intended.
    run_generic(N, p.temp_updates, p.store_step, p.sweeps_per_temp,
        [&](std::uniform_real_distribution<double>& udist) {
            std::vector<int> stack;
            stack.reserve(N);
            std::uniform_int_distribution<int> sitedist(0, N - 1);

            SweepFn sweep = [&, stack, sitedist](int temp_index, uint64_t* state, std::mt19937& rng) mutable {
                wolff_cluster(num_neighbors, neighbors.data(), Padd[temp_index], state, stack, udist, sitedist, rng);
            };
            return std::make_pair(RunInit{}, sweep);
        });
}

// ---------------------------------------------------------------------
// 3D cubic lattice, next-nearest-neighbor
// ---------------------------------------------------------------------
void run_ising_solver::cubic_nnn_solver() {
    int N = L * L * L;
    auto p = params::defaults(L);

    std::vector<int> nn_neighbors  = neighbors_cubic(L);
    std::vector<int> nnn_neighbors = neighbors_cubic_nnn(L);

    std::cout << "3D Ising solver with next-nearest-neighbor interactions on a cubic lattice of size " << L << "x" << L << "x" << L << std::endl;
    std::cout << "Runs number " << run_start << " to " << run_end << " on " << num_threads << " threads" << std::endl;

    run_generic(N, p.temp_updates, p.store_step, p.sweeps_per_temp_3d,
        [&](std::uniform_real_distribution<double>& udist) {
            SweepFn sweep = [&](int temp_index, uint64_t* state, std::mt19937& rng) {
                cubic_nnn(N, p.J1, p.J2, nn_neighbors.data(), nnn_neighbors.data(),
                          beta_schedule[temp_index], state, udist, rng);
            };
            return std::make_pair(RunInit{}, sweep);
        });
}

// ---------------------------------------------------------------------
// Long-range interactions, local field updates
// ---------------------------------------------------------------------
void run_ising_solver::lrim(double sigma) {
    int N = L * L;
    auto p = params::defaults(L);

    LongRange_DistanceResult distanceResult = LongRange_DistancePrecompute(L, sigma);

    std::cout << "Long-range interactions on a quadratic lattice of size " << L << "x" << L << std::endl;
    std::cout << "Runs number " << run_start << " to " << run_end << " on " << num_threads << " threads" << std::endl;

    run_generic(N, p.temp_updates, p.store_step, p.sweeps_per_temp,
        [&](std::uniform_real_distribution<double>& udist) {
            // h_local depends on each run's freshly-randomized state, so it
            // has to be rebuilt in run_init rather than precomputed once
            // here. Shared between run_init and sweep via shared_ptr since
            // they're two independent closures.
            auto h_local = std::make_shared<std::vector<double>>();

            RunInit run_init = [&, h_local](uint64_t* state) mutable {
                *h_local = construct_h(N, state, distanceResult);
            };
            SweepFn sweep = [&, h_local](int temp_index, uint64_t* state, std::mt19937& rng) {
                FieldUpdate(distanceResult, *h_local, N, L, state, beta_schedule[temp_index], udist, rng);
            };
            return std::make_pair(run_init, sweep);
        });
}

// ---------------------------------------------------------------------
// Long-range interactions, Luijten-Blote cluster updates
// ---------------------------------------------------------------------
void run_ising_solver::lb(double sigma) {
    int N = L * L;
    auto p = params::defaults(L);

    LongRange_DistanceResult distanceResult = LongRange_DistancePrecompute(L, sigma);
    LB_ClusterPrecomputeResult clusterResult = LB_ClusterPrecompute(distanceResult, beta_schedule, L);

    std::cout << "Long-range interactions with Luijten-Blote cluster updates on a quadratic lattice of size " << L << "x" << L << std::endl;
    std::cout << "Runs number " << run_start << " to " << run_end << " on " << num_threads << " threads" << std::endl;

    run_generic(N, p.temp_updates, p.store_step, p.sweeps_per_temp,
        [&](std::uniform_real_distribution<double>& udist) {
            SweepFn sweep = [&](int temp_index, uint64_t* state, std::mt19937& rng) {
                LuijtenBloteCluster(clusterResult, distanceResult, N, L, state, temp_index, udist, rng);
            };
            return std::make_pair(RunInit{}, sweep);
        });
}