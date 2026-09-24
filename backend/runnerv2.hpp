#ifndef BACKEND_RUNNERV2_HPP
#define BACKEND_RUNNERV2_HPP

#include <vector>
#include <thread>
#include <random>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <functional>
#include <utility>
#include <cstdint>

class run_ising_solver
{
public:
    run_ising_solver(
        int L,
        int num_threads,
        int run_start,
        int run_end,
        const std::vector<double>& temp_schedule,
        const std::vector<std::string>& temp_dirs
    );

    void quadratic_solver();
    void wolff_quadratic_solver();
    void quadratic_nnn_solver();
    void triangular_solver(int Lx, int Ly);
    void wolff_triangular_solver(int Lx, int Ly);
    void honeycomb_solver(int Lx, int Ly);
    void wolff_honeycomb_solver(int Lx, int Ly);
    void cubic_solver();
    void wolff_cubic_solver();
    void cubic_nnn_solver();
    void lrim(double sigma);
    void lb(double sigma);

private:
    int L;
    int num_threads;
    int run_start;
    int run_end;
    std::vector<double> temp_schedule;
    std::vector<std::string> temp_dirs;

    int runs_per_thread = (run_end - run_start) / num_threads;
    int extra           = (run_end - run_start) % num_threads;

    std::vector<double> beta_schedule;

    // --- shared machinery -------------------------------------------------
    //
    // Every solver method used to hand-roll: thread partitioning, per-run
    // RNG + state init, a temperature loop, a StateWriter, a sweep loop and
    // periodic snapshotting. Only two things actually differ between
    // methods:
    //   1) what a single sweep at a given temperature does, and
    //   2) (only for `lrim`) something that must be rebuilt whenever a new
    //      run's state has just been (re)randomized.
    //
    // run_generic() owns everything else. Each solver method supplies a
    // "thread factory": called once per worker thread, it builds any
    // thread-persistent scratch space (e.g. a Wolff cluster stack, which
    // must be allocated once per thread and reused across runs) and
    // returns the {run_init, sweep} pair that thread will use.

    // Called once per run, right after that run's state has been
    // randomized. Leave default-constructed (empty) if not needed.
    using RunInit = std::function<void(uint64_t* state)>;

    // Called once per sweep, at a given temperature index.
    using SweepFn = std::function<void(int temp_index, uint64_t* state, std::mt19937& rng)>;

    using ThreadFactory = std::function<
        std::pair<RunInit, SweepFn>(std::uniform_real_distribution<double>& udist)>;

    void run_generic(
        int N,
        int temp_updates,
        int store_step,
        int sweeps_per_temp,
        const ThreadFactory& make_worker
    );
};

#endif // BACKEND_RUNNERV2_HPP