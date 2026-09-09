#ifndef BACKEND_RUNNER_HPP
#define BACKEND_RUNNER_HPP

#include <vector>
#include <thread>
#include <random>
#include <algorithm>
#include <cstring>
#include <iostream>

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

    void nn_2d_solver();
    void wolff_solver();
    void nnn_2d_solver();
    void triangular_solver(int Lx, int Ly);
    void nn_3d_solver();
    void nnn_3d_solver();

private:
    int L;
    int num_threads;
    int run_start;
    int run_end;
    std::vector<double> temp_schedule;
    std::vector<std::string> temp_dirs;
    std::vector<std::thread> threads;

    int runs_per_thread = (run_end - run_start) / num_threads;
    int extra           = (run_end - run_start) % num_threads;

    std::vector<double> beta_schedule;
};

#endif // BACKEND_RUNNER_HPP