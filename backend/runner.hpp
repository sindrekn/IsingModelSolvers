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

    void quadratic_solver();
    void wolff_solver();
    void quadratic_nnn_solver();
    void triangular_solver(int Lx, int Ly);
    void honeycomb_solver(int Lx, int Ly);
    void cubic_solver();
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
    std::vector<std::thread> threads;

    int runs_per_thread = (run_end - run_start) / num_threads;
    int extra           = (run_end - run_start) % num_threads;

    std::vector<double> beta_schedule;
};

#endif // BACKEND_RUNNER_HPP