#ifndef BACKEND_RUNNER_HPP
#define BACKEND_RUNNER_HPP

#include <vector>
#include <thread>
#include <random>
#include <algorithm>
#include <cstring>

// ----------------------------------------------------------
// Run multiple independent simulations in parallel 
// ----------------------------------------------------------

void LRIM_metropolis(
    int L,
    double sigma,
    int num_threads,
    int run_start,
    int run_end,
    std::vector<double> temp_schedule,
    std::vector<std::string> temp_dirs
    );

void LRIM_LB(
    int L,
    double sigma,
    int num_threads,
    int run_start,
    int run_end,
    std::vector<double> temp_schedule,
    std::vector<std::string> temp_dirs
    );

#endif // BACKEND_RUNNER_HPP