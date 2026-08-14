#ifndef SOLVERS_METROPOLIS_HPP
#define SOLVERS_METROPOLIS_HPP

#include <vector>
#include <random>
#include <cstdint>
#include <cstring>
#include <thread>

#include "core/lattice/lattice.hpp"

std::vector<float> construct_h(
    const int N,
    uint64_t* state,
    const LongRange_DistanceResult& distanceResult
);

// Long-range Ising model update algorithm.
void update_h(
    const LongRange_DistanceResult& distanceResult,
    std::vector<float>& h, 
    int id, 
    int s_i, 
    int N, 
    int L
    );

// ----------------------------------------------------------
// The 2D Long-Range Ising Model Solver
// ----------------------------------------------------------
void FieldUpdate_solver(
    const LongRange_DistanceResult& distanceResult,
    std::vector<float>&     h,
    int                     N, 
    int                     L,
    uint64_t*               state,
    float                   Beta,
    std::uniform_real_distribution<double>& udist,
    std::mt19937&           rng
    );

#endif // SOLVERS_METROPOLIS_HPP