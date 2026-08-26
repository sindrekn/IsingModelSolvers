#ifndef SOLVERS_NN_2D_HPP
#define SOLVERS_NN_2D_HPP

#include <vector>
#include <random>
#include <cstdint>
#include <cstring>
#include <thread>

#include "core/lattice/lattice.hpp"

// ----------------------------------------------------------
// The 2D Nearest Neighbor Metropolis spin update algorithm
// ----------------------------------------------------------

// Metropolis sweep with checkerboard (even/odd sublattice) updates.
void nn_2d(
    const int*              neighbors,
    const double*           BetaJS_row,
    const std::vector<int>& even_sites,
    const std::vector<int>& odd_sites,
    int                     parity,
    uint64_t*               state,
    std::uniform_real_distribution<double>& udist,
    std::mt19937&           rng); 

#endif // SOLVERS_NN_2D_HPP