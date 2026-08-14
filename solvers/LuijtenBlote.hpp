#ifndef SOLVERS_LUIJTENBLOTECLUSTER_HPP
#define SOLVERS_LUIJTENBLOTECLUSTER_HPP

#include <vector>
#include <random>
#include <cstdint>
#include <cstring>

#include "core/lattice/lattice.hpp"

// ----------------------------------------------------------
// The 2D Long-Range Ising Model with Luijten-Blote cluster update 
// ----------------------------------------------------------
void LuijtenBloteCluster(
    const LB_ClusterPrecomputeResult& clusterResult,
    const LongRange_DistanceResult&      distanceResult,
    int                     L,
    uint64_t*               state,
    int                     beta_id,
    std::uniform_real_distribution<double>& udist,
    std::mt19937&           rng
    );

#endif // SOLVERS_LUIJTENBLOTECLUSTER_HPP