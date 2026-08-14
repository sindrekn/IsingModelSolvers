#ifndef CORE_LATTICE_HPP
#define CORE_LATTICE_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>

// ----------------------------------------------------------
// Precompute the distance and interaction strength for a 2D 
// lattice with periodic boundary conditions. The interaction
// strength decays as 1/r^(2+sigma) where r is the distance 
// between two sites.
// ----------------------------------------------------------

struct LongRange_DistanceResult {
    std::vector<int> dx;
    std::vector<int> dy;
    std::vector<float> J_r;
};

LongRange_DistanceResult LongRange_DistancePrecompute(int L, float sigma);

struct LB_ClusterPrecomputeResult {
    std::vector<std::vector<float>> scale;
    std::vector<std::vector<float>> cumulativeProb;
};

LB_ClusterPrecomputeResult LB_ClusterPrecompute(
    const LongRange_DistanceResult& distanceResult,
    const std::vector<float> beta_schedule,
    int L
    );

#endif // CORE_LATTICE_HPP
