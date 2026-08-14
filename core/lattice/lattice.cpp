#include "lattice.hpp"

LongRange_DistanceResult LongRange_DistancePrecompute(int L, float sigma)
{
    // Ensure L is a positive integer dividable by 2 for periodic boundary conditions
    if (L <= 0 || L % 2 != 0) {
        throw std::invalid_argument("L must be a positive even integer.");
    }

    int N = L * L;
    LongRange_DistanceResult result;

    result.dx.resize(N - 1);
    result.dy.resize(N - 1);
    result.J_r.resize(N - 1);

    // Precompute all distances from a reference site (e.g., site 0) to all other sites
    std::vector<float> r(N - 1);
    int id = 0; 
    for (int x = -L/2; x < L/2; x++) {
        for (int y = -L/2; y < L/2; y++) {
            if (x == 0 && y == 0) continue; // Skip the origin
            result.dx[id] = x;
            result.dy[id] = y;
            r[id] = std::sqrt(x*x + y*y);
            id += 1; 
        }
    }

    // Determine the coupeling constant based on the system size and sigma
    float sum = 0.0f;
    for (int j = 0; j < N - 1; ++j) {
        sum += 1.0f / std::pow(r[j], (2.0f + sigma));
    }

    float c_norm = 4.0f / sum; // Normalization constant

    for (int j = 0; j < N - 1; ++j) {
        result.J_r[j] = c_norm / std::pow(r[j], (2.0f + sigma));
    }

    // Create a vector of indices [0, 1, 2, ..., N-2]
    std::vector<size_t> indices(N - 1);
    std::iota(indices.begin(), indices.end(), 0);

    // Sort the indices based on the decreasing values of result.J_r
    std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
        return result.J_r[i] > result.J_r[j];
    });

    // Reorder the vectors using the sorted indices
    std::vector<int> sorted_dx(N - 1);
    std::vector<int> sorted_dy(N - 1);
    std::vector<float> sorted_r(N - 1);

    for (size_t i = 0; i < indices.size(); ++i) {
        sorted_dx[i] = result.dx[indices[i]];
        sorted_dy[i] = result.dy[indices[i]];
        sorted_r[i]  = result.J_r[indices[i]];
    }

    // Move the sorted vectors back into the result struct
    result.dx = std::move(sorted_dx);
    result.dy = std::move(sorted_dy);
    result.J_r  = std::move(sorted_r);

    return result;
}

LB_ClusterPrecomputeResult LB_ClusterPrecompute(
    const LongRange_DistanceResult& distanceResult,
    const std::vector<float> beta_schedule,
    int L 
)
    /*
        Precompute the cumulative distribution function (CDF) for the Luijten-Blote cluster update.
    */
{
    int N = L*L; 
    int num_temps = beta_schedule.size();

    LB_ClusterPrecomputeResult clusterResult;

    std::vector<std::vector<float>> bondProb(N - 1);
    std::vector<std::vector<float>> noBondProb(N - 1);

    clusterResult.cumulativeProb.resize(N - 1);
    clusterResult.scale.resize(N - 1);

    for (int k = 0; k < N - 1; ++k)
    {
        bondProb[k].resize(num_temps);
        noBondProb[k].resize(num_temps);
        clusterResult.cumulativeProb[k].resize(num_temps);
        clusterResult.scale[k].resize(num_temps);
    }

    for (int id = 0; id < num_temps; ++id) {
        // Compute the bondProb for each distance (p)
        for (int k = 0; k < N - 1; ++k) {
            float J_k = distanceResult.J_r[k];
            bondProb[k][id] = 1.0f - std::exp(-2.0f * beta_schedule[id] * J_k);
        }

        // Compute noBondProb (q)
        noBondProb[0][id] = bondProb[0][id];
        float product = 1.0f; 
        for (int l = 1; l < N - 1; l++) {
            product *= (1.0f - bondProb[l - 1][id]);
            noBondProb[l][id] = bondProb[l][id] * product;
        }

        // Compute cumulativeProb (C)
        clusterResult.cumulativeProb[0][id] = noBondProb[0][id];
        for (int m = 1; m < N - 1; m++) {
            clusterResult.cumulativeProb[m][id] = clusterResult.cumulativeProb[m - 1][id] + noBondProb[m][id];
        }
    }

    for (int id = 0; id < num_temps; ++id) {
        for (int m = 0; m < N - 1; ++m) {
            clusterResult.scale[m][id] = bondProb[m][id] / noBondProb[m][id];
        }
    }

    return clusterResult;
}



