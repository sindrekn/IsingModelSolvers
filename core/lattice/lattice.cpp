#include "lattice.hpp"

std::vector<int> neighbors_quadratic(int L) {
    int N = L * L;
    std::vector<int> neigh(N * 4);

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int id = i * L + j;
            neigh[id * 4 + 0] = ((i - 1 + L) % L) * L + j;      // up
            neigh[id * 4 + 1] = ((i + 1) % L) * L + j;          // down
            neigh[id * 4 + 2] = i * L + ((j - 1 + L) % L);      // left
            neigh[id * 4 + 3] = i * L + ((j + 1) % L);          // right
        }
    }
    return neigh;
}

std::vector<int> neighbors_quadratic_nnn(int L) {
    int N = L * L;
    std::vector<int> neigh(N * 4);

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int id = i * L + j;
            neigh[id * 4 + 0] = ((i - 1 + L) % L) * L + ((j - 1 + L) % L);  // up-left
            neigh[id * 4 + 1] = ((i - 1 + L) % L) * L + ((j + 1) % L);      // up-right
            neigh[id * 4 + 2] = ((i + 1) % L) * L + ((j - 1 + L) % L);      // down-left
            neigh[id * 4 + 3] = ((i + 1) % L) * L + ((j + 1) % L);          // down-right
        }
    }
    return neigh;
}

std::vector<int> neighbors_triangular(int Lx, int Ly) {
    if (Lx <= 0 || Ly <= 0) {
        throw std::invalid_argument("Lattice dimensions must be positive.");
    }

    if (Lx % 2 != 0) {
        throw std::invalid_argument("Lattice width (Lx) must be even for triangular lattice.");
    }

    if (Ly % 2 != 0) {
        throw std::invalid_argument("Lattice height (Ly) must be even for triangular lattice.");
    }

    int N = Lx * Ly;
    std::vector<int> neigh(N * 6);

    for (int i = 0; i < Ly; i++) {
        for (int j = 0; j < Lx; j++) {

            int id = i * Lx + j;

            int im = (i - 1 + Ly) % Ly;
            int ip = (i + 1) % Ly;
            int jm = (j - 1 + Lx) % Lx;
            int jp = (j + 1) % Lx;

            if (i % 2 == 0) {

                neigh[id * 6 + 0] = im * Lx + jm;
                neigh[id * 6 + 1] = im * Lx + j;
                neigh[id * 6 + 2] = i  * Lx + jm;
                neigh[id * 6 + 3] = i  * Lx + jp;
                neigh[id * 6 + 4] = ip * Lx + jm;
                neigh[id * 6 + 5] = ip * Lx + j;

            } else {

                neigh[id * 6 + 0] = im * Lx + j;
                neigh[id * 6 + 1] = im * Lx + jp;
                neigh[id * 6 + 2] = i  * Lx + jm;
                neigh[id * 6 + 3] = i  * Lx + jp;
                neigh[id * 6 + 4] = ip * Lx + j;
                neigh[id * 6 + 5] = ip * Lx + jp;
            }
        }
    }

    return neigh;
}

std::vector<int> neighbors_honeycomb(int Lx, int Ly) {
    if (Lx <= 0 || Ly <= 0) {
        throw std::invalid_argument("Lattice dimensions must be positive.");
    }

    if (Lx % 2 != 0) {
        throw std::invalid_argument("Lattice width (Lx) must be even for honeycomb lattice.");
    }

    if (Ly % 2 != 0) {
        throw std::invalid_argument("Lattice height (Ly) must be even for honeycomb lattice.");
    }

    int N = Lx * Ly;
    std::vector<int> neigh(N * 3);

    for (int i = 0; i < Ly; i++) {
        for (int j = 0; j < Lx; j++) {

            int id = i * Lx + j;

            int im = (i - 1 + Ly) % Ly;
            int ip = (i + 1) % Ly;
            int jm = (j - 1 + Lx) % Lx;
            int jp = (j + 1) % Lx;

            if ((i + j) % 2 == 0) {
                neigh[id * 3 + 0] = im * Lx + j; // up
                neigh[id * 3 + 1] = i * Lx + jp; // right
                neigh[id * 3 + 2] = ip * Lx + j; // down
            } else {
                neigh[id * 3 + 0] = im * Lx + j; // up
                neigh[id * 3 + 1] = i * Lx + jm; // left
                neigh[id * 3 + 2] = ip * Lx + j; // down
            }
        }
    }

    return neigh;
}

std::vector<int> neighbors_cubic(int L) {
    int N = L * L * L;
    std::vector<int> neigh(N * 6);

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < L; k++) {
                int id = i * L * L + j * L + k;
                neigh[id * 6 + 0] = ((i - 1 + L) % L) * L * L + j * L + k; // -i
                neigh[id * 6 + 1] = ((i + 1) % L) * L * L + j * L + k;     // +i
                neigh[id * 6 + 2] = i * L * L + ((j - 1 + L) % L) * L + k; // -j
                neigh[id * 6 + 3] = i * L * L + ((j + 1) % L) * L + k;     // +j
                neigh[id * 6 + 4] = i * L * L + j * L + ((k - 1 + L) % L); // -k
                neigh[id * 6 + 5] = i * L * L + j * L + ((k + 1) % L);     // +k
            }
        }
    }
    return neigh;
}

std::vector<int> neighbors_cubic_nnn(int L) {
    int N = L * L * L;
    std::vector<int> neigh(N * 8);

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < L; k++) {
                int id = i * L * L + j * L + k;
                neigh[id * 8 + 0] = ((i - 1 + L) % L) * L * L + ((j - 1 + L) % L) * L + ((k - 1 + L) % L);  // -i -j -k
                neigh[id * 8 + 1] = ((i - 1 + L) % L) * L * L + ((j - 1 + L) % L) * L + ((k + 1) % L);      // -i -j +k
                neigh[id * 8 + 2] = ((i - 1 + L) % L) * L * L + ((j + 1) % L) * L + ((k - 1 + L) % L);      // -i +j -k
                neigh[id * 8 + 3] = ((i - 1 + L) % L) * L * L + ((j + 1) % L) * L + ((k + 1) % L);          // -i +j +k
                neigh[id * 8 + 4] = ((i + 1) % L) * L * L + ((j - 1 + L) % L) * L + ((k - 1 + L) % L);      // +i -j -k
                neigh[id * 8 + 5] = ((i + 1) % L) * L * L + ((j - 1 + L) % L) * L + ((k + 1) % L);          // +i -j +k
                neigh[id * 8 + 6] = ((i + 1) % L) * L * L + ((j + 1) % L) * L + ((k - 1 + L) % L);          // +i +j -k
                neigh[id * 8 + 7] = ((i + 1) % L) * L * L + ((j + 1) % L) * L + ((k + 1) % L);              // +i +j +k
            }
        }
    }
    return neigh;
}

LongRange_DistanceResult LongRange_DistancePrecompute(int L, double sigma)
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
    std::vector<double> r(N - 1);
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
    double sum = 0.0f;
    for (int j = 0; j < N - 1; ++j) {
        sum += 1.0f / std::pow(r[j], (2.0f + sigma));
    }

    double c_norm = 4.0f / sum; // Normalization constant

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
    std::vector<double> sorted_r(N - 1);

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
    const std::vector<double> beta_schedule,
    int L 
)
    /*
        Precompute the cumulative distribution function (CDF) for the Luijten-Blote cluster update.
    */
{
    int N = L*L; 
    int num_temps = beta_schedule.size();

    LB_ClusterPrecomputeResult clusterResult;

    std::vector<std::vector<double>> bondProb(N - 1);
    std::vector<std::vector<double>> noBondProb(N - 1);

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
            double J_k = distanceResult.J_r[k];
            bondProb[k][id] = 1.0f - std::exp(-2.0f * beta_schedule[id] * J_k);
        }

        // Compute noBondProb (q)
        noBondProb[0][id] = bondProb[0][id];
        double product = 1.0f; 
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


