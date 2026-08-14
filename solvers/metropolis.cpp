#include "metropolis.hpp"

#include "core/bit_ops.hpp"

std::vector<float> construct_h(
    const int N,
    uint64_t* state,
    const LongRange_DistanceResult& distanceResult
)
{
    int L = static_cast<int>(std::sqrt(N));
    std::vector<float> h_vector(N, 0.0);
    
    for (int i = 0; i < N; ++i)
    {
        int xi = i % L;
        int yi = i / L;

        float h = 0.0f;

        for (int k = 0; k < N-1; ++k)
        {
            int xj = (xi + distanceResult.dx[k] + L) % L;
            int yj = (yi + distanceResult.dy[k] + L) % L;

            int j = xj + L*yj;

            h += distanceResult.J_r[k] * (2.0f * get_bit(state, j) - 1.0f);
        }

        h_vector[i] = h;
    }
    return h_vector;
}

void update_h(
    const LongRange_DistanceResult& distanceResult,
    std::vector<float>& h, 
    int id, 
    int s_i, 
    int N, 
    int L
    )
{
    int xi = id % L;
    int yi = id / L;

    for (int k = 0; k < N-1; ++k)
    {
        int x = (xi + distanceResult.dx[k] + L) % L;
        int y = (yi + distanceResult.dy[k] + L) % L;

        int j = x + L*y;

        h[j] -= 2.0f * s_i * distanceResult.J_r[k];
    }
}

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
    )
    /*
        One sweeps of the long-range Ising model using a Metropolis-like update.
    */
{
    for (int id = 0; id < N; id++) {

        int s_i = 2 * get_bit(state, id) - 1; // Convert {0,1} to {-1,1} and flip the sign for the energy calculation
        
        float dE = 2 * s_i * h[id];
        if (dE <= 0 || udist(rng) < std::exp(-Beta * dE)) {
            flip_bit(state, id);
            update_h(distanceResult, h, id, s_i, N, L);
        }
    }
}


