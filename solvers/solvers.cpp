#include "solvers.hpp"

#include "core/bit_ops.hpp"


void quadratic(
    const int*              neighbors,
    const double*           BetaJS_row,
    const std::vector<int>& even_sites,
    const std::vector<int>& odd_sites,
    int                     parity,
    uint64_t*               state,
    std::uniform_real_distribution<double>& udist,
    std::mt19937&           rng)

{
    const std::vector<int>& sites = (parity == 0) ? even_sites : odd_sites;
    
    for (int id : sites) {

        int c = get_bit(state, id);

        int S = (c ^ get_bit(state, neighbors[id * 4 + 0]))
                + (c ^ get_bit(state, neighbors[id * 4 + 1]))
                + (c ^ get_bit(state, neighbors[id * 4 + 2]))
                + (c ^ get_bit(state, neighbors[id * 4 + 3]));

        double dE = BetaJS_row[S];
        if (dE >= 1.0 || udist(rng) < dE)
            flip_bit(state, id);
    }
}


void wolff_cluster(
    const int*      neighbors,
    double          P_add,
    uint64_t*       state,
    std::vector<int>& stack,
    std::uniform_real_distribution<double>& udist,
    std::uniform_int_distribution<int>& site_dist,
    std::mt19937&   rng)
{
    int seed      = site_dist(rng);
    int seed_spin = get_bit(state, seed);

    stack.clear();
    stack.push_back(seed);
    flip_bit(state, seed);

    while (!stack.empty()) {
        int cur = stack.back();
        stack.pop_back();

        for (int d = 0; d < 4; d++) {
            int nb = neighbors[cur * 4 + d];
            if (get_bit(state, nb) == seed_spin && udist(rng) < P_add) {
                stack.push_back(nb);
                flip_bit(state, nb);
            }
        }
    }
}


void quadratic_nnn(
    int                     N,
    int                     J1, 
    int                     J2,
    const int*              neighbors_nn,
    const int*              neighbors_nnn,
    double                  beta,
    uint64_t*               state,
    std::uniform_real_distribution<double>& udist,
    std::mt19937&           rng)

{    
    for (int id = 0; id < N; id++) {

        double dE = 0.0; 

        int s_i = 2 * get_bit(state, id) - 1; 

        for (int j= 0; j < 4; j++) {
            int nn_id = neighbors_nn[id * 4 + j];
            int s_j = 2 * get_bit(state, nn_id) - 1; 
            dE += J1 * s_i * s_j;
            int nnn_id = neighbors_nnn[id * 4 + j];
            int s_k = 2 * get_bit(state, nnn_id) - 1; 
            dE += J2 * s_i * s_k; 
        }

        if (dE <= 0 || udist(rng) < std::exp(-beta * 2 * dE)) {
            flip_bit(state, id);
        }
    }
}

void triangular(
    int                     N,
    const int*              neighbors,
    const double*           BetaJS_row,
    uint64_t*               state,
    std::uniform_real_distribution<double>& udist,
    std::mt19937&           rng)

{   
    for (int id = 0; id < N; id++) {

        int c = get_bit(state, id);

        int S = (c ^ get_bit(state, neighbors[id * 6 + 0]))
                + (c ^ get_bit(state, neighbors[id * 6 + 1]))
                + (c ^ get_bit(state, neighbors[id * 6 + 2]))
                + (c ^ get_bit(state, neighbors[id * 6 + 3]))
                + (c ^ get_bit(state, neighbors[id * 6 + 4]))
                + (c ^ get_bit(state, neighbors[id * 6 + 5]));

        double dE = BetaJS_row[S];
        if (dE >= 1.0 || udist(rng) < dE)
            flip_bit(state, id);
    }
}

void honeycomb(
    int                     N,
    const int*              neighbors,
    const double*           BetaJS_row,
    uint64_t*               state,
    std::uniform_real_distribution<double>& udist,
    std::mt19937&           rng)

{   
    for (int id = 0; id < N; id++) {

        int c = get_bit(state, id);

        int S = (c ^ get_bit(state, neighbors[id * 3 + 0]))
                + (c ^ get_bit(state, neighbors[id * 3 + 1]))
                + (c ^ get_bit(state, neighbors[id * 3 + 2]));

        double dE = BetaJS_row[S];
        if (dE >= 1.0 || udist(rng) < dE)
            flip_bit(state, id);
    }
}

void cubic(
    const int*              neighbors,
    const double*           BetaJS_row,
    const std::vector<int>& even_sites,
    const std::vector<int>& odd_sites,
    int                     parity,
    uint64_t*               state,
    std::uniform_real_distribution<double>& udist,
    std::mt19937&           rng)

{
    const std::vector<int>& sites = (parity == 0) ? even_sites : odd_sites;
    
    for (int id : sites) {

        int c = get_bit(state, id);

        int S = (c ^ get_bit(state, neighbors[id * 6 + 0]))
                + (c ^ get_bit(state, neighbors[id * 6 + 1]))
                + (c ^ get_bit(state, neighbors[id * 6 + 2]))
                + (c ^ get_bit(state, neighbors[id * 6 + 3]))
                + (c ^ get_bit(state, neighbors[id * 6 + 4]))
                + (c ^ get_bit(state, neighbors[id * 6 + 5]));

        double dE = BetaJS_row[S];
        if (dE >= 1.0 || udist(rng) < dE)
            flip_bit(state, id);
    }
}

void cubic_nnn(
    int                     N,
    int                     J1, 
    int                     J2,
    const int*              neighbors_nn,
    const int*              neighbors_nnn,
    double                  beta,
    uint64_t*               state,
    std::uniform_real_distribution<double>& udist,
    std::mt19937&           rng)

{    
    for (int id = 0; id < N; id++) {

        double dE = 0.0; 

        int s_i = 2 * get_bit(state, id) - 1; 

        for (int j= 0; j < 6; j++) {
            int nn_id = neighbors_nn[id * 6 + j];
            int s_j = 2 * get_bit(state, nn_id) - 1; 
            dE += J1 * s_i * s_j;
        }
        for (int j= 0; j < 8; j++) {
            int nnn_id = neighbors_nnn[id * 8 + j];
            int s_k = 2 * get_bit(state, nnn_id) - 1; 
            dE += J2 * s_i * s_k; 
        }

        if (dE <= 0 || udist(rng) < std::exp(-beta * 2 * dE)) {
            flip_bit(state, id);
        }
    }
}

std::vector<double> construct_h(
    const int N,
    uint64_t* state,
    const LongRange_DistanceResult& distanceResult
)
{
    int L = static_cast<int>(std::sqrt(N));
    std::vector<double> h_vector(N, 0.0);
    
    for (int i = 0; i < N; ++i)
    {
        int xi = i % L;
        int yi = i / L;

        double h = 0.0;

        for (int k = 0; k < N-1; ++k)
        {
            int xj = (xi + distanceResult.dx[k] + L) % L;
            int yj = (yi + distanceResult.dy[k] + L) % L;

            int j = xj + L*yj;

            h += distanceResult.J_r[k] * (2.0 * get_bit(state, j) - 1.0);
        }

        h_vector[i] = h;
    }
    return h_vector;
}

void update_h(
    const LongRange_DistanceResult& distanceResult,
    std::vector<double>& h, 
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

void FieldUpdate(
    const LongRange_DistanceResult& distanceResult,
    std::vector<double>&     h,
    int                     N, 
    int                     L,
    uint64_t*               state,
    float                   Beta,
    std::uniform_real_distribution<double>& udist,
    std::mt19937&           rng
    )
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

void LuijtenBloteCluster(
    const LB_ClusterPrecomputeResult& clusterResult,
    const LongRange_DistanceResult&      distanceResult,
    int                     N, 
    int                     L, 
    uint64_t*               state,
    int                     temp_index,
    std::uniform_real_distribution<double>& udist,
    std::mt19937&           rng
)
    /*
        One sweeps of the long-range Ising model using a Luijten-Blote cluster update.
    */
{
    int update = 0; 

    std::vector<int> cluster;
    std::vector<int> stack; 
    std::vector<bool> inCluster(N, false);

    //---------------------------------------
    // Choose random seed
    //---------------------------------------

    std::uniform_int_distribution<int> siteDist(0,N-1);
    int seed = siteDist(rng);

    cluster.clear();
    stack.clear();

    std::fill(inCluster.begin(), inCluster.end(), false);

    cluster.push_back(seed);
    stack.push_back(seed);
    inCluster[seed] = true;

    //---------------------------------------
    // Grow cluster
    //---------------------------------------

    while(!stack.empty()) {
        int i = stack.back();
        int xi = i % L;
        int yi = i / L;
        stack.pop_back();

        //-----------------------------------
        // Find candidate neighbours
        //-----------------------------------
        int m = 0;

        while(m < N - 2)
        {
            //--------------------------------
            // LB jump
            //--------------------------------

            double u = udist(rng);

            float c_max = (clusterResult.cumulativeProb[N-2][temp_index] - 
                                clusterResult.cumulativeProb[m][temp_index]) 
                                * clusterResult.scale[m + 1][temp_index];

            if (u > c_max) {
                break; // No more candidates
            }

            // Search in the cumulative distribution for the next candidate
            int n = m + 1;
            
            while(true) { 
                if ((clusterResult.cumulativeProb[n][temp_index] - 
                    clusterResult.cumulativeProb[m][temp_index]) 
                    *clusterResult.scale[m + 1][temp_index]  < u) 
                    n += 1; 
                else 
                    break; 
                }

            //--------------------------------
            // Convert distance into lattice site
            //--------------------------------

            int dx = distanceResult.dx[n];
            int dy = distanceResult.dy[n];

            int xj = (xi + dx + L) % L;
            int yj = (yi + dy + L) % L;

            int j = (int)(xj + yj * L);

            //--------------------------------
            // Same spin?
            //--------------------------------

            if(are_equal(state,i,j) && !inCluster[j])
            {
                inCluster[j] = true;

                cluster.push_back(j);
                stack.push_back(j);
            }
            m = n;
        }
    }
    //---------------------------------------
    // Flip cluster
    //---------------------------------------

    for(int i : cluster)
        flip_bit(state, i);

    update += cluster.size(); 

}

