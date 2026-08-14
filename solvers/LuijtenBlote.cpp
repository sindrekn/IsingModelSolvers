#include "LuijtenBlote.hpp"

#include "core/bit_ops.hpp"

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
)
    /*
        One sweeps of the long-range Ising model using a Luijten-Blote cluster update.
    */
{
    int N = L * L;
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

            float c_max = (clusterResult.cumulativeProb[N-2][beta_id] - 
                                clusterResult.cumulativeProb[m][beta_id]) 
                                * clusterResult.scale[m + 1][beta_id];

            if (u > c_max) {
                break; // No more candidates
            }

            // Search in the cumulative distribution for the next candidate
            int n = m + 1;
            
            while(true) { 
                if ((clusterResult.cumulativeProb[n][beta_id] - 
                    clusterResult.cumulativeProb[m][beta_id]) 
                    *clusterResult.scale[m + 1][beta_id]  < u) 
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






