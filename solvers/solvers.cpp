#include "solvers.hpp"

#include "core/bit_ops.hpp"


void nn_2d(
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


void nnn_2d(
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

void nn_3d(
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

void nnn_3d(
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

