#include "nn_3d.hpp"

#include "core/bit_ops.hpp"

// ----------------------------------------------------------
// The 3D Nearest Neighbor Metropolis spin update algorithm
// ----------------------------------------------------------
void nn_3d(
    const int*              neighbors,
    const double*           BetaJS_row,
    const std::vector<int>& even_sites,
    const std::vector<int>& odd_sites,
    int                     parity,
    uint64_t*               state,
    std::uniform_real_distribution<double>& udist,
    std::mt19937&           rng)
    /*
        One checkerboard half-sweep over sites of the given parity.
        state is modified in-place — caller owns the buffer and handles
        snapshotting into state_storage.
    */

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


