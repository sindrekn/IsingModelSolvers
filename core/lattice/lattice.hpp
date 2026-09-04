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

std::vector<int> neighbors_2d_nn(int L); 

std::vector<int> neighbors_2d_nnn(int L);

std::vector<int> neighbors_honeycomb(int L); 

std::vector<int> neighbors_3d_nn(int L);

std::vector<int> neighbors_3d_nnn(int L);

#endif // CORE_LATTICE_HPP
