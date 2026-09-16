#ifndef CORE_LATTICE_HPP
#define CORE_LATTICE_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <iostream>

// ----------------------------------------------------------
// Precompute the distance and interaction strength for a 2D 
// lattice with periodic boundary conditions. The interaction
// strength decays as 1/r^(2+sigma) where r is the distance 
// between two sites.
// ----------------------------------------------------------

std::vector<int> neighbors_quadratic(int L); 

std::vector<int> neighbors_quadratic_nnn(int L);

std::vector<int> neighbors_triangular(int Lx, int Ly); 

std::vector<int> neighbors_honeycomb(int Lx, int Ly); 

std::vector<int> neighbors_cubic(int L);

std::vector<int> neighbors_cubic_nnn(int L);

#endif // CORE_LATTICE_HPP
