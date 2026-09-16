#ifndef SOLVERS_SOLVERS_HPP
#define SOLVERS_SOLVERS_HPP

#include <vector>
#include <random>
#include <cstdint>
#include <cstring>

void quadratic(
    const int*              neighbors,
    const double*           BetaJS_row,
    const std::vector<int>& even_sites,
    const std::vector<int>& odd_sites,
    int                     parity,
    uint64_t*               state,
    std::uniform_real_distribution<double>& udist,
    std::mt19937&           rng); 

void wolff_cluster(
    const int*              neighbors,
    double                  P_add,
    uint64_t*               state,
    std::vector<int>&       stack,
    std::uniform_real_distribution<double>& udist,
    std::uniform_int_distribution<int>& site_dist,
    std::mt19937&           rng); 

void quadratic_nnn(
    int                     N,
    int                     J1, 
    int                     J2,
    const int*              neighbors_nn,
    const int*              neighbors_nnn,
    double                  beta,
    uint64_t*               state,
    std::uniform_real_distribution<double>& udist,
    std::mt19937&           rng); 

void triangular(
    int                     N,
    const int*              neighbors,
    const double*           BetaJS_row,
    uint64_t*               state,
    std::uniform_real_distribution<double>& udist,
    std::mt19937&           rng); 

void honeycomb(
    int                     N,
    const int*              neighbors,
    const double*           BetaJS_row,
    uint64_t*               state,
    std::uniform_real_distribution<double>& udist,
    std::mt19937&           rng);

void cubic(
    const int*              neighbors,
    const double*           BetaJS_row,
    const std::vector<int>& even_sites,
    const std::vector<int>& odd_sites,
    int                     parity,
    uint64_t*               state,
    std::uniform_real_distribution<double>& udist,
    std::mt19937&           rng); 

void cubic_nnn(
    int                     N,
    int                     J1, 
    int                     J2,
    const int*              neighbors_nn,
    const int*              neighbors_nnn,
    double                  beta,
    uint64_t*               state,
    std::uniform_real_distribution<double>& udist,
    std::mt19937&           rng); 

#endif // SOLVERS_SOLVERS_HPP