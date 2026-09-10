#ifndef CORE_SCHEDULE_SCHEDULE_HPP
#define CORE_SCHEDULE_SCHEDULE_HPP

#include <vector>
#include <cmath>
#include <algorithm>

// ----------------------------------------------------------
// Precomputed schedules for simualtions. 
// ----------------------------------------------------------

// Precomputed temperature schedule concentrated around T_critical.
std::vector<double> peak_temperature_schedule(
    double t_min, double t_max, double t_critical,
    int num_updates, double sharpness);

// Precompute Metropolis acceptance probabilities.
std::vector<double> BetaJS_2d_nn(const std::vector<double>& temp_schedule, double J);

// Precompute Metropolis acceptance probabilities.
std::vector<double> BetaJS_triangular(const std::vector<double>& temp_schedule, double J);

// Precompute Metropolis acceptance probabilities.
std::vector<double> BetaJS_honeycomb(const std::vector<double>& temp_schedule, double J);

// Precompute Metropolis acceptance probabilities.
std::vector<double> BetaJS_3d_nn(const std::vector<double>& temp_schedule, double J);

// Precompute Wolff's cluster addition probabilities.
std::vector<double> Padd_2d_nn(const std::vector<double>& temp_schedule, double J);

#endif // CORE_SCHEDULE_SCHEDULE_HPP