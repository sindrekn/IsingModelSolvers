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

#endif // CORE_SCHEDULE_SCHEDULE_HPP