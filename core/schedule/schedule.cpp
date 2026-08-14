#include "schedule.hpp"

std::vector<double> peak_temperature_schedule(
    double t_min, double t_max, double t_critical,
    int num_updates, double sharpness)

{
    if (num_updates <= 1) {
            return { t_critical }; 
        }

    std::vector<double> schedule(num_updates);

    for (int i = 0; i < num_updates; i++) {
        double x = -1.0 + (2.0 * i) / static_cast<double>(num_updates - 1);
        double sign = (x < 0.0) ? -1.0 : 1.0;
        double wx   = sign * std::pow(std::abs(x), sharpness);  // warped x

        schedule[i] = (wx < 0.0)
            ? t_critical + wx * (t_critical - t_min)
            : t_critical + wx * (t_max - t_critical);
    }

    std::sort(schedule.begin(), schedule.end(), std::greater<double>());
    return schedule;
}

