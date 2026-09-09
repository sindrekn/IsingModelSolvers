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
 
// Precompute Metropolis acceptance probabilities.
std::vector<double> BetaJS_2d_nn(
    const std::vector<double>& temp_schedule, double J)

{
    constexpr int S_vals[] = {8, 4, 0, -4, -8};

    int temp_updates = static_cast<int>(temp_schedule.size());
    std::vector<double> BetaJS(temp_updates * 5, 0.0);

    for (int i = 0; i < temp_updates; i++) {
        double beta = 1.0 / temp_schedule[i];
        for (int j = 0; j < 5; j++) {
            BetaJS[i * 5 + j] = (J * S_vals[j] <= 0)
                ? 1.0
                : std::exp(-beta * J * S_vals[j]);
        }
    }
    return BetaJS;
}

// Precompute Metropolis acceptance probabilities.
std::vector<double> BetaJS_triangular(
    const std::vector<double>& temp_schedule, double J)
{
    return BetaJS_3d_nn(temp_schedule, J);
}

// Precompute Metropolis acceptance probabilities.
std::vector<double> BetaJS_3d_nn(
    const std::vector<double>& temp_schedule, double J)
{
    constexpr int S_vals[] = {12, 8, 4, 0, -4, -8, -12};

    int temp_updates = static_cast<int>(temp_schedule.size());
    std::vector<double> BetaJS(temp_updates * 7, 0.0);
    for (int i = 0; i < temp_updates; i++) {
        double beta = 1.0 / temp_schedule[i];
        for (int j = 0; j < 7; j++) {
            BetaJS[i * 7 + j] = (J * S_vals[j] <= 0)
                ? 1.0
                : std::exp(-beta * J * S_vals[j]);
        }
    }
    return BetaJS;
}

std::vector<double> Padd_2d_nn(
    const std::vector<double>& temp_schedule, double J)
{
    int temp_updates = static_cast<int>(temp_schedule.size());
    std::vector<double> P_add(temp_updates);

    for (int i = 0; i < temp_updates; i++) {
        double beta = 1.0 / temp_schedule[i];
        P_add[i] = 1.0 - std::exp(-2.0 * beta * J);
    }
    return P_add;
}

