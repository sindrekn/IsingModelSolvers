#ifndef CONFIG_PARAMS_HPP
#define CONFIG_PARAMS_HPP

#include <cmath>

struct params {
    // --- Lattice ---
    int    L;               // lattice side length (LxL spins)
    double J;        // coupling constant (positive for ferromagnetic)

    // --- Temp Schedule ---
    double t_c;             // critical temperature for 2D Ising (for reference)
    double t_min;           // lowest temperature in schedule
    double t_max;           // highest temperature in schedule
    double sharpness;       // power-law warp around T_c (higher = more points near T_c)

    // --- Run control ---
    int    temp_updates;    // number of temperature steps
    int    num_runs;              // independent parallel runs
    int    metro_sweeps_per_temp; // Metropolis sweeps per temperature step
    int    store_step;          // store state every N sweeps/updates

    // Default constructor reproduces the values from your original main().
    // Override individual fields after construction as needed.
    static params defaults(int L) {
        return params {
            .L                      = L,
            .J                      = 1.0,
            .t_c                    = 3.0,
            .t_min                  = 1.0,
            .t_max                  = 5.0,
            .sharpness              = 1.0,
            .temp_updates           = 20,
            .num_runs               = 50,
            .metro_sweeps_per_temp  = L*L,
            .store_step              = 20,
        };
    };
};

#endif // CONFIG_PARAMS_HPP