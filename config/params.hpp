#ifndef CONFIG_PARAMS_HPP
#define CONFIG_PARAMS_HPP

#include <cmath>

struct params {
    // --- Lattice ---
    int    L;               // lattice side length (LxL spins)
    double J;        // coupling constant (positive for ferromagnetic)

    // --- Interactions ---
    double J1;       // nearest-neighbor interaction strength
    double J2;       // next-nearest-neighbor interaction strength

    // --- Temp Schedule ---
    double sharpness;       // power-law warp around T_c (higher = more points near T_c)

    // --- Run control ---
    int    temp_updates;    // number of temperature steps
    int    sweeps_per_temp; // Sweeps per temperature step for 2D lattices
    int    sweeps_per_temp_3d; // Sweeps per temperature step for 3D lattices
    int    store_step;          // store state every N sweeps/updates

    // Default constructor reproduces the values from your original main().
    // Override individual fields after construction as needed.
    static params defaults(int L) {
        return params {
            .L                      = L,
            .J                      = 1.0,
            .J1                     = 1.0,
            .J2                     = 1.0,
            .sharpness              = 1.6,
            .temp_updates           = 60,
            .sweeps_per_temp        = L*L,
            .sweeps_per_temp_3d     = L*L*L,
            .store_step              = 500,
        };
    };
};

#endif // CONFIG_PARAMS_HPP