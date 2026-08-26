#include "lattice.hpp"

std::vector<int> precompute_2d_nn(int L) {
    int N = L * L;
    std::vector<int> neigh(N * 4);

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int id = i * L + j;
            neigh[id * 4 + 0] = ((i - 1 + L) % L) * L + j;  // up
            neigh[id * 4 + 1] = ((i + 1) % L) * L + j;       // down
            neigh[id * 4 + 2] = i * L + ((j - 1 + L) % L);   // left
            neigh[id * 4 + 3] = i * L + ((j + 1) % L);       // right
        }
    }
    return neigh;
}

std::vector<float> precompute_2d_nnn(int L) {
    int N = L * L;
    std::vector<float> neigh(N * 6);
    return neigh;
}

std::vector<int> precompute_honeycomb(int L) {
    int N = L * L;
    std::vector<int> neigh(N * 6);
    return neigh;
}

std::vector<int> precompute_3d_nn(int L) {
    int N = L * L * L;
    std::vector<int> neigh(N * 6);
    return neigh;
}
