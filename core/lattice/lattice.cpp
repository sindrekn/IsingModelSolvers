#include "lattice.hpp"

std::vector<int> neighbors_2d_nn(int L) {
    int N = L * L;
    std::vector<int> neigh(N * 4);

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int id = i * L + j;
            neigh[id * 4 + 0] = ((i - 1 + L) % L) * L + j;      // up
            neigh[id * 4 + 1] = ((i + 1) % L) * L + j;          // down
            neigh[id * 4 + 2] = i * L + ((j - 1 + L) % L);      // left
            neigh[id * 4 + 3] = i * L + ((j + 1) % L);          // right
        }
    }
    return neigh;
}

std::vector<int> neighbors_2d_nnn(int L) {
    int N = L * L;
    std::vector<int> neigh(N * 4);

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int id = i * L + j;
            neigh[id * 4 + 0] = ((i - 1 + L) % L) * L + ((j - 1 + L) % L);  // up-left
            neigh[id * 4 + 1] = ((i - 1 + L) % L) * L + ((j + 1) % L);      // up-right
            neigh[id * 4 + 2] = ((i + 1) % L) * L + ((j - 1 + L) % L);      // down-left
            neigh[id * 4 + 3] = ((i + 1) % L) * L + ((j + 1) % L);          // down-right
        }
    }
    return neigh;
}

std::vector<int> neighbors_honeycomb(int L) {
    int N = L * L;
    std::vector<int> neigh(N * 3);
    return neigh;
}

std::vector<int> neighbors_3d_nn(int L) {
    int N = L * L * L;
    std::vector<int> neigh(N * 6);

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < L; k++) {
                int id = i * L * L + j * L + k;
                neigh[id * 6 + 0] = ((i - 1 + L) % L) * L * L + j * L + k; // -i
                neigh[id * 6 + 1] = ((i + 1) % L) * L * L + j * L + k;     // +i
                neigh[id * 6 + 2] = i * L * L + ((j - 1 + L) % L) * L + k; // -j
                neigh[id * 6 + 3] = i * L * L + ((j + 1) % L) * L + k;     // +j
                neigh[id * 6 + 4] = i * L * L + j * L + ((k - 1 + L) % L); // -k
                neigh[id * 6 + 5] = i * L * L + j * L + ((k + 1) % L);     // +k
            }
        }
    }
    return neigh;
}

std::vector<int> neighbors_3d_nnn(int L) {
    int N = L * L * L;
    std::vector<int> neigh(N * 8);

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < L; k++) {
                int id = i * L * L + j * L + k;
                neigh[id * 8 + 0] = ((i - 1 + L) % L) * L * L + ((j - 1 + L) % L) * L + ((k - 1 + L) % L);  // -i -j -k
                neigh[id * 8 + 1] = ((i - 1 + L) % L) * L * L + ((j - 1 + L) % L) * L + ((k + 1) % L);      // -i -j +k
                neigh[id * 8 + 2] = ((i - 1 + L) % L) * L * L + ((j + 1) % L) * L + ((k - 1 + L) % L);      // -i +j -k
                neigh[id * 8 + 3] = ((i - 1 + L) % L) * L * L + ((j + 1) % L) * L + ((k + 1) % L);          // -i +j +k
                neigh[id * 8 + 4] = ((i + 1) % L) * L * L + ((j - 1 + L) % L) * L + ((k - 1 + L) % L);      // +i -j -k
                neigh[id * 8 + 5] = ((i + 1) % L) * L * L + ((j - 1 + L) % L) * L + ((k + 1) % L);          // +i -j +k
                neigh[id * 8 + 6] = ((i + 1) % L) * L * L + ((j + 1) % L) * L + ((k - 1 + L) % L);          // +i +j -k
                neigh[id * 8 + 7] = ((i + 1) % L) * L * L + ((j + 1) % L) * L + ((k + 1) % L);              // +i +j +k
            }
        }
    }
    return neigh;
}
