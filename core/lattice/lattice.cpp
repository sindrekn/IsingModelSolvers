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

std::vector<int> neighbors_triangular(int Lx, int Ly) {
    if (Lx <= 0 || Ly <= 0) {
        throw std::invalid_argument("Lattice dimensions must be positive.");
    }

    if (Lx % 2 != 0) {
        throw std::invalid_argument("Lattice width (Lx) must be even for triangular lattice.");
    }

    if (Ly % 2 != 0) {
        throw std::invalid_argument("Lattice height (Ly) must be even for triangular lattice.");
    }

    int N = Lx * Ly;
    std::vector<int> neigh(N * 6);

    for (int i = 0; i < Ly; i++) {
        for (int j = 0; j < Lx; j++) {

            int id = i * Lx + j;

            int im = (i - 1 + Ly) % Ly;
            int ip = (i + 1) % Ly;
            int jm = (j - 1 + Lx) % Lx;
            int jp = (j + 1) % Lx;

            if (i % 2 == 0) {

                neigh[id * 6 + 0] = im * Lx + jm;
                neigh[id * 6 + 1] = im * Lx + j;
                neigh[id * 6 + 2] = i  * Lx + jm;
                neigh[id * 6 + 3] = i  * Lx + jp;
                neigh[id * 6 + 4] = ip * Lx + jm;
                neigh[id * 6 + 5] = ip * Lx + j;

            } else {

                neigh[id * 6 + 0] = im * Lx + j;
                neigh[id * 6 + 1] = im * Lx + jp;
                neigh[id * 6 + 2] = i  * Lx + jm;
                neigh[id * 6 + 3] = i  * Lx + jp;
                neigh[id * 6 + 4] = ip * Lx + j;
                neigh[id * 6 + 5] = ip * Lx + jp;
            }
        }
    }

    return neigh;
}

std::vector<int> neighbors_honeycomb(int Lx, int Ly) {
    if (Lx <= 0 || Ly <= 0) {
        throw std::invalid_argument("Lattice dimensions must be positive.");
    }

    if (Lx % 2 != 0) {
        throw std::invalid_argument("Lattice width (Lx) must be even for honeycomb lattice.");
    }

    if (Ly % 2 != 0) {
        throw std::invalid_argument("Lattice height (Ly) must be even for honeycomb lattice.");
    }

    int N = Lx * Ly;
    std::vector<int> neigh(N * 3);

    for (int i = 0; i < Ly; i++) {
        for (int j = 0; j < Lx; j++) {

            int id = i * Lx + j;

            int im = (i - 1 + Ly) % Ly;
            int ip = (i + 1) % Ly;
            int jm = (j - 1 + Lx) % Lx;
            int jp = (j + 1) % Lx;

            if ((i + j) % 2 == 0) {
                neigh[id * 3 + 0] = im * Lx + j; // up
                neigh[id * 3 + 1] = i * Lx + jp; // right
                neigh[id * 3 + 2] = ip * Lx + j; // down
            } else {
                neigh[id * 3 + 0] = im * Lx + j; // up
                neigh[id * 3 + 1] = i * Lx + jm; // left
                neigh[id * 3 + 2] = ip * Lx + j; // down
            }
        }
    }

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
