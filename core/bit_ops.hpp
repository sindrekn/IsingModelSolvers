#ifndef CORE_BIT_OPS_HPP
#define CORE_BIT_OPS_HPP

#include <cstdint>
#include <random>

// ----------------------------------------------------------
// Bit manipulation utilities for representing binary states 
// as arrays of 64-bit integers.
// ----------------------------------------------------------

static constexpr int BITS = 64;

inline int num_words(int N) { return (N + BITS - 1) / BITS; }

inline int get_bit(const uint64_t* v, int i) {
    return (v[i / BITS] >> (i % BITS)) & 1;
}

inline void flip_bit(uint64_t* v, int i) {
    v[i / BITS] ^= (1ULL << (i % BITS));
}

inline bool are_equal(const uint64_t* v, int i, int j) {
    // Shift both bits to the least significant bit (LSB) position and XOR them.
    // If the LSB of the result is 0, they are the same.
    return !(((v[i / BITS] >> (i % BITS)) ^ (v[j / BITS] >> (j % BITS))) & 1);
}

// ----------------------------------------------------------
// Random configuration generator for the initial state of each run.
// ----------------------------------------------------------
inline void random_binary_state(uint64_t* state, int N, std::mt19937& rng)
{
    int W = num_words(N);
    std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
    for (int w = 0; w < W; w++)
        state[w] = dist(rng);
}

#endif // CORE_BIT_OPS_HPP