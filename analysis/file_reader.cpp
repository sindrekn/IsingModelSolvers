#include <fstream>
#include <vector>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 

void print_lattice_snapshots(const std::string& filepath, int lattice_dim) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    const int num_spins = lattice_dim * lattice_dim;
    const uint64_t expected_words = (num_spins + 63) / 64; // ceil division, bit-packed

    int snapshot_index = 0;

    while (true) {
        uint64_t num_words = 0;
        file.read(reinterpret_cast<char*>(&num_words), sizeof(num_words));

        if (file.eof()) {
            break; // clean end of file
        }
        if (!file || file.gcount() < static_cast<std::streamsize>(sizeof(num_words))) {
            throw std::runtime_error("Truncated header at snapshot " + std::to_string(snapshot_index));
        }

        std::cout << "=== Snapshot " << snapshot_index << " ===\n";
        std::cout << "Header: num_words = " << num_words << "\n";

        if (num_words != expected_words) {
            std::cerr << "  Warning: expected " << expected_words << " words for a "
                      << lattice_dim << "x" << lattice_dim << " lattice, got "
                      << num_words << ". Skipping unpack.\n";
            // still need to consume the data bytes to stay aligned for the next header
            file.seekg(static_cast<std::streamoff>(num_words * sizeof(uint64_t)), std::ios::cur);
            ++snapshot_index;
            continue;
        }

        std::vector<uint64_t> state(num_words);
        file.read(reinterpret_cast<char*>(state.data()),
                   static_cast<std::streamsize>(num_words * sizeof(uint64_t)));
        if (!file) {
            throw std::runtime_error("Truncated snapshot data at index " + std::to_string(snapshot_index));
        }

        // Unpack bits into the lattice, row-major, LSB-first within each word
        double avg_mag = 0.0;
        for (int row = 0; row < lattice_dim; ++row) {
            for (int col = 0; col < lattice_dim; ++col) {
                int spin_idx = row * lattice_dim + col;
                int word_idx = spin_idx / 64;
                int bit_idx  = spin_idx % 64;
                int bit = static_cast<int>((state[word_idx] >> bit_idx) & 1ULL);
                std::cout << (bit ? " +1" : " -1");
                avg_mag += (bit ? 1 : -1);
            }
            std::cout << "\n";
        }
        avg_mag /= (lattice_dim * lattice_dim);
        std::cout << "Average Magnetization: " << avg_mag << "\n";

        ++snapshot_index;
    }

    std::cout << "Total snapshots read: " << snapshot_index << "\n";
}

std::vector<double> return_avg_mag(const std::string& filepath, int lattice_dim) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    const int num_spins = lattice_dim * lattice_dim;
    const uint64_t expected_words = (num_spins + 63) / 64; // ceil division, bit-packed

    int snapshot_index = 0;

    std::vector<double> avg_mags;

    while (true) {
        uint64_t num_words = 0;
        file.read(reinterpret_cast<char*>(&num_words), sizeof(num_words));

        if (file.eof()) {
            break; // clean end of file
        }
        if (!file || file.gcount() < static_cast<std::streamsize>(sizeof(num_words))) {
            throw std::runtime_error("Truncated header at snapshot " + std::to_string(snapshot_index));
        }

        if (num_words != expected_words) {
            std::cerr << "  Warning: expected " << expected_words << " words for a "
                      << lattice_dim << "x" << lattice_dim << " lattice, got "
                      << num_words << ". Skipping unpack.\n";
            // still need to consume the data bytes to stay aligned for the next header
            file.seekg(static_cast<std::streamoff>(num_words * sizeof(uint64_t)), std::ios::cur);
            ++snapshot_index;
            continue;
        }

        std::vector<uint64_t> state(num_words);
        file.read(reinterpret_cast<char*>(state.data()),
                   static_cast<std::streamsize>(num_words * sizeof(uint64_t)));
        if (!file) {
            throw std::runtime_error("Truncated snapshot data at index " + std::to_string(snapshot_index));
        }

        // Unpack bits into the lattice, row-major, LSB-first within each word
        double avg_mag = 0.0;
        for (int row = 0; row < lattice_dim; ++row) {
            for (int col = 0; col < lattice_dim; ++col) {
                int spin_idx = row * lattice_dim + col;
                int word_idx = spin_idx / 64;
                int bit_idx  = spin_idx % 64;
                int bit = static_cast<int>((state[word_idx] >> bit_idx) & 1ULL);
                avg_mag += (bit ? 1 : -1);
            }
        }
        avg_mag /= (lattice_dim * lattice_dim);
        ++snapshot_index;
        avg_mags.push_back(avg_mag);
    }
    return avg_mags;
}

PYBIND11_MODULE(file_reader, m) {
    m.doc() = "pybind11 file reader plugin for reading lattice snapshots";
    m.def("return_avg_mag", &return_avg_mag, "A function that returns the average magnetization for each snapshot");
    m.def("print_lattice_snapshots", &print_lattice_snapshots, "A function that prints the lattice snapshots and their average magnetization");
}

