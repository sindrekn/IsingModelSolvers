#include <fstream>
#include <vector>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// ---------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double ReLu(double x) {
    return std::max(0.0, x);
}

// ---------------------------------------------------------------------------
// SnapshotReader
//
// Encapsulates the file format:
//   [uint64_t num_words][num_words * uint64_t bit-packed spin words]...
// repeated until EOF. This is the single place that knows how to walk the
// file, validate headers, and stay byte-aligned when a snapshot's word count
// doesn't match what's expected (it still consumes the payload so the next
// header is read correctly).
// ---------------------------------------------------------------------------
class SnapshotReader {
public:
    SnapshotReader(const std::string& filepath, int num_spins)
        : file_(filepath, std::ios::binary),
          num_spins_(num_spins),
          expected_words_((num_spins + 63) / 64) {
        if (!file_.is_open()) {
            throw std::runtime_error("Failed to open file: " + filepath);
        }
    }

    int num_spins() const { return num_spins_; }
    int snapshot_index() const { return snapshot_index_; }

    // Reads the next snapshot's raw bit-packed words.
    // Returns std::nullopt at clean EOF.
    // Snapshots whose word count doesn't match expected_words_ are skipped
    // (payload still consumed to stay aligned) rather than returned.
    std::optional<std::vector<uint64_t>> next() {
        while (true) {
            uint64_t num_words = 0;
            file_.read(reinterpret_cast<char*>(&num_words), sizeof(num_words));

            if (file_.eof()) {
                return std::nullopt; // clean end of file
            }
            if (!file_ || file_.gcount() < static_cast<std::streamsize>(sizeof(num_words))) {
                throw std::runtime_error("Truncated header at snapshot " + std::to_string(snapshot_index_));
            }

            if (num_words != expected_words_) {
                std::cerr << "  Warning: expected " << expected_words_ << " words for a "
                          << num_spins_ << " spin lattice, got "
                          << num_words << ". Skipping unpack.\n";
                file_.seekg(static_cast<std::streamoff>(num_words * sizeof(uint64_t)), std::ios::cur);
                ++snapshot_index_;
                continue; // try the next snapshot instead of returning garbage
            }

            std::vector<uint64_t> state(num_words);
            file_.read(reinterpret_cast<char*>(state.data()),
                       static_cast<std::streamsize>(num_words * sizeof(uint64_t)));
            if (!file_) {
                throw std::runtime_error("Truncated snapshot data at index " + std::to_string(snapshot_index_));
            }

            ++snapshot_index_;
            return state;
        }
    }

private:
    std::ifstream file_;
    int num_spins_;
    uint64_t expected_words_;
    int snapshot_index_ = 0;
};

// ---------------------------------------------------------------------------
// Bit-unpacking helpers
// ---------------------------------------------------------------------------

// Unpacks bit-packed words into +1/-1 spins, row-major, LSB-first per word.
std::vector<int> unpack_spins(const std::vector<uint64_t>& state, int num_spins) {
    std::vector<int> spins(num_spins);
    for (int spin_idx = 0; spin_idx < num_spins; ++spin_idx) {
        int word_idx = spin_idx / 64;
        int bit_idx  = spin_idx % 64;
        int bit = static_cast<int>((state[word_idx] >> bit_idx) & 1ULL);
        spins[spin_idx] = bit ? 1 : -1;
    }
    return spins;
}

double average_magnetization(const std::vector<int>& spins) {
    double sum = 0.0;
    for (int s : spins) sum += s;
    return sum / static_cast<double>(spins.size());
}

// ---------------------------------------------------------------------------
// Public API (unchanged signatures/behavior, now built on SnapshotReader)
// ---------------------------------------------------------------------------

void print_lattice_snapshots(const std::string& filepath, int num_spins) {
    SnapshotReader reader(filepath, num_spins);

    while (auto state = reader.next()) {
        std::cout << "=== Snapshot " << (reader.snapshot_index() - 1) << " ===\n";
        std::cout << "Header: num_words = " << state->size() << "\n";

        auto spins = unpack_spins(*state, num_spins);
        for (int s : spins) {
            std::cout << (s == 1 ? " +1" : " -1");
        }
        std::cout << "\n";

        std::cout << "Average Magnetization: " << average_magnetization(spins) << "\n";
    }

    std::cout << "Total snapshots read: " << reader.snapshot_index() << "\n";
}

std::vector<double> return_avg_mag(const std::string& filepath, int num_spins) {
    SnapshotReader reader(filepath, num_spins);
    std::vector<double> avg_mags;

    while (auto state = reader.next()) {
        auto spins = unpack_spins(*state, num_spins);
        avg_mags.push_back(std::abs(average_magnetization(spins)));
    }
    return avg_mags;
}

std::vector<double> return_simple_ann(const std::string& filepath, int num_spins, double ymin, double ymax) {
    double v0 = -std::log(1 / ymin - 1);
    double a = -v0 - std::log(1 / ymax - 1);
    std::vector<double> avg_mags = return_avg_mag(filepath, num_spins);

    for (auto& m : avg_mags) {
        m = sigmoid(a * m + v0);
    }
    return avg_mags;
}

std::vector<double> return_region_mag(const std::string& filepath, int num_spins, int A, double ymin, double ymax) {
    SnapshotReader reader(filepath, num_spins);
    std::vector<double> y;

    while (auto state = reader.next()) {
        auto spins = unpack_spins(*state, num_spins);
        // TODO: region-restricted magnetization using region size A and
        // ymin/ymax rescaling (unimplemented in the original as well).
        (void)A; (void)ymin; (void)ymax; (void)spins;
    }
    return y;
}

PYBIND11_MODULE(file_reader, m) {
    m.doc() = "pybind11 file reader plugin for reading lattice snapshots";
    m.def("return_avg_mag", &return_avg_mag, "A function that returns the average magnetization for each snapshot");
    m.def("return_simple_ann", &return_simple_ann, "A function that returns the simple average magnetization based ann for each snapshot");
    m.def("print_lattice_snapshots", &print_lattice_snapshots, "A function that prints the lattice snapshots and their average magnetization");
}