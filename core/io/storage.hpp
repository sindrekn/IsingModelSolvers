#ifndef CORE_IO_STORAGE_HPP
#define CORE_IO_STORAGE_HPP

#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <bits/stdc++.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------
// Utility functions for directory and file management
// Use setup_temperature_directories() to create a set of directories 
// for each temperature to store the snapshots
// --------------------------------------------------------------------:

std::vector<std::string> setup_temperature_directories(
    const std::vector<double>& temps, 
    const std::string& base_dir
);

// ---------------------------------------------------------------------
// StateWriter
//
// Buffers spin-state snapshots (bit-packed as std::vector<uint64_t>) and
// writes them to <path>/Snapshots_<run_id>.dat. The buffer auto-flushes
// once it holds buffer_capacity_snapshots_ snapshots; flush() appends to
// the same open file and resets the buffer, it never truncates or reopens.
//
// On-disk layout per snapshot: [uint64_t num_words][num_words * uint64_t]
// ---------------------------------------------------------------------
class StateWriter {
public:
    StateWriter(std::string path, uint16_t run_id,
                std::size_t buffer_capacity_bytes = 100 * 1024 * 1024); // 100 MB default
    // A guard to enure that the file is closed properly. If the user 
    // forgets to call close(). 
    ~StateWriter();

    StateWriter(const StateWriter&) = delete;
    StateWriter& operator=(const StateWriter&) = delete;
    StateWriter(StateWriter&&) = delete;
    StateWriter& operator=(StateWriter&&) = delete;
 
    void add_snapshot(std::vector<uint64_t> state);
    void close();
 
    // Exposed only for the demo/test below.
    std::size_t buffered_count() const { return snapshots_.size(); }
 
private:
    void open_file();
    void flush();
 
    std::string path_;
    uint16_t run_id_;
    std::fstream output_file_;
 
    std::vector<std::vector<uint64_t>> snapshots_;
    std::size_t buffer_capacity_bytes_;
    std::size_t buffer_bytes_ = 0;
    bool closed_ = false;
};

#endif // CORE_IO_STORAGE_HPP