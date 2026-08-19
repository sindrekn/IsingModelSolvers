#include "core/io/storage.hpp"

void create_directory(const std::string& path) {
    if (mkdir(path.c_str(), 0777) == -1) {
        if (errno == EEXIST) {
            std::cerr << "Warning: Directory already exists: " << path << std::endl;
        } else {
            std::cerr << "Warning: Failed to create directory: " << path
                       << " (" << std::strerror(errno) << ")" << std::endl;
        }

        std::cerr << "Continue anyway? [y/n]: ";
        char response;
        std::cin >> response;

        if (response != 'y' && response != 'Y') {
            std::cerr << "Aborting." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

std::vector<std::string> setup_temperature_directories(const std::vector<double>& temps, const std::string& base_dir) {
    if (!std::filesystem::exists(base_dir)) {
        std::cerr << "Error: Base directory does not exist: " << base_dir << std::endl;
        return {};
    }

    std::fstream temp_index_file(base_dir + "/temp_index.txt", std::ios::out);
    if (temp_index_file) {
        temp_index_file << "Index\tTemperature\n";
        int index = 0;
        for (const auto& temp : temps) {
            temp_index_file << index << "\t \t" << temp << "\n";
            index++;
        }
        temp_index_file.close();
    }
    else {
        std::cerr << "Error opening temp_index.txt for writing." << std::endl;
        return {};
    }

    std::vector<std::string> temp_dirs;
    for (std::size_t temp_index = 0; temp_index < temps.size(); temp_index++) {
        std::string temp_dir = base_dir + "/temp_" + std::to_string(temp_index);
        create_directory(temp_dir);
        temp_dirs.push_back(temp_dir);
    }
    return temp_dirs;
}

StateWriter::StateWriter(std::string path, uint16_t run_id,
                          std::size_t buffer_capacity_bytes)
    : path_(std::move(path)),
      run_id_(run_id),
      buffer_capacity_bytes_(buffer_capacity_bytes) {
    open_file();
}

void StateWriter::open_file() {
    const std::string filename = "Snapshots_" + std::to_string(run_id_) + ".dat";
    const std::string full_path = path_ + "/" + filename;
 
    output_file_.open(full_path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!output_file_.is_open()) {
        throw std::runtime_error("StateWriter: failed to open file " + full_path);
    }
}

void StateWriter::add_snapshot(std::vector<uint64_t> state) {
    if (closed_) {
        throw std::runtime_error("StateWriter: cannot add snapshot after close()");
    }
    buffer_bytes_ += state.size() * sizeof(uint64_t);
    snapshots_.push_back(std::move(state));

    if (buffer_bytes_ >= buffer_capacity_bytes_) {
        flush();
    }
}

void StateWriter::flush() {
    if (snapshots_.empty()) {
        return;
    }
 
    for (const auto& state : snapshots_) {
        const uint64_t num_words = state.size();
        output_file_.write(reinterpret_cast<const char*>(&num_words), sizeof(num_words));
        output_file_.write(reinterpret_cast<const char*>(state.data()),
                            static_cast<std::streamsize>(num_words * sizeof(uint64_t)));
    }
    output_file_.flush();
    if (!output_file_) {
        throw std::runtime_error("StateWriter: write failed during flush()");
    }
 
    std::vector<std::vector<uint64_t>>().swap(snapshots_);
    buffer_bytes_ = 0;
}
 
void StateWriter::close() {
    if (closed_) {
        return;
    }
    flush();
    output_file_.close();
    closed_ = true;
}
 
StateWriter::~StateWriter() {
    if (!closed_) {
        try {
            close();
        } catch (...) {
        }
    }
}
