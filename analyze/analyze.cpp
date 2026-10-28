#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <fftw3.h>

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
// File format:
//   [uint64_t num_words][num_words * uint64_t bit-packed spin words]...
// repeated until EOF.
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

    std::optional<std::vector<uint64_t>> next() {
        while (true) {
            uint64_t num_words = 0;
            file_.read(reinterpret_cast<char*>(&num_words), sizeof(num_words));

            if (file_.eof()) {
                return std::nullopt;
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
                continue;
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

double magnetization(const std::vector<int>& spins) {
    double sum = 0.0;
    for (int s : spins) sum += s;
    return sum / static_cast<double>(spins.size());
}


struct EnergyAnalyzer {
    int L, N, Nk;
    std::vector<double> Khat;            // FFT of kernel, size L*(L/2+1)
    double* real_buf;                    // L*L
    fftw_complex* cplx_buf;              // L*(L/2+1)
    fftw_plan fwd, inv;

    EnergyAnalyzer(int L_, double sigma) : L(L_), N(L_*L_), Nk(L_*(L_/2 + 1)) {
        real_buf = fftw_alloc_real(N);
        cplx_buf = fftw_alloc_complex(Nk);
        fwd = fftw_plan_dft_r2c_2d(L, L, real_buf, cplx_buf, FFTW_MEASURE);
        inv = fftw_plan_dft_c2r_2d(L, L, cplx_buf, real_buf, FFTW_MEASURE);

        // Kernel indexed by displacement mod L, same box [-L/2, L/2) and
        // normalisation (sum = 4) as your LongRange_DistancePrecompute.
        std::vector<double> K(N, 0.0);
        double sum = 0.0;
        for (int y = 0; y < L; ++y)
            for (int x = 0; x < L; ++x) {
                if (x == 0 && y == 0) continue;
                int dx = (x >= L/2) ? x - L : x;
                int dy = (y >= L/2) ? y - L : y;
                double v = std::pow(double(dx*dx + dy*dy), -(2.0 + sigma) / 2.0);
                K[y*L + x] = v;
                sum += v;
            }
        for (double& v : K) v *= 4.0 / sum;

        for (int i = 0; i < N; ++i) real_buf[i] = K[i];
        fftw_execute(fwd);
        Khat.resize(Nk);
        for (int q = 0; q < Nk; ++q) Khat[q] = cplx_buf[q][0];  // real, since K is symmetric
    }

    ~EnergyAnalyzer() {
        fftw_destroy_plan(fwd); fftw_destroy_plan(inv);
        fftw_free(real_buf);    fftw_free(cplx_buf);
    }

    double energy(const std::vector<int>& s) {
        for (int i = 0; i < N; ++i) real_buf[i] = s[i];
        fftw_execute(fwd);                                   // s -> s_hat
        for (int q = 0; q < Nk; ++q) {                       // h_hat = K_hat * s_hat
            cplx_buf[q][0] *= Khat[q];
            cplx_buf[q][1] *= Khat[q];
        }
        fftw_execute(inv);                                   // h (unnormalised: times N)
        double acc = 0.0;
        for (int i = 0; i < N; ++i) acc += s[i] * real_buf[i];
        return -0.5 * acc / N;                               // 1/N: FFTW normalisation, 1/2: each pair once
    }
};

// ---------------------------------------------------------------------------
// Reader-based observables
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

        std::cout << "Magnetization: " << magnetization(spins) << "\n";
    }

    std::cout << "Total snapshots read: " << reader.snapshot_index() << "\n";
}

std::vector<double> return_mag(const std::string& filepath, int num_spins) {
    SnapshotReader reader(filepath, num_spins);
    std::vector<double> m;

    while (auto state = reader.next()) {
        auto spins = unpack_spins(*state, num_spins);
        m.push_back(magnetization(spins));
    }
    return m;
}


std::vector<double> return_energyLRIM(const std::string& filepath, int num_spins, double sigma) {
    SnapshotReader reader(filepath, num_spins);
    int L = static_cast<int>(std::sqrt(num_spins));
    EnergyAnalyzer analyzer(L, sigma);
    std::vector<double> e;

    while (auto state = reader.next()) {
        auto spins = unpack_spins(*state, num_spins);
        e.push_back(analyzer.energy(spins));
    }
    return e;
}

// NOTE: 2D only (L = sqrt(N)).
std::vector<int64_t> build_regions(int64_t N, int64_t A)
{
    const int64_t num_regions = N / (A * A);
    const int64_t L = static_cast<int64_t>(std::sqrt(static_cast<double>(N)));
    const int64_t Asize = L / A;
    const int64_t region_size = A * A;

    std::vector<int64_t> regions(num_regions * region_size, 0);

    int64_t a = 0;
    for (int64_t arow = 0; arow < Asize; ++arow) {
        for (int64_t acol = 0; acol < Asize; ++acol) {
            int64_t i = 0;
            for (int64_t row = arow * A; row < (arow + 1) * A; ++row) {
                for (int64_t col = acol * A; col < (acol + 1) * A; ++col) {
                    regions[a * region_size + i] = row * L + col;
                    ++i;
                }
            }
            ++a;
        }
    }
    return regions;
}

std::vector<double> return_regmag(
    const std::string& filepath,
    int num_spins,
    int A,
    double v0,
    double w,
    double b)
{
    std::vector<double> y;
    SnapshotReader reader(filepath, num_spins);

    const int region_size = A * A;
    const int num_regions = num_spins / region_size;
    const std::vector<int64_t> regions = build_regions(num_spins, A);

    while (auto state = reader.next()) {
        auto spins = unpack_spins(*state, num_spins);

        double psi_sum = 0.0;
        for (int a = 0; a < num_regions; ++a) {
            const int64_t* idx = &regions[static_cast<size_t>(a) * region_size];
            int s = 0;
            for (int i = 0; i < region_size; ++i)
                s += spins[idx[i]];

            const double m = static_cast<double>(s) / region_size;
            psi_sum += ReLu(std::abs(m) + b);
        }

        y.push_back(sigmoid(w * psi_sum + v0));
    }
    return y;
}

// ---------------------------------------------------------------------------
// CSV export (ported from the Python script)
// ---------------------------------------------------------------------------

struct Args {
    std::string order_parameter;
    int N;
    int L;
    int num_temps = 0;
    int num_runs = 0;
    int A;
    double v0 = 0.0;
    double w = 1.0;
    double b = 0.0;
    double sigma = 0.0;
    std::string dir_path;
    std::string save_path;
};

// Writes columns temp_0..temp_{n-1}, rows snapshot_0..snapshot_{m-1},
// matching pandas' to_csv(index=True) layout.
void write_csv(const std::string& save_path, const std::vector<std::vector<double>>& columns) {
    const size_t rows = columns.empty() ? 0 : columns[0].size();
    for (size_t t = 0; t < columns.size(); ++t) {
        if (columns[t].size() != rows) {
            throw std::runtime_error("Column temp_" + std::to_string(t) + " has " +
                                     std::to_string(columns[t].size()) + " rows, expected " +
                                     std::to_string(rows) + " (all temperatures need equal snapshot counts)");
        }
    }

    std::filesystem::path p(save_path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());

    std::ofstream out(save_path);
    if (!out.is_open()) throw std::runtime_error("Failed to open for writing: " + save_path);
    out << std::setprecision(std::numeric_limits<double>::max_digits10);

    out << "";
    for (size_t t = 0; t < columns.size(); ++t) out << ",temp_" << t;
    out << "\n";
    for (size_t r = 0; r < rows; ++r) {
        out << "snapshot_" << r;
        for (size_t t = 0; t < columns.size(); ++t) out << "," << columns[t][r];
        out << "\n";
    }
}

template <typename Observable>
void save_observable(const Args& args, const std::string& dir_path,
                     const std::string& save_path, Observable&& obs) {
    std::vector<std::vector<double>> columns(args.num_temps);
    for (int temp = 0; temp < args.num_temps; ++temp) {
        for (int run = 0; run < args.num_runs; ++run) {
            const std::string filename = dir_path + "/temp_" + std::to_string(temp) +
                                         "/Snapshots_" + std::to_string(run) + ".dat";
            std::vector<double> v = obs(filename);
            columns[temp].insert(columns[temp].end(), v.begin(), v.end());
        }
    }
    write_csv(save_path, columns);
}

void save_magnetization(const Args& args, const std::string& dir_path,
                        const std::string& save_path, int num_spins) {
    save_observable(args, dir_path, save_path, [&](const std::string& f) {
        return return_mag(f, num_spins);
    });
}

void save_energyLR(const Args& args, const std::string& dir_path,
                   const std::string& save_path, int num_spins, double sigma) {
    save_observable(args, dir_path, save_path, [&](const std::string& f) {
        return return_energyLRIM(f, num_spins, sigma);
    });
}

void save_regmag(const Args& args, const std::string& dir_path,
                 const std::string& save_path, int num_spins, int A) {
    if (num_spins % (A * A) != 0) {
        throw std::invalid_argument("num_spins (" + std::to_string(num_spins) +
                                    ") must be divisible by A^2 (" + std::to_string(A * A) +
                                    ") for regmag calculation.");
    }
    save_observable(args, dir_path, save_path, [&](const std::string& f) {
        return return_regmag(f, num_spins, A, args.v0, args.w, args.b);
    });
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

static void print_usage(const char* prog) {
    std::cerr
        << "Usage: " << prog << " -op <magnetization|energyLRIM|regmag> -N <sizes...> -I <solver> -t <num_temps> -r <num_runs>\n"
        << "         [-si <sigma_index>] [-A <region sizes...>] [-v0 <x>] [-w <x>] [-b <x>] [--base-root <dir>]\n";
}

static bool is_flag(const std::string& s) {
    // A flag starts with '-' followed by a non-digit/non-dot (so negatives are values).
    return s.size() > 1 && s[0] == '-' && !(std::isdigit(static_cast<unsigned char>(s[1])) || s[1] == '.');
}

Args parse_args(int argc, char** argv) {
    Args args;
    auto need_value = [&](int& i) -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(std::string("Missing value for ") + argv[i]);
        return argv[++i];
    };
    auto int_list = [&](int& i) {
        std::vector<int> v;
        while (i + 1 < argc && !is_flag(argv[i + 1])) v.push_back(std::stoi(argv[++i]));
        if (v.empty()) throw std::invalid_argument(std::string("Missing values for ") + argv[i]);
        return v;
    };

    bool have_op = false, have_N = false, have_L = false, have_t = false, have_r = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-op" || a == "--order-parameter") { args.order_parameter = need_value(i); have_op = true; }
        else if (a == "-L" || a == "--lattice-size") { args.L = std::stoi(need_value(i)); have_L = true; }
        else if (a == "-N" || a == "--num-spins") { args.N = std::stoi(need_value(i)); have_N = true; }
        else if (a == "-t" || a == "--num-temps") { args.num_temps = std::stoi(need_value(i)); have_t = true; }
        else if (a == "-r" || a == "--num-runs") { args.num_runs = std::stoi(need_value(i)); have_r = true; }
        else if (a == "-A" || a == "--A") { args.A = std::stoi(need_value(i)); }
        else if (a == "-v0" || a == "--v0") { args.v0 = std::stod(need_value(i)); }
        else if (a == "-w" || a == "--w") { args.w = std::stod(need_value(i)); }
        else if (a == "-b" || a == "--b") { args.b = std::stod(need_value(i)); }
        else if (a == "--sigma") { args.sigma = std::stod(need_value(i)); }
        else if (a == "--dir-path") { args.dir_path = need_value(i); }
        else if (a == "--save-path") { args.save_path = need_value(i); }
        else throw std::invalid_argument("Unknown argument: " + a);
    }
    if (!(have_op && have_N && have_L && have_t && have_r)) {
        throw std::invalid_argument("Missing required argument(s)");
    }
    return args;
}

static bool contains(const std::string& s, const char* sub) {
    return s.find(sub) != std::string::npos;
}

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 2;
    }

    try {
        if (args.order_parameter == "energyLRIM") {
            if (args.sigma <= 0) {
                throw std::invalid_argument("energyLR needs sigma > 0, but got sigma=" + std::to_string(args.sigma));
            }
            save_energyLR(args, args.dir_path, args.save_path, args.N, args.sigma);
        } else if (args.order_parameter == "magnetization") {
            save_magnetization(args, args.dir_path, args.save_path, args.N);
        } else if (args.order_parameter == "regmag") {
            if (args.A <= 0) {
                throw std::invalid_argument("regmag needs A > 0, but got A=" + std::to_string(args.A));
            }
            save_regmag(args, args.dir_path, args.save_path, args.N, args.A);
        } else {
            throw std::invalid_argument("Unknown order parameter: " + args.order_parameter);
        }

        std::cout << "Saved average " << args.order_parameter
                    << " to " << args.save_path << "\n";
    
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}