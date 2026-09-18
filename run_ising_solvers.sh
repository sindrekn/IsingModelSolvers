#!/bin/sh

cd /home/sindrekampennesheim/Documents/PhD/Optimizing/IsingModelSolver

# cmake --build build -j$(nproc)

# ./build/run --ising_solver quadratic --L 16 --t_c 2.269 --t_min 1.90 --t_max 2.50 --run_end 1000
# ./build/run --ising_solver quadratic --L 32 --t_c 2.269 --t_min 1.90 --t_max 2.50 --run_end 1000
# ./build/run --ising_solver quadratic --L 64 --t_c 2.269 --t_min 1.90 --t_max 2.50 --run_end 1000

# ./build/run --ising_solver quadratic_nnn --L 16 --t_c 5.75 --t_min 4.50 --t_max 7.00 --run_end 1000
# ./build/run --ising_solver quadratic_nnn --L 32 --t_c 5.75 --t_min 4.50 --t_max 7.00 --run_end 1000
# ./build/run --ising_solver quadratic_nnn --L 64 --t_c 5.75 --t_min 4.50 --t_max 7.00 --run_end 1000

# ./build/run --ising_solver triangular --L 16 --t_c 3.641 --t_min 3.00 --t_max 4.20 --run_end 1000
# ./build/run --ising_solver triangular --L 32 --t_c 3.641 --t_min 3.00 --t_max 4.20 --run_end 1000
# ./build/run --ising_solver triangular --L 64 --t_c 3.641 --t_min 3.00 --t_max 4.20 --run_end 1000

# ./build/run --ising_solver honeycomb --L 16 --t_c 1.519 --t_min 1.20 --t_max 1.90 --run_end 1000
# ./build/run --ising_solver honeycomb --L 32 --t_c 1.519 --t_min 1.20 --t_max 1.90 --run_end 1000
# ./build/run --ising_solver honeycomb --L 64 --t_c 1.519 --t_min 1.20 --t_max 1.90 --run_end 1000

./build/run --ising_solver cubic --L 12 --t_c 4.511 --t_min 4.00 --t_max 5.00 --run_end 1000
# ./build/run --ising_solver cubic --L 16 --t_c 4.511 --t_min 4.00 --t_max 5.00 --run_end 1000
# ./build/run --ising_solver cubic --L 22 --t_c 4.511 --t_min 4.00 --t_max 5.00 --run_end 1000

./build/run --ising_solver cubic_nnn --L 12 --t_c 11.3 --t_min 10.00 --t_max 12.60 --run_end 1000
./build/run --ising_solver cubic_nnn --L 16 --t_c 11.3 --t_min 10.00 --t_max 12.60 --run_end 1000
./build/run --ising_solver cubic_nnn --L 22 --t_c 11.3 --t_min 10.00 --t_max 12.60 --run_end 1000

# ./build/run --ising_solver wolff --L 16 --t_c 2.269 --t_min 1.90 --t_max 2.50 --run_end 1000
# ./build/run --ising_solver wolff --L 32 --t_c 2.269 --t_min 1.90 --t_max 2.50 --run_end 1000
# ./build/run --ising_solver wolff --L 64 --t_c 2.269 --t_min 1.90 --t_max 2.50 --run_end 1000

./build/run --ising_solver lrim --L 16 --sigma 1.75 --sigma_index 0 --run_end 1000 --t_c 3.04 --t_min 2.60 --t_max 3.40
./build/run --ising_solver lrim --L 32 --sigma 1.75 --sigma_index 0 --run_end 1000 --t_c 3.04 --t_min 2.60 --t_max 3.40
./build/run --ising_solver lrim --L 48 --sigma 1.75 --sigma_index 0 --run_end 1000 --t_c 3.04 --t_min 2.60 --t_max 3.40

./build/run --ising_solver lrim --L 16 --sigma 2.00 --sigma_index 1 --run_end 1000 --t_c 2.90 --t_min 2.50 --t_max 3.30
./build/run --ising_solver lrim --L 32 --sigma 2.00 --sigma_index 1 --run_end 1000 --t_c 2.90 --t_min 2.50 --t_max 3.30
./build/run --ising_solver lrim --L 48 --sigma 2.00 --sigma_index 1 --run_end 1000 --t_c 2.90 --t_min 2.50 --t_max 3.30

./build/run --ising_solver lrim --L 16 --sigma 2.50 --sigma_index 2 --run_end 1000 --t_c 2.70 --t_min 2.30 --t_max 3.10
./build/run --ising_solver lrim --L 32 --sigma 2.50 --sigma_index 2 --run_end 1000 --t_c 2.70 --t_min 2.30 --t_max 3.10
./build/run --ising_solver lrim --L 48 --sigma 2.50 --sigma_index 2 --run_end 1000 --t_c 2.70 --t_min 2.30 --t_max 3.10


