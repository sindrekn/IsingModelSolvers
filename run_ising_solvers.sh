#!/bin/sh

cd /home/sindrekampennesheim/Documents/PhD/Optimizing/IsingModelSolver

# cmake --build build -j$(nproc)

# ============================== Testing L = 8 ==============================
# ./build/run --ising_solver wolff_quadratic --L 8 --t_c 2.269 --t_min 2.200 --t_max 2.338 --run_end 1500

# ./build/run --ising_solver quadratic_nnn --L 8 --t_c 5.31 --t_min 5.10 --t_max 5.52 --run_end 1500

# ./build/run --ising_solver wolff_triangular --L 8 --t_c 3.641 --t_min 3.50 --t_max 3.782 --run_end 1500

# ./build/run --ising_solver wolff_honeycomb --L 8 --t_c 1.519 --t_min 1.450 --t_max 1.588 --run_end 1500

# ./build/run --ising_solver wolff_cubic --L 6 --t_c 4.511 --t_min 4.300 --t_max 4.722 --run_end 1500

# ./build/run --ising_solver cubic_nnn --L 6 --t_c 12.2 --t_min 11.50 --t_max 12.90 --run_end 1500

# ./build/run --ising_solver lb_lrim --L 8 --sigma 1.75 --sigma_index 0 --run_end 1500 --t_c 3.04 --t_min 2.950 --t_max 3.126

# ./build/run --ising_solver lb_lrim --L 8 --sigma 2.00 --sigma_index 1 --run_end 1500 --t_c 2.903 --t_min 2.75 --t_max 3.056

# # ./build/run --ising_solver quadratic --L 8 --t_c 2.269 --t_min 1.90 --t_max 2.50 --run_end 1000

# # ./build/run --ising_solver wolff_quadratic --L 8 --t_c 2.269 --t_min 1.90 --t_max 2.50 --run_end 1000

# # ./build/run --ising_solver quadratic_nnn --L 8 --t_c 5.75 --t_min 4.50 --t_max 7.00 --run_end 1000

# # ./build/run --ising_solver triangular --L 8 --t_c 3.641 --t_min 3.00 --t_max 4.20 --run_end 1000

# # ./build/run --ising_solver wolff_triangular --L 8 --t_c 3.641 --t_min 3.00 --t_max 4.20 --run_end 1000

# # ./build/run --ising_solver honeycomb --L 8 --t_c 1.519 --t_min 1.20 --t_max 1.90 --run_end 1000

# # ./build/run --ising_solver wolff_honeycomb --L 8 --t_c 1.519 --t_min 1.20 --t_max 1.90 --run_end 1000

# # ./build/run --ising_solver cubic --L 8 --t_c 4.511 --t_min 4.00 --t_max 5.00 --run_end 1000

# # ./build/run --ising_solver wolff_cubic --L 8 --t_c 4.511 --t_min 4.00 --t_max 5.00 --run_end 1000

# # ./build/run --ising_solver cubic_nnn --L 8 --t_c 11.3 --t_min 10.00 --t_max 12.60 --run_end 1000

# # ./build/run --ising_solver lrim --L 8 --sigma 1.75 --sigma_index 0 --run_end 1000 --t_c 3.04 --t_min 2.60 --t_max 3.40

# # ./build/run --ising_solver lb_lrim --L 8 --sigma 1.75 --sigma_index 0 --run_end 1000 --t_c 3.04 --t_min 2.60 --t_max 3.40

# ============================================================================

# ============================== Testing L_1 ==============================

./build/run --ising_solver wolff_quadratic --L 64 --t_c 2.269 --t_min 2.200 --t_max 2.338 --run_end 1500

./build/run --ising_solver quadratic_nnn --L 64 --t_c 5.31 --t_min 5.10 --t_max 5.52 --run_end 1500

./build/run --ising_solver wolff_triangular --L 64 --t_c 3.641 --t_min 3.50 --t_max 3.782 --run_end 1500

./build/run --ising_solver wolff_honeycomb --L 64 --t_c 1.519 --t_min 1.450 --t_max 1.588 --run_end 1500

./build/run --ising_solver wolff_cubic --L 14 --t_c 4.511 --t_min 4.300 --t_max 4.722 --run_end 1500

./build/run --ising_solver cubic_nnn --L 14 --t_c 12.2 --t_min 11.50 --t_max 12.90 --run_end 1500

./build/run --ising_solver lb_lrim --L 48 --sigma 1.75 --sigma_index 0 --run_end 1500 --t_c 3.04 --t_min 2.950 --t_max 3.126

./build/run --ising_solver lb_lrim --L 48 --sigma 2.00 --sigma_index 1 --run_end 1500 --t_c 2.903 --t_min 2.75 --t_max 3.056

# ============================================================================

# ============================== Testing L_2 ==============================

./build/run --ising_solver wolff_quadratic --L 80 --t_c 2.269 --t_min 2.200 --t_max 2.338 --run_end 1500

./build/run --ising_solver quadratic_nnn --L 80 --t_c 5.31 --t_min 5.10 --t_max 5.52 --run_end 1500

./build/run --ising_solver wolff_triangular --L 80 --t_c 3.641 --t_min 3.50 --t_max 3.782 --run_end 1500

./build/run --ising_solver wolff_honeycomb --L 80 --t_c 1.519 --t_min 1.450 --t_max 1.588 --run_end 1500

./build/run --ising_solver wolff_cubic --L 18 --t_c 4.511 --t_min 4.300 --t_max 4.722 --run_end 1500

./build/run --ising_solver cubic_nnn --L 18 --t_c 12.2 --t_min 11.50 --t_max 12.90 --run_end 1500

./build/run --ising_solver lb_lrim --L 60 --sigma 1.75 --sigma_index 0 --run_end 1500 --t_c 3.04 --t_min 2.950 --t_max 3.126

./build/run --ising_solver lb_lrim --L 60 --sigma 2.00 --sigma_index 1 --run_end 1500 --t_c 2.903 --t_min 2.75 --t_max 3.056

# ============================================================================

# ============================== Testing L_3 ==============================

./build/run --ising_solver wolff_quadratic --L 96 --t_c 2.269 --t_min 2.200 --t_max 2.338 --run_end 1500

./build/run --ising_solver quadratic_nnn --L 96 --t_c 5.31 --t_min 5.10 --t_max 5.52 --run_end 1500

./build/run --ising_solver wolff_triangular --L 96 --t_c 3.641 --t_min 3.50 --t_max 3.782 --run_end 1500

./build/run --ising_solver wolff_honeycomb --L 96 --t_c 1.519 --t_min 1.450 --t_max 1.588 --run_end 1500

./build/run --ising_solver wolff_cubic --L 22 --t_c 4.511 --t_min 4.300 --t_max 4.722 --run_end 1500

./build/run --ising_solver cubic_nnn --L 22 --t_c 12.2 --t_min 11.50 --t_max 12.90 --run_end 1500

./build/run --ising_solver lb_lrim --L 72 --sigma 1.75 --sigma_index 0 --run_end 1500 --t_c 3.04 --t_min 2.950 --t_max 3.126

./build/run --ising_solver lb_lrim --L 72 --sigma 2.00 --sigma_index 1 --run_end 1500 --t_c 2.903 --t_min 2.75 --t_max 3.056

# ============================================================================


# ./build/run --ising_solver wolff_triangular --L 32 --t_c 3.641 --t_min 3.00 --t_max 4.20 --run_end 1000
# ./build/run --ising_solver wolff_honeycomb --L 32 --t_c 1.519 --t_min 1.20 --t_max 1.90 --run_end 1000
# ./build/run --ising_solver wolff_cubic --L 14 --t_c 4.511 --t_min 4.00 --t_max 5.00 --run_end 1000

# ./build/run --ising_solver wolff_triangular --L 48 --t_c 3.641 --t_min 3.00 --t_max 4.20 --run_end 1000
# ./build/run --ising_solver wolff_honeycomb --L 48 --t_c 1.519 --t_min 1.20 --t_max 1.90 --run_end 1000
# ./build/run --ising_solver wolff_cubic --L 18 --t_c 4.511 --t_min 4.00 --t_max 5.00 --run_end 1000

# ./build/run --ising_solver wolff_triangular --L 64 --t_c 3.641 --t_min 3.00 --t_max 4.20 --run_end 1000
# ./build/run --ising_solver wolff_honeycomb --L 64 --t_c 1.519 --t_min 1.20 --t_max 1.90 --run_end 1000
# ./build/run --ising_solver wolff_cubic --L 22 --t_c 4.511 --t_min 4.00 --t_max 5.00 --run_end 1000


# ./build/run --ising_solver lb --L 16 --sigma 1.75 --sigma_index 0 --run_end 1000 --t_c 3.04 --t_min 2.60 --t_max 3.40
# ./build/run --ising_solver lb --L 16 --sigma 2.00 --sigma_index 1 --run_end 1000 --t_c 2.90 --t_min 2.50 --t_max 3.30

# ./build/run --ising_solver lb --L 32 --sigma 1.75 --sigma_index 0 --run_end 1000 --t_c 3.04 --t_min 2.60 --t_max 3.40
# ./build/run --ising_solver lb --L 32 --sigma 2.00 --sigma_index 1 --run_end 1000 --t_c 2.90 --t_min 2.50 --t_max 3.30

# ./build/run --ising_solver lb --L 48 --sigma 1.75 --sigma_index 0 --run_end 1000 --t_c 3.04 --t_min 2.60 --t_max 3.40
# ./build/run --ising_solver lb --L 48 --sigma 2.00 --sigma_index 1 --run_end 1000 --t_c 2.90 --t_min 2.50 --t_max 3.30

# ./build/run --ising_solver lb --L 64 --sigma 1.75 --sigma_index 0 --run_end 1000 --t_c 3.04 --t_min 2.60 --t_max 3.40
# ./build/run --ising_solver lb --L 64 --sigma 2.00 --sigma_index 1 --run_end 1000 --t_c 2.90 --t_min 2.50 --t_max 3.30

# python analysis/simple_ann.py -ymin 0.01 -ymax 0.99 -L 32 48 64 -N 1024 2304 4096 -I quadratic -t 60 -r 1000 -tc 2.27 -ts True
# python analysis/simple_ann.py -ymin 0.01 -ymax 0.99 -L 32 48 64 -N 1024 2304 4096 -I wolff -t 60 -r 1000 -tc 2.27 -ts True
# python analysis/simple_ann.py -ymin 0.01 -ymax 0.99 -L 32 48 64 -N 1024 2304 4096 -I quadratic_nnn -t 60 -r 1000
# python analysis/simple_ann.py -ymin 0.01 -ymax 0.99 -L 32 48 64 -N 1024 2304 4096 -I honeycomb -t 60 -r 1000 -tc 1.52 -ts True
# python analysis/simple_ann.py -ymin 0.01 -ymax 0.99 -L 32 48 64 -N 1024 2304 4096 -I triangular -t 60 -r 1000 -tc 3.64 -ts True
# python analysis/simple_ann.py -ymin 0.01 -ymax 0.99 -L 32 48 64 -N 1024 2304 4096 -I lb -t 60 -r 1000 -tc 3.04 -si 0
# python analysis/simple_ann.py -ymin 0.01 -ymax 0.99 -L 32 48 64 -N 1024 2304 4096 -I lb -t 60 -r 1000 -tc 2.90 -si 1
# python analysis/simple_ann.py -ymin 0.01 -ymax 0.99 -L 12 16 22 -N 1728 4096 10648 -I cubic -t 60 -r 1000 -tc 4.511
# python analysis/simple_ann.py -ymin 0.01 -ymax 0.99 -L 12 16 22 -N 1728 4096 10648 -I cubic_nnn -t 60 -r 1000 
# python analysis/simple_ann.py -ymin 0.01 -ymax 0.99 -L 8 16 32 -N 64 256 1024 -I lrim -t 60 -r 1000 -tc 3.04 -si 0

# python analysis/avg_mag.py -L 32 48 64 -N 1024 2304 4096 -I quadratic -t 60 -r 1000 -tc 2.27 -ts True
# python analysis/avg_mag.py -L 32 48 64 -N 1024 2304 4096 -I wolff -t 60 -r 1000 -tc 2.27 -ts True
# python analysis/avg_mag.py -L 32 48 64 -N 1024 2304 4096 -I quadratic_nnn -t 60 -r 1000
# python analysis/avg_mag.py -L 32 48 64 -N 1024 2304 4096 -I honeycomb -t 60 -r 1000 -tc 1.52 -ts True
# python analysis/avg_mag.py -L 32 48 64 -N 1024 2304 4096 -I triangular -t 60 -r 1000 -tc 3.64 -ts True
# python analysis/avg_mag.py -L 32 48 64 -N 1024 2304 4096 -I lb -t 60 -r 1000 -tc 3.04 -si 0
# python analysis/avg_mag.py -L 32 48 64 -N 1024 2304 4096 -I lb -t 60 -r 1000 -tc 2.90 -si 1
# python analysis/avg_mag.py -L 12 16 22 -N 1728 4096 10648 -I cubic -t 60 -r 1000 -tc 4.511
# python analysis/avg_mag.py -L 12 16 22 -N 1728 4096 10648 -I cubic_nnn -t 60 -r 1000 
# python analysis/avg_mag.py -L 8 16 32 -N 64 256 1024 -I lrim -t 60 -r 1000 -tc 3.04 -si 0


