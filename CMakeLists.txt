cmake_minimum_required(VERSION 3.20)
project(IsingModelSolver CXX)

set(CMAKE_INTERPROCEDURAL_IsingModelSolver TRUE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Werror -march=native)

find_package(Threads REQUIRED)

# --- CLI11 ---
include(FetchContent)
FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.4.1
)
FetchContent_MakeAvailable(CLI11)

# --- Core library ---
add_library(core
    core/lattice/lattice.cpp
    core/schedule/schedule.cpp
    core/io/storage.cpp
)
target_include_directories(core PUBLIC ${CMAKE_SOURCE_DIR})

# --- solvers ---
add_library(solvers
    solvers/solvers.cpp
)
target_link_libraries(solvers PUBLIC core)

add_library(runner backend/runner.cpp)
target_link_libraries(runner PUBLIC solvers)

# --- Executable ---
add_executable(run main.cpp)
target_link_libraries(run PRIVATE runner core CLI11::CLI11)


