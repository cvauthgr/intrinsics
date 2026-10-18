# Minimum CMake versions
cmake_minimum_required(VERSION 3.20)

# For clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++ Standard details
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Project name
project(intrinsics LANGUAGES CXX)

# Enable testing
enable_testing()

# Look for the packages for gtest and gbenchmark
find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

# Create the test/benchmark binaries
add_executable(intrinsics_tests tests/intrinsics_tests.cpp)
add_executable(intrinsics_benchmark benchmark/intrinsics_benchmark.cpp)

# Link against the main functions of the gtest and gbenchmark framework
target_link_libraries(intrinsics_tests PRIVATE intrinsics_lib GTest::gtest_main)
target_link_libraries(intrinsics_benchmark PRIVATE intrinsics_lib benchmark::benchmark_main)

# Find and include the testing framework
include(GoogleTest)
gtest_discover_tests(intrinsics_tests)

# Headers linking for main interface library
add_library(intrinsics_lib INTERFACE)
target_include_directories(intrinsics_lib INTERFACE include)

# Compiler flags - AVX2 and FMA flags added here!
target_compile_options(intrinsics_lib INTERFACE
    -Wall -Wextra -Wpedantic
    -O3 -march=native -mavx2 -mfma
)

# Executable
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE intrinsics_lib)
