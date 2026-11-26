# Intrinsics

AVX2/FMA intrinsics experiments in C++23

The kernels are header-only and live in `include/`. Each operation is written several times,
going from a scalar baseline to SIMD versions with more work done per iteration:

- **naive** : plain scalar loop, auto-vectorization disabled with `#pragma GCC optimize("no-tree-vectorize")` (GCC-specific, Clang ignores it)
- **simd** : AVX2 with unaligned loads/stores
- **aligned** : AVX2 with aligned loads/stores
- **unrolled** : several independent vectors per iteration
- **threaded** : a small spinning thread pool (`include/thread_pool.hpp`) with 7 workers pinned to logical CPUs 2, 4, ..., 14

Operations :

- `int32` vector add (`C = A + B`)
- `double` FMA (`C += A * B`)
- `double` vector add, split across the thread pool
- `double` inner product with 4 accumulators, compared against `std::inner_product`

**Note** : the aligned kernels require 32-byte aligned inputs (aligned AVX loads/stores fault otherwise).
The inner product sums in a different order than a sequential loop, so its result can differ from
`std::inner_product` in the last bits.

## Benchmarks

The small single-threaded benchmarks use 1000-element inputs, which fit in the 32 KiB L1d cache,
so they measure in-cache throughput, not memory bandwidth. The threaded add and the inner product are
also run on larger inputs (up to 10'000'000 elements).

Measured on an AMD Ryzen 7 5825U (Zen 3, 32 KiB L1d, 512 KiB L2, 16 MiB L3).

**Method :** single-threaded benchmarks pinned to one core with `taskset`, mean of 10 repetitions.
The governor was `powersave` with boost enabled, so absolute numbers depend on boost behaviour.

1000 elements (L1-resident), ns per call :

| Kernel | int32 add | double FMA |
|:-------|----------:|-----------:|
| naive (no auto-vectorization) | 339 ns | 375 ns |
| AVX2 unaligned | 46.9 ns | 139 ns |
| AVX2 aligned | 36.9 ns | 99.4 ns |
| AVX2 unaligned, unrolled x2 | - | 129 ns |
| AVX2 aligned, unrolled x2 | - | 98.6 ns |

Inner product, AVX2 (4 accumulators) vs `std::inner_product` :

| Elements | Working set | AVX2 | `std::inner_product` | Speedup |
|---------:|:------------|-----:|---------------------:|--------:|
| 1'000 | 16 KB (L1) | 60.6 ns | 659 ns | 10.9x |
| 10'000 | 160 KB (L2) | 1.19 µs | 6.94 µs | 5.8x |
| 100'000 | 1.6 MB (L3) | 15.0 µs | 70.7 µs | 4.7x |
| 1'000'000 | 16 MB (L3 boundary) | 313 µs | 737 µs | 2.4x |
| 10'000'000 | 160 MB (DRAM) | 5.70 ms | 7.60 ms | 1.33x |

`std::inner_product` runs at roughly the same rate at every size. With GCC at `-O3` the multiplications get
vectorized, but without `-ffast-math` the compiler may not reorder the sum, so the additions stay one sequential
scalar chain (`vaddsd`) and are limited by add latency. The AVX2 version slows down as the data leaves the caches,
so the gap shrinks until both are close at 10M elements.

Threaded double add (7 workers on separate physical cores) :

| Elements | Working set | Time | Bytes/s (A, B, C) |
|---------:|:------------|-----:|------------------:|
| 8'000 | 192 KB | 340 ns | 526 GiB/s |
| 100'000 | 2.4 MB | 5.05 µs | 443 GiB/s |
| 10'000'000 | 240 MB (DRAM) | 15.3 ms | 14.6 GiB/s |

The first two sizes fit in the per-core caches, so the bytes/s is the summed cache bandwidth of 7 cores, not memory bandwidth.
At 10M elements the add is DRAM-bound. The counter only includes the A, B and C streams, not the extra read of C
caused by write-allocate, so the real DRAM traffic is higher than shown.

## Requirements

- CMake version **3.20 and above**
- g++ (or any C++ compiler) supporting **C++23**
- GoogleTest and Google Benchmark installed
- An x86-64 CPU with **AVX2** and **FMA** (the project compiles with `-O3 -march=native -mavx2 -mfma`)
- Linux (the thread pool uses `pthread_setaffinity_np`)

## Build

```bash
# Generate the build scripts for Ninja
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
# Build the project
cmake --build build
# Run the tests
ctest --test-dir build
# Run the benchmarks
./build/intrinsics_benchmark
```
