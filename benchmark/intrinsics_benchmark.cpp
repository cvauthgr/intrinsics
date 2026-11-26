#include <benchmark/benchmark.h>

#include "naive_add.hpp"
#include "simd_add.hpp"
#include "naive_fma.hpp"
#include "simd_fma.hpp"
#include "simd_fma_unrolled.hpp"
#include "simd_fma_unrolled_and_aligned.hpp"
#include "simd_add_aligned.hpp"
#include "simd_fma_aligned.hpp"
#include "thread_pool.hpp"
#include "simd_inner_product_quad_unrolled_and_aligned.hpp"

#include <cstddef>
#include <immintrin.h>
#include <numeric>
#include <vector>
#include <span>
#include <cstdlib>
#include <cstring>

static void BM_NAIVE_ADD(benchmark::State& state){

    std::vector<int> vec1(1000,1);
    std::vector<int> vec2(1000,2);
    std::vector<int> result_naive(1000,0);

    benchmark::DoNotOptimize(vec1.data());
    benchmark::DoNotOptimize(vec2.data());
    benchmark::DoNotOptimize(result_naive.data());

    for(auto _ : state){
        naive_addition(vec1, vec2, result_naive);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations()*result_naive.size());
    state.SetBytesProcessed(state.iterations()*result_naive.size()*3*sizeof(int));

}
BENCHMARK(BM_NAIVE_ADD);

static void BM_SIMD_ADD(benchmark::State& state){

    std::vector<int> vec1(1000,1);
    std::vector<int> vec2(1000,2);
    std::vector<int> result_simd(1000,0);

    benchmark::DoNotOptimize(vec1.data());
    benchmark::DoNotOptimize(vec2.data());
    benchmark::DoNotOptimize(result_simd.data());

    for(auto _ : state){
        simd_addition(vec1, vec2, result_simd);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations()*result_simd.size());
    state.SetBytesProcessed(state.iterations()*result_simd.size()*3*sizeof(int));

}
BENCHMARK(BM_SIMD_ADD);

static void BM_NAIVE_FMA(benchmark::State& state){
    std::vector<double> A(1000,10);
    std::vector<double> B(1000,10);
    std::vector<double> C(1000,50);

    benchmark::DoNotOptimize(A.data());
    benchmark::DoNotOptimize(B.data());
    benchmark::DoNotOptimize(C.data());

    for(auto _ : state){
        naive_fma(A,B,C);
        benchmark::ClobberMemory() ;
    }

    state.SetItemsProcessed(state.iterations()*C.size());
    state.SetBytesProcessed(state.iterations()*C.size()*3*sizeof(double));

}
BENCHMARK(BM_NAIVE_FMA);

static void BM_SIMD_FMA(benchmark::State& state){
    std::vector<double> A(1000,10);
    std::vector<double> B(1000,10);
    std::vector<double> C(1000,50);

    benchmark::DoNotOptimize(A.data());
    benchmark::DoNotOptimize(B.data());
    benchmark::DoNotOptimize(C.data());

    for(auto _ : state){
        simd_fma(A,B,C);
        benchmark::ClobberMemory() ;
    }

    state.SetItemsProcessed(state.iterations()*C.size());
    state.SetBytesProcessed(state.iterations()*C.size()*3*sizeof(double));

}
BENCHMARK(BM_SIMD_FMA);

static void BM_SIMD_FMA_UNROLLED(benchmark::State& state){
    std::vector<double> A(1000,10);
    std::vector<double> B(1000,10);
    std::vector<double> C(1000,50);

    benchmark::DoNotOptimize(A.data());
    benchmark::DoNotOptimize(B.data());
    benchmark::DoNotOptimize(C.data());

    for(auto _ : state){
        simd_fma_unrolled(A,B,C);
        benchmark::ClobberMemory() ;
    }

    state.SetItemsProcessed(state.iterations()*C.size());
    state.SetBytesProcessed(state.iterations()*C.size()*3*sizeof(double));

}
BENCHMARK(BM_SIMD_FMA_UNROLLED);

static void BM_SIMD_ADD_ALIGNED(benchmark::State& state){
    // Assign the buffers

    std::size_t n_elements = 1000 ;
    std::size_t bytes = n_elements*sizeof(__m256i);
    std::size_t alignment = alignof(__m256i);

   if (bytes % alignment != 0) {
        bytes += alignment - (bytes % alignment); // std::aligned_alloc requires size to be a multiple of alignment
    }

    int* A = static_cast<int*>(std::aligned_alloc(alignment , bytes));
    int* B = static_cast<int*>(std::aligned_alloc(alignment , bytes));
    int* C = static_cast<int*>(std::aligned_alloc(alignment , bytes));

    for(std::size_t i = 0 ; i < n_elements ; ++i){
        A[i] = 10 ;
        B[i] = 10 ;
        C[i] = 0 ;
    }

    benchmark::DoNotOptimize(A);
    benchmark::DoNotOptimize(B);
    benchmark::DoNotOptimize(C);

    for(auto _ : state){
        simd_addition_aligned(std::span<int>(A,n_elements),std::span<int>(B,n_elements),std::span<int>(C,n_elements));
        benchmark::ClobberMemory();
    }

    std::free(A);
    std::free(B);
    std::free(C);

    state.SetItemsProcessed(state.iterations()*n_elements);
    state.SetBytesProcessed(state.iterations()*n_elements*3*sizeof(int));

}
BENCHMARK(BM_SIMD_ADD_ALIGNED);

static void BM_SIMD_FMA_ALIGNED(benchmark::State& state){
    // Assign the buffers

    std::size_t n_elements = 1000 ;
    std::size_t bytes = n_elements*sizeof(__m256d);
    std::size_t alignment = alignof(__m256d);

   if (bytes % alignment != 0) {
        bytes += alignment - (bytes % alignment); // std::aligned_alloc requires size to be a multiple of alignment
    }

    double* A = static_cast<double*>(std::aligned_alloc(alignment , bytes));
    double* B = static_cast<double*>(std::aligned_alloc(alignment , bytes));
    double* C = static_cast<double*>(std::aligned_alloc(alignment , bytes));

    for(std::size_t i = 0 ; i < n_elements ; ++i){
        A[i] = 10 ;
        B[i] = 10 ;
        C[i] = 0 ;
    }

    benchmark::DoNotOptimize(A);
    benchmark::DoNotOptimize(B);
    benchmark::DoNotOptimize(C);

    for(auto _ : state){
        simd_fma_aligned(std::span<double>(A,n_elements),std::span<double>(B,n_elements),std::span<double>(C,n_elements));
        benchmark::ClobberMemory();
    }

    std::free(A);
    std::free(B);
    std::free(C);

    state.SetItemsProcessed(state.iterations()*n_elements);
    state.SetBytesProcessed(state.iterations()*n_elements*3*sizeof(double));

}
BENCHMARK(BM_SIMD_FMA_ALIGNED);

static void BM_SIMD_FMA_UNROLLED_AND_ALIGNED(benchmark::State& state){
    // Assign the buffers

    std::size_t n_elements = 1000 ;
    std::size_t bytes = n_elements*sizeof(__m256d);
    std::size_t alignment = alignof(__m256d);

   if (bytes % alignment != 0) {
        bytes += alignment - (bytes % alignment); // std::aligned_alloc requires size to be a multiple of alignment
    }

    double* A = static_cast<double*>(std::aligned_alloc(alignment , bytes));
    double* B = static_cast<double*>(std::aligned_alloc(alignment , bytes));
    double* C = static_cast<double*>(std::aligned_alloc(alignment , bytes));

    for(std::size_t i = 0 ; i < n_elements ; ++i){
        A[i] = 10 ;
        B[i] = 10 ;
        C[i] = 50 ;
    }

    benchmark::DoNotOptimize(A);
    benchmark::DoNotOptimize(B);
    benchmark::DoNotOptimize(C);

    for(auto _ : state){
        simd_fma_unrolled_and_aligned(std::span<double>(A,n_elements),std::span<double>(B,n_elements),std::span<double>(C,n_elements));
        benchmark::ClobberMemory();
    }

    std::free(A);
    std::free(B);
    std::free(C);

    state.SetItemsProcessed(state.iterations()*n_elements);
    state.SetBytesProcessed(state.iterations()*n_elements*3*sizeof(double));

}
BENCHMARK(BM_SIMD_FMA_UNROLLED_AND_ALIGNED);

static void BM_SIMD_ADD_UNROLLED_AND_ALIGNED_THREADED(benchmark::State& state) {

    std::size_t n_elements = state.range(0) ;
    std::size_t bytes = n_elements*sizeof(double);
    std::size_t alignment = alignof(__m256d);

    if (bytes % alignment != 0) {
        bytes += alignment - (bytes % alignment); // std::aligned_alloc requires size to be a multiple of alignment
    }

    double* A = static_cast<double*>(std::aligned_alloc(alignment , bytes)) ;
    double* B = static_cast<double*>(std::aligned_alloc(alignment , bytes)) ;
    double* C = static_cast<double*>(std::aligned_alloc(alignment , bytes)) ;

    for(std::size_t i = 0 ; i < n_elements ; ++i){
        A[i] = 10 ;
        B[i] = 10 ;
        C[i] = 50 ;
    }

    benchmark::DoNotOptimize(A);
    benchmark::DoNotOptimize(B);
    benchmark::DoNotOptimize(C);

    thread_pool pool;
    pool.set_up_buffers(A, B, C, n_elements);

    benchmark::DoNotOptimize(&pool);

    // Pin the main thread to CPU 0 (workers are pinned to CPUs 2, 4, ..., 14)
    cpu_set_t main_set;
    CPU_ZERO(&main_set);
    CPU_SET(0, &main_set);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &main_set);

    for (auto _ : state) {
        pool.call_for_work();
        pool.stall_main();
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * n_elements);
    state.SetBytesProcessed(state.iterations() * n_elements * 3 * sizeof(double));

    std::free(A);
    std::free(B);
    std::free(C);
}
BENCHMARK(BM_SIMD_ADD_UNROLLED_AND_ALIGNED_THREADED)->Arg(8'000)
                                                    ->Arg(100'000)
                                                    ->Arg(10'000'000);

static void BM_SIMD_INNER_PRODUCT_QUAD_UNROLLED_AND_ALIGNED(benchmark::State& state){
    std::size_t elements = state.range(0);
    std::size_t alignment = alignof(__m256d);
    std::size_t size_of_t = sizeof(double);
    std::size_t bytes = size_of_t*elements;

    double* A = static_cast<double*>(std::aligned_alloc(alignment , bytes));
    double* B = static_cast<double*>(std::aligned_alloc(alignment , bytes));

    for(std::size_t i = 0 ; i < elements ; ++i){
        A[i] = 2.0 ;
        B[i] = 3.0 ;
    }

    benchmark::DoNotOptimize(A);
    benchmark::DoNotOptimize(B);

    for(auto _ : state){
        benchmark::DoNotOptimize(simd_inner_product_quad_unrolled_and_aligned(A, B, elements));
        benchmark::ClobberMemory();
    }

    std::free(A);
    std::free(B);

    state.SetItemsProcessed(state.iterations()*elements);
    state.SetBytesProcessed(state.iterations()*2*elements*sizeof(double));

}
BENCHMARK(BM_SIMD_INNER_PRODUCT_QUAD_UNROLLED_AND_ALIGNED)->Arg(1'000)
                                                          ->Arg(10'000)
                                                          ->Arg(100'000)
                                                          ->Arg(1'000'000)
                                                          ->Arg(10'000'000);

static void BM_STL_INNER_PRODUCT(benchmark::State& state){
std::size_t elements = state.range(0);
    std::size_t alignment = alignof(__m256d);
    std::size_t size_of_t = sizeof(double);
    std::size_t bytes = size_of_t*elements;

    double* A = static_cast<double*>(std::aligned_alloc(alignment , bytes));
    double* B = static_cast<double*>(std::aligned_alloc(alignment , bytes));

    for(std::size_t i = 0 ; i < elements ; ++i){
        A[i] = 2.0 ;
        B[i] = 3.0 ;
    }

    std::span<double> A_s(A,elements);
    std::span<double> B_s(B,elements);

    benchmark::DoNotOptimize(A_s.data());
    benchmark::DoNotOptimize(B_s.data());

    for(auto _ : state){
        benchmark::DoNotOptimize(std::inner_product(A_s.begin(),A_s.end(),B_s.begin(),0.0));
        benchmark::ClobberMemory();
    }

    std::free(A);
    std::free(B);

    state.SetItemsProcessed(state.iterations()*elements);
    state.SetBytesProcessed(state.iterations()*2*elements*sizeof(double));
}
BENCHMARK(BM_STL_INNER_PRODUCT)->Arg(1'000)
                               ->Arg(10'000)
                               ->Arg(100'000)
                               ->Arg(1'000'000)
                               ->Arg(10'000'000);
