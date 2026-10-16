#include <benchmark/benchmark.h>

#include "naive_ops.hpp"
#include "simd_ops.hpp"

#include <vector>

static void BM_Naive_Add(benchmark::State& state){
    
    std::vector<int> vec1(250000,1);
    std::vector<int> vec2(250000,2);
    std::vector<int> result_naive(250000,0);
    
    benchmark::DoNotOptimize(vec1.data());
    benchmark::DoNotOptimize(vec2.data()); 
    benchmark::DoNotOptimize(result_naive.data());

    for(auto _ : state){
        naive_addition(vec1, vec2, result_naive);
        benchmark::ClobberMemory();
    }
    
    state.SetItemsProcessed(state.iterations()*result_naive.size());
    state.SetBytesProcessed(state.iterations()*result_naive.size()*3);

}
BENCHMARK(BM_Naive_Add);

static void BM_SIMD_Add(benchmark::State& state){
    
    std::vector<int> vec1(250000,1);
    std::vector<int> vec2(250000,2);
    std::vector<int> result_simd(250000,0);
    
    benchmark::DoNotOptimize(vec1.data());
    benchmark::DoNotOptimize(vec2.data()); 
    benchmark::DoNotOptimize(result_simd.data());

    for(auto _ : state){
        simd_addition(vec1, vec2, result_simd);
        benchmark::ClobberMemory();
    }
    
    state.SetItemsProcessed(state.iterations()*result_simd.size());
    state.SetBytesProcessed(state.iterations()*result_simd.size()*3);

}
BENCHMARK(BM_SIMD_Add);
