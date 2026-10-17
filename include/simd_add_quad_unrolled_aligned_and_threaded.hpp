#pragma once 

#include <cstddef>
#include <immintrin.h>

#include "thread_pool.hpp"

// Alligned as we guarantee that all contiguous blocks of memory will be aligned for our data type

// TODO Benchmark with more result buffers (like 4 or 8)
inline auto simd_addition_quad_unrolled_aligned_and_threaded(const double* A, const double* B, double* C,const work_block& block) -> void {
    std::size_t size = block.end ;
    std::size_t i = block.start ;
    
    std::size_t limit = i + ((size - i) & ~15ULL);

    for(; i < limit ; i += 16){ 
        __m256d buffer1 = _mm256_load_pd(&A[i]);
        __m256d buffer2 = _mm256_load_pd(&B[i]);
        __m256d result_buffer1 = _mm256_add_pd(buffer1 , buffer2) ;
        _mm256_store_pd(&C[i] , result_buffer1);

        __m256d buffer3 = _mm256_load_pd(&A[i+4]);
        __m256d buffer4 = _mm256_load_pd(&B[i+4]);
        __m256d result_buffer2 = _mm256_add_pd(buffer3 , buffer4);
        _mm256_store_pd(&C[i+4], result_buffer2);

        __m256d buffer5 = _mm256_load_pd(&A[i+8]);
        __m256d buffer6 = _mm256_load_pd(&B[i+8]);
        __m256d result_buffer3 = _mm256_add_pd(buffer5 , buffer6);
        _mm256_store_pd(&C[i+8], result_buffer3);

        __m256d buffer7 = _mm256_load_pd(&A[i+12]);
        __m256d buffer8 = _mm256_load_pd(&B[i+12]);
        __m256d result_buffer4 = _mm256_add_pd(buffer7 , buffer8);
        _mm256_store_pd(&C[i+12], result_buffer4);
    }

    for(; i < size ; ++i){ // Clear the tail the good old way 
        C[i] = A[i] + B[i] ;
    }

    return ;
}
