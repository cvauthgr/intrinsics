#pragma once 

#include <cstddef>
#include <immintrin.h>
#include <span>

#include "thread_pool.hpp"

// C[i] = A[i] + B[i] over [block.start, block.end), unrolled x2 (8 doubles per iteration).
// Precondition: &A[block.start], &B[block.start] and &C[block.start] are 32-byte aligned.
inline auto simd_addition_unrolled_and_aligned_threaded(const std::span<double> A, const std::span<double> B, std::span<double> C,const work_block& block) -> void {
    std::size_t size = block.end ;
    std::size_t i = block.start ;
    
    for(; i + 8 <= size ; i += 8){ 
        __m256d buffer1 = _mm256_load_pd(&A[i]);
        __m256d buffer2 = _mm256_load_pd(&B[i]);
        __m256d result_buffer1 = _mm256_add_pd(buffer1 , buffer2) ;
        _mm256_store_pd(&C[i] , result_buffer1);

        __m256d buffer3 = _mm256_load_pd(&A[i+4]);
        __m256d buffer4 = _mm256_load_pd(&B[i+4]);
        __m256d result_buffer2 = _mm256_add_pd(buffer3 , buffer4);
        _mm256_store_pd(&C[i+4], result_buffer2);
    }

    for(; i < size ; ++i){ // Scalar tail
        C[i] = A[i] + B[i] ;
    }

    return ;
}
