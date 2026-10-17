#pragma once

#include <cstddef>
#include <immintrin.h>
#include <span>
#include "thread_pool.hpp"

inline auto simd_fma_unrolled_and_aligned_threaded(const std::span<double> A, const std::span<double> B, std::span<double> C,const work_block& per_thread_block) -> void {
    std::size_t size = per_thread_block.end ;
    std::size_t i = per_thread_block.start ;

    // For unrolling we will use both FMA in our core 
    for(; i + 8 <= size ; i += 8){
        __m256d buffer1 = _mm256_load_pd(&A[i]) ;
        __m256d buffer2 = _mm256_load_pd(&B[i]) ;
        __m256d result_buffer1 = _mm256_load_pd(&C[i]) ;

        result_buffer1 = _mm256_fmadd_pd(buffer1, buffer2, result_buffer1) ; 
        _mm256_store_pd(&C[i], result_buffer1) ;

        __m256d buffer3 = _mm256_load_pd(&A[i+4]);
        __m256d buffer4 = _mm256_load_pd(&B[i+4]);
        __m256d result_buffer2 = _mm256_load_pd(&C[i+4]);

        result_buffer2 = _mm256_fmadd_pd(buffer3, buffer4, result_buffer2) ;
        _mm256_store_pd(&C[i+4], result_buffer2) ;
    }

    // Finish off anything that remains 
    for(; i < size ; ++i){
        C[i] += A[i] * B[i] ;
    }
    
    return ;
}

