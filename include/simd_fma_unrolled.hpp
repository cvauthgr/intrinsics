#pragma once

#include <cstddef>
#include <immintrin.h>

#include <span>

inline auto simd_fma_unrolled(const std::span<double> A, const std::span<double> B, std::span<double> C) -> void {
    std::size_t size = C.size() ;
    std::size_t i = 0 ;

    // For unrolling we will use both FMA in our core 
    for(; i + 8 <= size ; i += 8){ // i + 8 instead of i + 4 because we grad 8 doubles at a SINGULAR CPU INSTRUCTION
        __m256d buffer1 = _mm256_loadu_pd(&A[i]) ;
        __m256d buffer2 = _mm256_loadu_pd(&B[i]) ;
        __m256d result_buffer1 = _mm256_loadu_pd(&C[i]) ;

        __m256d buffer3 = _mm256_loadu_pd(&A[i+4]);
        __m256d buffer4 = _mm256_loadu_pd(&B[i+4]);
        __m256d result_buffer2 = _mm256_loadu_pd(&C[i+4]);

        result_buffer1 = _mm256_fmadd_pd(buffer1, buffer2, result_buffer1) ;
        result_buffer2 = _mm256_fmadd_pd(buffer3, buffer4, result_buffer2) ;

        _mm256_storeu_pd(&C[i], result_buffer1);
        _mm256_storeu_pd(&C[i+4], result_buffer2);
    }

    // Finish off anything that remains 
    for(; i < size ; ++i){
        C[i] += A[i] * B[i] ;
    }
    
    return ;
}
