#pragma once

#include <cstddef>
#include <immintrin.h>

#include <span>

// AVX2/FMA C[i] += A[i] * B[i], aligned and unrolled x2.
// Precondition: A, B and C must be 32-byte aligned.
inline auto simd_fma_unrolled_and_aligned(const std::span<double> A, const std::span<double> B, std::span<double> C) -> void {
    std::size_t size = C.size() ;
    std::size_t i = 0 ;

    // Unrolled x2: two 4-double vectors (8 doubles) per iteration
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

    // Scalar tail
    for(; i < size ; ++i){
        C[i] += A[i] * B[i] ;
    }
    
    return ;
}

