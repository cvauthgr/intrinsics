#pragma once 

#include <cstddef>
#include <immintrin.h>

#include <span>

// AVX2/FMA C[i] += A[i] * B[i] using unaligned loads/stores (no alignment precondition).
inline auto simd_fma(const std::span<double> A, const std::span<double> B, std::span<double> C) -> void {
    std::size_t size = C.size() ;
    std::size_t i = 0 ;

    // 4 doubles per iteration; i + 4 <= size keeps the last vector load in bounds
    for(; i + 4 <= size ; i += 4){
        __m256d buffer1 = _mm256_loadu_pd(&A[i]) ;
        __m256d buffer2 = _mm256_loadu_pd(&B[i]) ;
        __m256d result_buffer = _mm256_loadu_pd(&C[i]) ;

        result_buffer = _mm256_fmadd_pd(buffer1, buffer2, result_buffer) ;

        _mm256_storeu_pd(&C[i], result_buffer);
    }

    // Scalar tail
    for(; i < size ; ++i){
        C[i] += A[i] * B[i] ;
    }
    
    return ;
}

