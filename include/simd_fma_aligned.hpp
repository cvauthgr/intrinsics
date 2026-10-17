#pragma once 

#include <cstddef>
#include <immintrin.h>

#include <span>

// Aligned for doubles but not unrolled

inline auto simd_fma_aligned(const std::span<double> A, const std::span<double> B, std::span<double> C) -> void {
    std::size_t size = C.size() ;
    std::size_t i = 0 ;

    for(; i + 4 <= size ; i += 4){ // i + 4 instead of i to avoid trying to access elements out of bounds 
        __m256d buffer1 = _mm256_load_pd(&A[i]) ;
        __m256d buffer2 = _mm256_load_pd(&B[i]) ;
        __m256d result_buffer = _mm256_load_pd(&C[i]) ;

        result_buffer = _mm256_fmadd_pd(buffer1, buffer2, result_buffer) ;

        _mm256_store_pd(&C[i], result_buffer);
    }

    // Finish off anything that remains 
    for(; i < size ; ++i){
        C[i] += A[i] * B[i] ;
    }
    
    return ;
}

