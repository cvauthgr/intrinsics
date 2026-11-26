#pragma once 

#include <cstddef>
#include <immintrin.h>
#include <span>

// AVX2 C[i] = A[i] + B[i] using unaligned loads/stores (no alignment precondition).
inline auto simd_addition(const std::span<int> A, const std::span<int> B, std::span<int> C) -> void {
    std::size_t size = C.size() ;
    std::size_t i = 0 ;

    // 8 x int32 (256 bits) per iteration; i + 8 <= size keeps the last vector load in bounds
    for(; i + 8 <= size ; i += 8){
        __m256i buffer1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&A[i]));
        __m256i buffer2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&B[i]));

        // 8 int32 additions in a single vpaddd
        __m256i result_buffer = _mm256_add_epi32(buffer1 , buffer2) ;

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&C[i]) , result_buffer);
    }

    for(; i < size ; ++i){ // Scalar tail
            C[i] = A[i] + B[i] ;
    }

    return ;
}
