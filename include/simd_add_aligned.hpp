#pragma once 

#include <cstddef>
#include <immintrin.h>
#include <span>

// Alligned as we guarantee that all contiguous blocks of memory will be aligned for our data type

// TODO Benchmark with more result buffers (like 4 or 8)
inline auto simd_addition_aligned(const std::span<int> A, const std::span<int> B, std::span<int> C) -> void {
    std::size_t size = C.size() ;
    std::size_t i = 0 ;
    
    for(; i + 8 <= size ; i += 8){ // We load 8 integers move it up by 8 but check for out of bounds
        // Forcibly reinterpret_cast the bits of the integer sequence to a __m256i
        __m256i buffer1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&A[i]));
        __m256i buffer2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&B[i]));

        // Execute the addition on 8 integers in ONE CPU INSTRUCTION INSTEAD OF 8
        __m256i result_buffer = _mm256_add_epi32(buffer1 , buffer2) ;

        // Store the result in our C vector 
        _mm256_store_si256(reinterpret_cast<__m256i*>(&C[i]) , result_buffer);
    }

    for(; i < size ; ++i){ // Clear the tail the good old way 
        C[i] = A[i] + B[i] ;
    }

    return ;
}
