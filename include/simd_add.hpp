#pragma once 

#include <cstddef>
#include <immintrin.h>
#include <span>

// TODO Benchmark with more result buffers (like 4 or 8)
inline auto simd_addition(const std::span<int> A, const std::span<int> B, std::span<int> C) -> void {
    __m256i buffer1 = { } ; 
    __m256i buffer2 = { } ;
    __m256i result_buffer = { } ;

    std::size_t iterations = C.size()/(sizeof(__m256i)/(sizeof(int)*8)) ; // Calculate if we are going to have a tail
    
    for(std::size_t i = 0 ; i < iterations ; i += 8){ // We load 8 integers move it up by 8 
        // Forcibly reinterpret_cast the bits of the integer sequence to a __m256i
        buffer1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&A[i]));
        buffer2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&B[i]));

        // Execute the addition on 8 integers in ONE CPU INSTRUCTION INSTEAD OF 8
        result_buffer = _mm256_add_epi32(buffer1 , buffer2) ;

        // Store the result in our C vector 
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&C[i]) , result_buffer);
    }

    if(C.size() - iterations != 0){
        for(std::size_t i = C.size() - iterations ; i < C.size() ; ++i){ // Clear the tail the old way 
            C[i] = A[i] + B[i] ;
        }
    }

    return ;
}
