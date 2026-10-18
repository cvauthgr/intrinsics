#pragma once

#include <cstddef>
#include <emmintrin.h>
#include <immintrin.h>

// SIMD function to calculate the inner product of two vectors A,B and store the result in C
// Inner product := A[0]*B[0]+A[1]*B[1]+...+A[i]*B[i] in simd Vs std::inner_product (already vectorized)
inline auto simd_inner_product_quad_unrolled_and_aligned(const double* A, const double* B,const std::size_t number_of_elements) -> double {
    // First assume the memory is aligned and then pre-calculate the loop terminating condition
    std::size_t size = number_of_elements ;
    std::size_t index = 0 ; // Loop index

    // Calculate the cutoff point
    // This is equal to limit = size - (size % 16)
    // Although the latter is 13-20 CPU cycles
    // vs mine which is 1 cycle
    std::size_t limit = index + ((size-index) & ~15ULL);

    // Here we define 4 registers and zero them out
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();
    __m256d acc4 = _mm256_setzero_pd();

    // Define the reducment var
    double reduced_to = 0.0 ;

    // We will be loading 16 doubles per 4 ports in our Zen3 architecture
    for(; index < limit ; index += 16){
        // Load 4 doubles from A and 4 doubles from B
        // and immediately FMA them into acc , we do
        // that 4 times
        acc1 = _mm256_fmadd_pd(_mm256_load_pd(&A[index]) , _mm256_load_pd(&B[index]) , acc1);
        acc2 = _mm256_fmadd_pd(_mm256_load_pd(&A[index+4]) , _mm256_load_pd(&B[index+4]) , acc2);
        acc3 = _mm256_fmadd_pd(_mm256_load_pd(&A[index+8]) , _mm256_load_pd(&B[index+8]) , acc3);
        acc4 = _mm256_fmadd_pd(_mm256_load_pd(&A[index+12]) , _mm256_load_pd(&B[index+12]) , acc4);
    }

    // Crash all four accumulators into one final __256d total_sum
    __m256d total_sum = _mm256_add_pd(_mm256_add_pd(acc1, acc2), _mm256_add_pd(acc3, acc4));

    // Perform the horizontal reduction
    __m128d upper128 = _mm256_extractf128_pd(total_sum, 1) ; // 1 exctracts the upper 128 set of doubles
    __m128d lower128 = _mm256_castpd256_pd128(total_sum) ; // Cast it to 128 bit , losing the upper term we copied
    // Sum them up together into 2 final doubles
    __m128d sum_128 = _mm_add_pd(upper128, lower128);
    // Sum the last doubles
    __m128d upper_64 = _mm_unpackhi_pd(sum_128, sum_128);
    // The previous , does this literally
    // sum_128 = [1.0,2.0] sum_128 = [1.0,2.0]
    // Because we do it on itself we get
    // new_128 = [2.0,2.0]
    __m128d final_128 = _mm_add_pd(sum_128, upper_64);
    // Now in final_128 we have in either spot the result
    reduced_to += _mm_cvtsd_f64(final_128);

    // Calculate what remains and put it in C with
    // a simple plain for-loop
    for(; index < size ; ++index){
        reduced_to += A[index]*B[index] ;
    }

    return reduced_to ;
}
