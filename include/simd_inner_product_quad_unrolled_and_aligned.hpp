#pragma once

#include <cstddef>
#include <emmintrin.h>
#include <immintrin.h>

// Returns the inner product A[0]*B[0] + A[1]*B[1] + ... + A[n-1]*B[n-1].
// Precondition: A and B must be 32-byte aligned.
// Note: the summation order differs from a sequential loop, so the result can differ
// from std::inner_product in the last bits (floating-point addition is not associative).
inline auto simd_inner_product_quad_unrolled_and_aligned(const double* A, const double* B,const std::size_t number_of_elements) -> double {
    std::size_t size = number_of_elements ;
    std::size_t index = 0 ;

    // Largest multiple of 16 not exceeding size, i.e. size - (size % 16)
    std::size_t limit = index + ((size-index) & ~15ULL);

    // Four independent accumulators, so consecutive FMAs do not all wait on
    // the result of the previous one (FMA has multi-cycle latency)
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();
    __m256d acc4 = _mm256_setzero_pd();

    double reduced_to = 0.0 ;

    // 16 doubles per iteration: each accumulator takes one 4-double FMA
    for(; index < limit ; index += 16){
        acc1 = _mm256_fmadd_pd(_mm256_load_pd(&A[index]) , _mm256_load_pd(&B[index]) , acc1);
        acc2 = _mm256_fmadd_pd(_mm256_load_pd(&A[index+4]) , _mm256_load_pd(&B[index+4]) , acc2);
        acc3 = _mm256_fmadd_pd(_mm256_load_pd(&A[index+8]) , _mm256_load_pd(&B[index+8]) , acc3);
        acc4 = _mm256_fmadd_pd(_mm256_load_pd(&A[index+12]) , _mm256_load_pd(&B[index+12]) , acc4);
    }

    // Combine the four accumulators into one vector of 4 partial sums
    __m256d total_sum = _mm256_add_pd(_mm256_add_pd(acc1, acc2), _mm256_add_pd(acc3, acc4));

    // Horizontal reduction: 4 lanes -> 2 lanes -> 1 lane
    __m128d upper128 = _mm256_extractf128_pd(total_sum, 1) ; // high 128 bits (lanes 2, 3)
    __m128d lower128 = _mm256_castpd256_pd128(total_sum) ;   // low 128 bits (lanes 0, 1), no instruction emitted
    __m128d sum_128 = _mm_add_pd(upper128, lower128);         // [s0, s1]
    __m128d upper_64 = _mm_unpackhi_pd(sum_128, sum_128);     // [s1, s1]
    __m128d final_128 = _mm_add_pd(sum_128, upper_64);        // lane 0 = s0 + s1
    reduced_to += _mm_cvtsd_f64(final_128);

    // Scalar tail
    for(; index < size ; ++index){
        reduced_to += A[index]*B[index] ;
    }

    return reduced_to ;
}
