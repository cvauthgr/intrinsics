#include "simd_inner_product_quad_unrolled_and_aligned.hpp"
#include "naive_fma.hpp"
#include "simd_fma.hpp"
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <gtest/gtest.h>
#include <immintrin.h>
#include <numeric>
#include <span>
#include <vector>

// std::aligned_alloc requires the size to be a multiple of the alignment
static auto aligned_doubles(std::size_t elements) -> double* {
    constexpr std::size_t alignment = alignof(__m256d);
    std::size_t bytes = elements*sizeof(double);
    bytes = (bytes + alignment - 1) / alignment * alignment;
    return static_cast<double*>(std::aligned_alloc(alignment , bytes));
}

TEST(SIMD , inner_product){
    std::size_t elements = 1'000;

    double simd_reduce = 0.0 ;
    double std_reduce = 0.0 ;

    double* A = aligned_doubles(elements);
    double* B = aligned_doubles(elements);

    for(std::size_t i = 0 ; i < elements ; ++i){
        A[i] = 2.0 ;
        B[i] = 3.0 ;
    }

    std::span<double> A_s(A,elements);
    std::span<double> B_s(B,elements);

    simd_reduce = simd_inner_product_quad_unrolled_and_aligned(A, B, elements);
    std_reduce = std::inner_product(A_s.begin(),A_s.end(),B_s.begin(),0.0);

    EXPECT_DOUBLE_EQ(simd_reduce, std_reduce) ;

    std::free(A);
    std::free(B);

}

// Size not a multiple of 16 so the scalar tail runs, with non-integer values.
// Summation order differs from std::inner_product, so compare with a relative tolerance.
TEST(SIMD , inner_product_with_tail){
    std::size_t elements = 1'003;

    double* A = aligned_doubles(elements);
    double* B = aligned_doubles(elements);

    for(std::size_t i = 0 ; i < elements ; ++i){
        A[i] = 0.1 * static_cast<double>(i) ;
        B[i] = 1.0 / static_cast<double>(i + 1) ;
    }

    std::span<double> A_s(A,elements);
    std::span<double> B_s(B,elements);

    double simd_reduce = simd_inner_product_quad_unrolled_and_aligned(A, B, elements);
    double std_reduce = std::inner_product(A_s.begin(),A_s.end(),B_s.begin(),0.0);

    EXPECT_NEAR(simd_reduce, std_reduce, 1e-12 * std::abs(std_reduce)) ;

    std::free(A);
    std::free(B);
}

// Size not a multiple of 4 so the scalar tail of simd_fma runs.
// Each element is a single fused/unfused multiply-add, so the values below are exact either way.
TEST(SIMD , fma_with_tail){
    std::size_t elements = 1'003;

    std::vector<double> A(elements);
    std::vector<double> B(elements);
    std::vector<double> C_simd(elements, 1.0);
    std::vector<double> C_naive(elements, 1.0);

    for(std::size_t i = 0 ; i < elements ; ++i){
        A[i] = static_cast<double>(i) ;
        B[i] = 2.0 ;
    }

    simd_fma(A, B, C_simd);
    naive_fma(A, B, C_naive);

    for(std::size_t i = 0 ; i < elements ; ++i){
        EXPECT_DOUBLE_EQ(C_simd[i], C_naive[i]) << "at index " << i ;
    }
}
