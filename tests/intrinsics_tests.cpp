#include "simd_inner_product_quad_unrolled_and_aligned.hpp"
#include <cstddef>
#include <cstdlib>
#include <gtest/gtest.h>
#include <immintrin.h>
#include <numeric>

TEST(SIMD , inner_product){
    std::size_t elements = 1'000;
    std::size_t alignment = alignof(__m256d);
    std::size_t size_of_t = sizeof(double);
    std::size_t bytes = size_of_t*elements;

    double simd_reduce = 0.0 ;
    double std_reduce = 0.0 ;

    double* A = static_cast<double*>(std::aligned_alloc(alignment , bytes));
    double* B = static_cast<double*>(std::aligned_alloc(alignment , bytes));

    for(std::size_t i = 0 ; i < elements ; ++i){
        A[i] = 2.0 ;
        B[i] = 3.0 ;
    }

    // For the std::inner_product
    std::span<double> A_s(A,elements);
    std::span<double> B_s(B,elements);

    simd_reduce = simd_inner_product_quad_unrolled_and_aligned(A, B, elements);
    std_reduce = std::inner_product(A_s.begin(),A_s.end(),B_s.begin(),0);

    EXPECT_DOUBLE_EQ(simd_reduce, std_reduce) ;

    std::free(A);
    std::free(B);

}
