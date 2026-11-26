#pragma once 

#include <span>

// Scalar baseline for C[i] += A[i] * B[i].
// Auto-vectorization is disabled for this function (GCC-specific pragma; Clang ignores it).
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize")
inline auto naive_fma(const std::span<double> A, const std::span<double> B, std::span<double> C) -> void {
    for(std::size_t i = 0 ; i < C.size() ; ++i){
        C[i] += A[i]*B[i] ;
    }
    return ;
}
#pragma GCC pop_options
