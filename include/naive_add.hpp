#pragma once 

#include <span>
#include <cstddef>

// Scalar baseline for C[i] = A[i] + B[i].
// Auto-vectorization is disabled for this function so the comparison against the
// intrinsic versions is meaningful. The pragma is GCC-specific; Clang ignores it.

#pragma GCC push_options
#pragma GCC optimize ("no-tree-vectorize")
inline auto naive_addition(const std::span<int> A , const std::span<int> B , std::span<int> C) -> void {
    for(std::size_t i = 0 ; i < C.size() ; ++ i){
        C[i] = A[i] + B[i] ;
    }
}
#pragma GCC pop_options 


