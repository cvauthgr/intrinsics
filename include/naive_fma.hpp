#pragma once 

#include <span>

#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize")
inline auto naive_fma(const std::span<double> A, const std::span<double> B, std::span<double> C) -> void {
    // First we are going to do the elementwise A[i]*B[i] and add this to every C[i]
    for(std::size_t i = 0 ; i < C.size() ; ++i){
        C[i] += A[i]*B[i] ;
    }
    return ;
}
#pragma GCC pop_options
