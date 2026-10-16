#pragma once 

#include <span>
#include <cstddef>

// Here we define the function for the naive_addition which is not vectorised by the vectorised
// by the compiler through the pragma flags , the spans are pointer + size of the underlying 
// contiguous memory containers (array or vector) 

#pragma GCC push_options
#pragma GCC optimize ("no-tree-vectorize")
inline auto naive_addition(const std::span<int> A , const std::span<int> B , std::span<int> C) -> void {
    for(std::size_t i = 0 ; i < C.size() ; ++ i){
        C[i] = A[i] + B[i] ;
    }
}
#pragma GCC pop_options 


