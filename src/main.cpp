#include "naive_ops.hpp"
#include "simd_ops.hpp"

#include <vector>
#include <cassert>

int main(){

    std::vector<int> vec1(250000,1);
    std::vector<int> vec2(250000,2);
    std::vector<int> result_naive(250000,0);
    std::vector<int> result_simd(250000,0); 
    
    naive_addition(vec1 , vec2, result_naive) ;
    simd_addition(vec1 , vec2 , result_simd) ;

    assert(result_naive == result_simd);

    return 0;
}
