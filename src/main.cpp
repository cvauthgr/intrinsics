#include "naive_add.hpp"
#include "simd_add.hpp"
#include "naive_fma.hpp"
#include "simd_fma.hpp"

#include <vector>
#include <cassert>

int main(){

    std::vector<int> vec1(250000,1);
    std::vector<int> vec2(250000,2);
    std::vector<int> result_naive(250000,0);
    std::vector<int> result_simd(250000,0); 
    
    naive_addition(vec1 , vec2, result_naive) ;
    simd_addition(vec1 , vec2 , result_simd) ;

    for(std::size_t i = 0 ; i < result_simd.size() ; ++i)
        assert(result_naive[i] == result_simd[i]);

    // Now for our SIMD FMA we are going to do (A*B)+C
    std::vector<double> A(250000,10);
    std::vector<double> B(250000,10);
    std::vector<double> C_naive(250000,50);
    std::vector<double> C_simd(250000,50);
    
    naive_fma(A,B,C_naive);
    simd_fma(A,B,C_simd);

    for(std::size_t i = 0 ; i < result_simd.size() ; ++i)
        assert(C_simd[i] == C_naive[i]); 

    return 0;
}
