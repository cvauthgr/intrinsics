#pragma once 

#include <cstddef>
#include <pthread.h>
#include <sched.h>

// Define the work as blocks to avoid of by one
// errors and out of bounds access
struct alignas(64) work_block{
    std::size_t start = 0 ;
    std::size_t end = 0 ;
};

#include "simd_add_quad_unrolled_aligned_and_threaded.hpp"
#include <atomic>
#include <emmintrin.h>
#include <vector>
#include <thread>
#include <cassert>

#include <immintrin.h>

class thread_pool{
    // Private members and member functions
    // Here I will create the thread vector 
    // the atomic bool and my custom atomic latch
    private :
        // Don't forget to reserve for it in the constructor
        std::vector<std::thread> workers = { } ;
        // The variable that the workers will be waiting on
        std::atomic_bool spin_mark = true ;
        // Custom "latch" implementation
        struct alignas(64) latch{
            // Suround our int in 60 dead bytes 
            // to prevent cache line invalidation
            std::atomic_int state = {0} ;
        };
        // Store the latches in an array
        // In my case I have 16 threads so 15 
        // without the main one 
        std::array<latch,7> latches = { } ;
        // Every latch is zero , when a thread 
        // finished it sets his to 1 to mark it 
    
        auto kill_all() -> void {
            for(auto& latch : latches){
                // Initiate the kill protocol
                latch.state.store(2 , std::memory_order_release) ;       
            }
            // By setting 2 at every latch we break out
            // of the infinite while loop and kill the thread 
            return ;
        }

        // Lets do the FMA with the unrolled loop and 
        // the alligned storage , initially I thought
        // about using a shared pointer but each core 
        // just needs a slice of the data 
        double* operants_of_A = { } ;
        double* operants_of_B = { } ;
        // Our result buffer here we will add A[i]*B[i]
        double* operants_of_C = { } ;
        // Also save the number of operants
        std::size_t operants_count = { } ;

        // Remember the number of threads as we are 
        // going to precalculate the amount of work each has to
        // do in an array 
        std::size_t thread_count = 7 ;

        // Amount of work per thread 
        std::array<::work_block, 7> work = { } ;

        // Precalculate the work 
        auto calculate_work_per_worker() noexcept -> void {
            std::size_t raw_work = operants_count / thread_count ;
            std::size_t work_per_thread = raw_work & ~15ULL ; // Align the memory 
            std::size_t remaining_work = operants_count - (thread_count*work_per_thread) ;
            assert((work_per_thread*thread_count)+remaining_work == operants_count); 

            std::size_t i = 0 ;

            for(::work_block& block : work){
                block.start = i*work_per_thread ;
                block.end = (i*work_per_thread)+work_per_thread;
                i++ ;
            }  

            // Assign the remaining_work
            work.back().end += remaining_work ;

            return ; 
        }

    public :
        // Create the vector and make all threads wait 
        // for the spin_mark symbol to be set to false
        explicit thread_pool() noexcept {
            workers.reserve(thread_count) ;

        // Here I will implement the worker function
        auto worker_function = [&](std::size_t worker_index) -> void {
            while(true){
                while(latches[worker_index].state.load(std::memory_order_acquire) == 0){
                    _mm_pause();
                }
                
                if(latches[worker_index].state.load(std::memory_order_acquire) == 2){
                    break ;
                }
                simd_addition_quad_unrolled_aligned_and_threaded(operants_of_A,operants_of_B,operants_of_C,work[worker_index]);

                latches[worker_index].state.store(0 , std::memory_order_release) ;
            }

            return ;
        }; 

            // Make all the threads wait 
            for(std::size_t i = 0 ; i < thread_count ; ++i){
                workers.emplace_back(worker_function,i);
                // Here I will pin the real cores
                // so the scheduler doesn't move them 
                // around messing the data topology
                cpu_set_t pin_to_core ;
                CPU_ZERO(&pin_to_core) ;
                CPU_SET((i+1)*2,&pin_to_core);
                pthread_setaffinity_np(workers.back().native_handle(),sizeof(cpu_set_t),&pin_to_core);
            }


        }

        // Kill all the spinning threads and join them
        // so std::terminate() is not called
        ~thread_pool(){
            this->kill_all();

            for(auto& worker : workers){
                worker.join();
            }
        }

        // Set the buffers 
        auto set_up_buffers(double* A , double* B , double* C , std::size_t number_of_elements) noexcept -> void 
        {
            // Find out anything you need about the buffers
            operants_of_A = A ;
            operants_of_B = B ;
            operants_of_C = C ;
            operants_count = number_of_elements ;

            // Automatically run it here , avoiding 
            // calculating on none existing buffers 
            calculate_work_per_worker() ;

            return ;
        }
        
        auto call_for_work() noexcept -> void {
            for(auto& latch : latches){
                // Set all threads to be freed and do work
                latch.state.store(1,std::memory_order_release);    
            }

            return ;
        }

        auto stall_main() noexcept -> void {
            // Make main wait on all the cores to finish
            // look for the 0 flag to be set 
            for(const auto& core_latch : latches){
                while(core_latch.state.load(std::memory_order_acquire) == 1){
                    _mm_pause();
                }
            }

            return ;
        }

};
