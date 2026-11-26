#pragma once 

#include <cstddef>
#include <pthread.h>
#include <sched.h>

// Half-open range [start, end) of elements assigned to one worker
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

// Fixed pool of 7 spinning workers that compute C = A + B in parallel,
// each on its own slice of the buffers.
class thread_pool{
    private :
        std::vector<std::thread> workers = { } ;
        std::atomic_bool spin_mark = true ;
        // One latch per worker, each on its own 64-byte cache line so that
        // workers spinning on their latch do not cause false sharing.
        // Protocol: 0 = idle, 1 = work requested, 2 = shut down.
        // The main thread sets 1 (or 2); the worker sets 0 back when its slice is done.
        struct alignas(64) latch{
            std::atomic_int state = {0} ;
        };
        std::array<latch,7> latches = { } ;

        // Sets every latch to 2 so each worker leaves its loop and returns
        auto kill_all() -> void {
            for(auto& latch : latches){
                latch.state.store(2 , std::memory_order_release) ;
            }
            return ;
        }

        // Buffers for the current job (C = A + B); each worker only touches its own slice
        double* operants_of_A = { } ;
        double* operants_of_B = { } ;
        double* operants_of_C = { } ;
        std::size_t operants_count = { } ;

        std::size_t thread_count = 7 ;

        // Precomputed slice for each worker
        std::array<::work_block, 7> work = { } ;

        // Splits [0, operants_count) into thread_count slices; the last slice takes the remainder
        auto calculate_work_per_worker() noexcept -> void {
            std::size_t raw_work = operants_count / thread_count ;
            // Round down to a multiple of 16 elements, so every slice starts on a
            // 128-byte boundary (keeps the 32-byte alignment of the base pointers)
            std::size_t work_per_thread = raw_work & ~15ULL ;
            std::size_t remaining_work = operants_count - (thread_count*work_per_thread) ;
            assert((work_per_thread*thread_count)+remaining_work == operants_count); 

            std::size_t i = 0 ;

            for(::work_block& block : work){
                block.start = i*work_per_thread ;
                block.end = (i*work_per_thread)+work_per_thread;
                i++ ;
            }  

            work.back().end += remaining_work ;

            return ;
        }

    public :
        // Starts the workers; each one spins on its latch until call_for_work()
        explicit thread_pool() noexcept {
            workers.reserve(thread_count) ;

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

            for(std::size_t i = 0 ; i < thread_count ; ++i){
                workers.emplace_back(worker_function,i);
                // Pin worker i to logical CPU 2*(i+1), i.e. CPUs 2, 4, ..., 14
                // (CPU 0 is left for the main thread)
                cpu_set_t pin_to_core ;
                CPU_ZERO(&pin_to_core) ;
                CPU_SET((i+1)*2,&pin_to_core);
                pthread_setaffinity_np(workers.back().native_handle(), 
                                       sizeof(cpu_set_t),&pin_to_core);
            }


        }

        // Stops all workers and joins them (destroying a joinable std::thread calls std::terminate)
        ~thread_pool(){
            this->kill_all();

            for(auto& worker : workers){
                worker.join();
            }
        }

        // Sets the buffers for the next job and recomputes the per-worker slices.
        // Must not be called while workers are running.
        auto set_up_buffers(double* A , double* B , double* C , std::size_t number_of_elements) noexcept -> void
        {
            operants_of_A = A ;
            operants_of_B = B ;
            operants_of_C = C ;
            operants_count = number_of_elements ;

            calculate_work_per_worker() ;

            return ;
        }
        
        // Releases every worker to process its slice
        auto call_for_work() noexcept -> void {
            for(auto& latch : latches){
                latch.state.store(1,std::memory_order_release);
            }

            return ;
        }

        // Spins until every worker has set its latch back to 0
        auto stall_main() noexcept -> void {
            for(const auto& core_latch : latches){
                while(core_latch.state.load(std::memory_order_acquire) == 1){
                    _mm_pause();
                }
            }

            return ;
        }

};
