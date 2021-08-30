///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief CUDA utilities for timing, data management and error-checking
///

#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <cuda.h>
#include <cublas_v2.h>

typedef struct gpu_timer {
  cudaEvent_t start;
  cudaEvent_t stop;
  float ms = 0;
} * gpu_timer_p;

gpu_timer_p gpu_timer_init();
void gpu_timer_start(gpu_timer_p timer, cudaStream_t stream);
void gpu_timer_stop(gpu_timer_p timer, cudaStream_t stream);
float gpu_timer_get(gpu_timer_p timer);

/// Memory layout struct for matrices
enum mem_layout { ROW_MAJOR = 0, COL_MAJOR };

const char *print_mem(mem_layout mem);

/// Print name of loc for transfers
const char *print_loc(short loc);

/// Check if there are CUDA errors on the stack
void cudaCheckErrors();

/// Allocate 'count' bytes of CUDA device memory (+errorcheck)
//void* gpu_malloc(long long count);
/// Allocate 'count' bytes of CUDA host pinned memory (+errorcheck)
//void* pin_malloc(long long count);
/// Generalized malloc in loc 
void vec_alloc(void ** ptr, long long N_bytes, int loc);

/// Free the CUDA device  memory pointed by 'gpuptr' (+errorcheck)
//void gpu_free(void* gpuptr);
//void pin_free(void* gpuptr);
/// Generalized free in loc 
void vec_free(void ** ptr, int loc);

/// Initialize length random floats/doubles in dev_prt using seed via cuRAND
void Sinit_cuRAND(float * dev_ptr, long long length, int seed);
void Dinit_cuRAND(double * dev_ptr, long long length, int seed);

#endif
