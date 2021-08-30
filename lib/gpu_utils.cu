#include <float.h>
#include <cstdio>
#include "cpu_utils.hpp"
#include "gpu_utils.hpp"
#include <curand.h>
#include <curand_kernel.h>

gpu_timer_p gpu_timer_init() {
	gpu_timer_p timer = (gpu_timer_p)malloc(sizeof(struct gpu_timer));
	cudaEventCreate(&timer->start);
	cudaEventCreate(&timer->stop);
	return timer;
}

void gpu_timer_start(gpu_timer_p timer, cudaStream_t stream) { cudaEventRecord(timer->start, stream); }

void gpu_timer_stop(gpu_timer_p timer, cudaStream_t stream) { cudaEventRecord(timer->stop, stream); }

float gpu_timer_get(gpu_timer_p timer) {
	cudaEventSynchronize(timer->stop);
	cudaEventElapsedTime(&timer->ms, timer->start, timer->stop);
	return timer->ms;
}

const char *print_mem(mem_layout mem) {
	if (mem == ROW_MAJOR) return "Row major";
	else if (mem == COL_MAJOR) return "Col major";
	else return "ERROR";
}

const char *print_loc(short loc) {
	int dev_count;
	cudaGetDeviceCount(&dev_count);
	if (loc == -2)  return "Host"; 
	else if (loc == -1 || loc == -3)  return "Pinned Host";
	else if (loc < dev_count) return "Device";
	else return "ERROR";
}

void cudaCheckErrors() {
	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess) printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess) printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

/// Allocate 'count' bytes of CUDA device memory (+errorcheck)
void *gpu_malloc(long long count) {
	void *ret;
	massert(cudaMalloc(&ret, count) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
	return ret;
}

/// Allocate 'count' bytes of CUDA host pinned memory (+errorcheck)
void *pin_malloc(long long count) {
	void *ret;
	massert(cudaMallocHost(&ret, count) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
	return ret;
}

void vec_alloc(void ** ptr, long long N_bytes, int loc){
	int count = 666;
	cudaGetDeviceCount(&count);

	if (-2 == loc) {
    		//fprintf(stderr, "Allocating %lld bytes to host...\n", N_bytes);
		*ptr = (void*) malloc(N_bytes);
	}
	else if (-1 == loc || -3 == loc) {
		//fprintf(stderr, "Allocating %lld bytes to pinned host...\n", N_bytes);
		*ptr = pin_malloc(N_bytes);

	} else if (loc >= count || loc < 0) error("vec_init: Invalid device");
	else {
		//fprintf(stderr, "Allocating %lld bytes to device(%d)...\n", N_bytes, loc);
		cudaSetDevice(loc);
		*ptr = gpu_malloc(N_bytes);
	}
	cudaCheckErrors();
}

/// Free the CUDA device  memory pointed by 'gpuptr' (+errorcheck)
void gpu_free(void *gpuptr) {
	massert(cudaFree(gpuptr) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
}

/// Free the CUDA pinned memory pointed by 'gpuptr' (+errorcheck)
void pin_free(void *gpuptr) {
	massert(cudaFreeHost(gpuptr) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
}

void vec_free(void ** ptr, int loc){
	int count = 666;
	cudaGetDeviceCount(&count);

	if (-2 == loc) free(*ptr);
	else if (-1 == loc || -3 == loc) pin_free(*ptr);
	else if (loc >= count || loc < 0) error("vec_free: Invalid device");
	else {
		cudaSetDevice(loc);
		gpu_free(*ptr);
	}
	cudaCheckErrors();
}

void Sinit_cuRAND(float * dev_ptr, long long length, int seed){
    	curandGenerator_t gen;
	/* Create pseudo-random number generator */
	massert(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
    
	/* Set seed */
	massert(curandSetPseudoRandomGeneratorSeed(gen, seed) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));

	/* Generate length floats on device */
	massert(curandGenerateUniform(gen, dev_ptr, length) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
}

void Dinit_cuRAND(double * dev_ptr, long long length, int seed){
    	curandGenerator_t gen;
	/* Create pseudo-random number generator */
	massert(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
    
	/* Set seed */
	massert(curandSetPseudoRandomGeneratorSeed(gen, seed) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));

	/* Generate length doubles on device */
	massert(curandGenerateUniformDouble(gen, dev_ptr, length) == cudaSuccess,
          cudaGetErrorString(cudaGetLastError()));
}
