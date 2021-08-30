///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief A transfer micro-benchmark from->to for a) contiguous transfers, b) non-cont square transfers, c) full bidirectional overlapped transfers 
///

#include <unistd.h>
#include <cassert>
#include <cuda.h>
#include "cublas_v2.h"
#include "cpu_utils.hpp"
#include "gpu_utils.hpp"

char* check_benchmark( short to, short from, long long bytes_low, long long bytes_upper, int classes, int samples){
	char *filename = (char *) malloc(256* sizeof(char));
	sprintf(filename, "%s/BenchOutputs/%s/cublasSet_Get/EMLR_to-%d_from-%d_MinB-%ld_UpperStartB-%ld_classes-%d_samples-%d.log", PROJECTDIR, MACHINE, to, from, bytes_low, bytes_upper, classes, samples);

	FILE* fp = fopen(filename,"r");
	if (!fp) { 
		fp = fopen(filename,"w+");
		if (!fp) error("report_results: LogFile failed to open");
		else warning("Generating Logfile...");
		fclose(fp);
	}
	else {
		fprintf(stderr,"Transfer benchmark@%s %d->%d found: (%ld,%ld) for %d classes with %d samples \n", MACHINE, from, to, bytes_low, bytes_upper, classes, samples);
		fclose(fp);	
		exit(1); 
	}
	return filename;		  	
}

void report_run(char* filename, long long bytes,  double t_normal, double t_bid, double t_sq_chunk){

	FILE* fp = fopen(filename,"a");
	if (!fp) error("report_run: LogFile failed to open");
   	fprintf(fp,"%lld, %e,%e,%e\n", bytes, t_normal, t_bid, t_sq_chunk);
        fclose(fp); 
}

/// Split the benchmark total samples into logarithmic classes ->
/// 1) Uniformly distribute the total samples between the classes.
/// 2) Each class has a range [Cmin,Cmax].
/// 3) Class 0 initialized with Cmax = bytes_upper.
/// 4) For all classes Cmin[i] = Cmax[i]/2, Cmax[i] = Cmin[i-1]
long long * distribute_in_classes(int classes, long long bytes_low, long long bytes_upper, int samples)
{
	long long *bench_bytes = (long long*) malloc(samples*sizeof(long long));
	long long bytes = bytes_upper, byte_step;
	int class_samples = samples/classes;
	for (int i = 0; i < classes; i++){
		byte_step = bytes/2/class_samples; 
		for (int j = 0; j < class_samples; j++){
			bench_bytes[i* class_samples + j] = bytes; 
			bytes-= byte_step;
			if ( bytes < bytes_low) bytes = bytes_low; 
		}
	}
	byte_step = bytes/2/class_samples;
	for (int i = 0; i < samples%classes; i++){
		bench_bytes[class_samples*classes + i] = bytes; 
		bytes-= byte_step;
		if ( bytes < bytes_low) bytes = bytes_low; 
	}
	return 	bench_bytes;
}


int main(const int argc, const char *argv[]) {

  	int ctr = 1, samples, dev_id, dev_count;

	char machine[256], *filename;
	short from, to, classes; 
	long long byte_low, byte_upper, bytes;

	switch (argc) {
	case (8):
		sprintf(machine , "%s", argv[ctr++]);
		to = atoi(argv[ctr++]);
		from = atoi(argv[ctr++]);
		byte_low = atol(argv[ctr++]);
		byte_upper = atol(argv[ctr++]);
		classes = atoi(argv[ctr++]);
		samples = atoi(argv[ctr++]);
		break;
	default:
		error("Incorrect input arguments. Usage: ./correct_run machine to from byte_low byte_upper log2_classes samples");
  	}

	if (strcmp(MACHINE, machine)) error("transfers_microbench_gpu: Running on wrong machine");
	filename = check_benchmark( to, from, byte_low, byte_upper, classes, samples);


	fprintf(stderr,"\nTransfer benchmark@%s %s->%s : (%ld,%ld) with %d samples \n", MACHINE, print_loc(from), print_loc(to), byte_low, byte_upper, samples);

	cudaGetDeviceCount(&dev_count);

	if (byte_low < 1) error("Transfer Microbench: Bytes must be > 0"); 
	else if ( dev_count < from + 1) error("Transfer Microbench: Src device does not exist"); 
	else if ( dev_count < to + 1) error("Transfer Microbench: Dest device does not exist"); 

	void* src, *dest, *rev_src, *rev_dest; 

	//Only model pinned memory transfers from host to dev and visa versa
  	if (from < 0 && to < 0) error("Transfer Microbench: Both locations are in host");
  	else if ( from >= 0 && to >= 0) error("Transfer Microbench: Both locations are devices - device communication not implemented");
	else if (from == -2 || to == -2) error("Transfer Microbench: Not pinned memory (synchronous)");
	
	long long *bench_bytes =  distribute_in_classes(classes, byte_low, byte_upper, samples);
	
	size_t cols, rows = sqrt(byte_upper/8), ldsrc, ldest; 

	vec_alloc(&src, byte_upper + rows*8, from);
	vec_alloc(&dest, byte_upper + rows*8, to);
	vec_alloc(&rev_src, byte_upper + rows*8, to);
	vec_alloc(&rev_dest, byte_upper + rows*8, from);

	/// Local Timers 
	double cpu_timer, t_normal = 0 , t_bid = 0, t_cublas = 0;
	gpu_timer_p cuda_timer = gpu_timer_init();

	if (from < 0) memset(src, 1, byte_upper + rows*8);

	cudaStream_t stream, reverse_stream;
	cudaStreamCreate(&stream);
	cudaStreamCreate(&reverse_stream);
	
	fprintf(stderr, "Warming up...\n");
	/// Warmup.
	for (int it = 0; it < 10; it++) {
		if(from == -1) cublasSetMatrixAsync(byte_upper, 1, sizeof(double), src, byte_upper, dest, byte_upper,stream);
		else cublasGetMatrixAsync(byte_upper, 1, sizeof(double), src, byte_upper, dest, byte_upper,stream);
	}
	cudaCheckErrors();

	fprintf(stderr, "Overhead Link %s->%s (%ld bytes):", print_loc(from), print_loc(to), 1);
	cpu_timer = - csecond();
	for (int it = 0; it < 1000*ITER ; it++) {
		if(from == -1) cublasSetMatrixAsync(1, 1, 1, src, 1, dest, 1, stream);
		else cublasGetMatrixAsync(1, 1, 1, src, 1, dest, 1, stream);
	}
	cudaStreamSynchronize(stream);
	cpu_timer = csecond() + cpu_timer;
	t_cublas = t_normal = cpu_timer/(1000*ITER);
	fprintf(stderr, "%lf ms\n", 1000*t_normal);

	fprintf(stderr, "Overhead cublas Link %s->%s (%ld bytes):", print_loc(from), print_loc(to), 1);
	fprintf(stderr, "%lf ms\n", 1000*t_cublas);

	fprintf(stderr, "Overhead Reverse overlapped Link %s->%s (%ld bytes):", print_loc(from), print_loc(to), 1);
	cpu_timer = - csecond();
	for (int it = 0; it < 1000*ITER ; it++) {
		for (int rep = 0; rep < 10 ; rep++) {
			if(to == -1) cublasSetMatrixAsync(cols, rows, 8, rev_src, ldsrc, rev_dest, ldest,reverse_stream);
			else cublasGetMatrixAsync(cols, 8, rows, rev_src, ldsrc, rev_dest, ldest,reverse_stream);
		}
		gpu_timer_start(cuda_timer, stream);
		if(from == -1) cublasSetMatrixAsync(1, 1, 1, src, 1, dest, 1,stream);
		else cublasGetMatrixAsync(1, 1, 1, src, 1, dest, 1,stream);
	}
	cudaStreamSynchronize(stream);
	cpu_timer = csecond() + cpu_timer;
	t_bid = cpu_timer/(1000*ITER);
	report_run(filename, 1, t_normal, t_bid, t_cublas);

	for (int rep = 0; rep < samples; rep++){

		bytes = bench_bytes[rep];
		rows = cols = (size_t) sqrt(bytes/8); 
		ldsrc = ldest = rows + 1;
		if (rows*cols*8 != bytes) warning("Transfer Microbench: sqrt for chunk is not exact");
	
		fprintf(stderr, "Normal Link %s->%s (%ld bytes):", print_loc(from), print_loc(to), bytes);
		for (int it = 0; it < ITER ; it++) {
			cpu_timer = - csecond();
			if(from == -1) cublasSetMatrixAsync(cols*rows, 1,  8, src, cols*rows, dest, cols*rows,stream);
			else cublasGetMatrixAsync(cols*rows, 1, 8, src, cols*rows, dest, cols*rows,stream);
			cudaStreamSynchronize(stream);
			cpu_timer = csecond() + cpu_timer;
			t_normal += cpu_timer;
		}
		t_normal = t_normal/ITER/*/1000*/;
		fprintf(stderr, "%lf ms ( %lf Gb/s)\n", 1000*t_normal, Gval_per_s(bytes,t_normal));

		fprintf(stderr, "Cublas-chunk Link %s->%s (%ld bytes):", print_loc(from), print_loc(to), bytes);
		for (int it = 0; it < ITER ; it++) {
			cpu_timer = - csecond();
			if(from == -1) cublasSetMatrixAsync(rows, cols, 8, src, ldsrc, dest, ldest,stream);
			else cublasGetMatrixAsync(rows, cols, 8, src, ldsrc, dest, ldest,stream);
			cudaStreamSynchronize(stream);
			cpu_timer = csecond() + cpu_timer;
			t_cublas += cpu_timer;
		}
		t_cublas = t_cublas/ITER/*/1000*/;
		fprintf(stderr, "%lf ms ( %lf Gb/s)\n", 1000*t_cublas, Gval_per_s(bytes,t_cublas));

		fprintf(stderr, "Reverse overlapped Link %s->%s (%ld bytes):", print_loc(from), print_loc(to), bytes);
		for (int it = 0; it < ITER ; it++) {
			for (int rep = 0; rep < 10 ; rep++) {
				if(to == -1) cublasSetMatrixAsync(rows, cols, 8, rev_src, ldsrc, rev_dest, ldest,reverse_stream);
				else cublasGetMatrixAsync(rows, cols, 8, rev_src, ldsrc, rev_dest, ldest,reverse_stream);
			}
			gpu_timer_start(cuda_timer, stream);
			if(from == -1) cublasSetMatrixAsync(rows, cols, 8, src, ldsrc, dest, ldest,stream);
			else cublasGetMatrixAsync(rows, cols, 8, src, ldsrc, dest, ldest,stream);
			gpu_timer_stop(cuda_timer, stream);
			cudaDeviceSynchronize();
			t_bid += gpu_timer_get(cuda_timer);
		}
		cudaCheckErrors();
		t_bid = t_bid/ITER/1000;
		fprintf(stderr, "%lf ms ( %lf Gb/s)\n", 1000*t_bid, Gval_per_s(bytes,t_bid));		
		report_run(filename, bytes, t_normal, t_bid, t_cublas);

	}
	vec_free(&src, from);
	vec_free(&dest, to); 
	vec_free(&rev_src, to);
	vec_free(&rev_dest, from); 
	return 0;
}
