///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief A cublasDgemv micro-benchmark
///

#include <cassert>
#include "cpu_utils.hpp"
#include "gpu_utils.hpp"

char* check_benchmark(short dev_id, size_t minDim, size_t Mmax, size_t Nmax, size_t M_step, size_t N_step, size_t samples_max){
	char *filename = (char *) malloc(256* sizeof(char));
	sprintf(filename, "%s/BenchOutputs/%s/cublasDgemv/EMLR_dev-%d_minDim-%d_max-%d-%d_step-%d-%d_lim-%d.log", PROJECTDIR, MACHINE, dev_id, minDim, Mmax, Nmax, M_step, N_step, samples_max);

	FILE* fp = fopen(filename,"r");
	if (!fp) { 
		fp = fopen(filename,"w+");
		if (!fp) error("report_results: LogFile failed to open");
		else warning("Generating Logfile...");
		fclose(fp);
	}
	else {
		fprintf(stderr,"GPU DGEMV benchmark@%s dev=%d found: minDim = %d, max dims(%d,%d) with steps(%d,%d) and max %d benchmarks\n", MACHINE, dev_id, minDim, Mmax, Nmax, M_step, N_step, samples_max);
		fclose(fp);	
		exit(1); 
	}
	return filename;		  	
}
void report_run(char* filename, short dev_id, size_t M, size_t N, double cublas_t_av, double cublas_t_min, double cublas_t_max){

	FILE* fp = fopen(filename,"a");
	if (!fp) error("report_run: LogFile failed to open");
   	fprintf(fp,"%d,%d, %e,%e,%e\n", M, N, cublas_t_av, cublas_t_min, cublas_t_max);
        fclose(fp); 
}

int main(const int argc, const char *argv[]) {

  	double alpha, beta;
  	alpha = 1.1234, beta = 1.2345;

  	int ctr = 1, dev_id, samples_max, bench_num = 0;

	char machine[256];
	size_t minDim, Mmax, Nmax;
	size_t M_step, N_step;
	size_t incx = 1, incy = 1;

	switch (argc) {
	case (9):
		sprintf(machine , "%s", argv[ctr++]);
		dev_id = atoi(argv[ctr++]);
		minDim = atoi(argv[ctr++]);
		Mmax = atoi(argv[ctr++]);
		Nmax = atoi(argv[ctr++]);
		M_step = atoi(argv[ctr++]);
		N_step = atoi(argv[ctr++]);
		samples_max = atoi(argv[ctr++]);
		break;
	default:
		error("Incorrect input arguments. Usage: ./correct_run machine dev_id minDim Mmax Nmax M_step N_step max_benchmarks");
  	}

	if (strcmp(MACHINE, machine)) error("dgemv_microbench_gpu: Running on wrong machine");
	char *filename = check_benchmark(dev_id, minDim, Mmax, Nmax, M_step, N_step, samples_max);

	/// Matrix Layouts for GPU GEMM
	cublasOperation_t gpu_op_A = CUBLAS_OP_N;  // CUBLAS_OP_N, CUBLAS_OP_T

	size_t ldA = Mmax;

	/// Set device 
	cudaSetDevice(dev_id);

	cublasHandle_t handle0;
 	cudaStream_t host_stream;

  	cudaStreamCreate(&host_stream);
	assert(CUBLAS_STATUS_SUCCESS == cublasCreate(&handle0));
	assert(CUBLAS_STATUS_SUCCESS == cublasSetStream(handle0, host_stream));

	double cpu_timer = csecond();

	double *A_dev, *x_dev, *y_dev;
  	vec_alloc((void**)&A_dev, Mmax * Nmax * sizeof(double), dev_id);
  	vec_alloc((void**)&x_dev, Nmax * sizeof(double), dev_id);
  	vec_alloc((void**)&y_dev, Mmax * sizeof(double), dev_id);
	cudaCheckErrors();

	cpu_timer  = csecond() - cpu_timer ;
	fprintf(stderr, "Allocated Device memory t_aloc = %lf ms\n", cpu_timer  * 1000);

	cpu_timer = csecond();

	Dinit_cuRAND(A_dev, Mmax * Nmax, 42);
	Dinit_cuRAND(x_dev, Nmax, 42*42);
	Dinit_cuRAND(y_dev, Mmax, 42*42*42);

	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer ;	
	fprintf(stderr, "Initialized Device Mem: t_init = %lf ms\n", cpu_timer  * 1000);

	fprintf(stderr, "\nTile details: A(%s) x(inc=%d) y(inc=%d) -> Mmax = %d, Nmax = %d\n",
            print_mem(COL_MAJOR), 1, 1, Mmax, Nmax);
	fprintf(stderr, "Constants: alpha = %lf, beta = %lf\n", alpha, beta);

	// Warmup 
	for ( int itt = 0; itt <10; itt++){
		assert(CUBLAS_STATUS_SUCCESS == cublasDgemv(handle0, gpu_op_A, Mmax, Nmax, &alpha, A_dev, ldA, x_dev, incx, &beta, y_dev, incy));
		cudaStreamSynchronize(host_stream);
	}
	cudaCheckErrors();
	double cublas_t_av, cublas_t_min , cublas_t_max; 
	size_t bench_ctr = 0;
	for (size_t M = minDim; M < Mmax + 1; M+=M_step) 
		for (size_t N = minDim; N < Nmax + 1; N+=N_step) {
			ldA = M;
			if(bench_ctr > samples_max) break; 
			fprintf(stderr,"Running CUBLAS GPU-> M = %d, N = %d\n", M, N);
			cublas_t_av = cublas_t_max = 0;
			cublas_t_min = 1e9;
			for (int itt = 0; itt < ITER; itt ++) {
				cpu_timer = csecond();
				assert(CUBLAS_STATUS_SUCCESS == cublasDgemv(handle0, gpu_op_A, M, N, &alpha, A_dev, ldA, x_dev, incx, &beta, y_dev, incy));
				cudaStreamSynchronize(host_stream);
				cpu_timer  = csecond() - cpu_timer ;
				cublas_t_av += cpu_timer;
				if (cpu_timer > cublas_t_max) cublas_t_max = cpu_timer; 
				if (cpu_timer < cublas_t_min) cublas_t_min = cpu_timer; 
			}
			cublas_t_av /= ITER;
			fprintf(stderr, "CUBLAS GPU:\n GPU exec time:\t Average=%lf ms, Min = %lf ms, Max = %lf ms\n", cublas_t_av  * 1000, cublas_t_min  * 1000, cublas_t_max  * 1000);
			cudaCheckErrors();

			report_run(filename, dev_id, M, N, cublas_t_av, cublas_t_min, cublas_t_max); 
			bench_ctr++;
	}
	fprintf(stderr, "Ran %d Benchmarks.Finallizing...\n", bench_ctr);
	return 0;
}
