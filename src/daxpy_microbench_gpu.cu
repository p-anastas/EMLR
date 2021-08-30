///
/// \author Anastasiadis Petros (panastas@cslab.ece.ntua.gr)
///
/// \brief A cublasDaxpy micro-benchmark
///

#include <cassert>
#include "cpu_utils.hpp"
#include "gpu_utils.hpp"

char* check_benchmark(short dev_id, size_t Nmin, size_t Nmax, size_t N_step, size_t samples_max){
	char *filename = (char *) malloc(256* sizeof(char));
	sprintf(filename, "%s/BenchOutputs/%s/cublasDaxpy/EMLR_dev-%d_min-%d_max-%d_step-%d_lim-%d.log", PROJECTDIR, MACHINE, dev_id, Nmin, Nmax, N_step, samples_max);

	FILE* fp = fopen(filename,"r");
	if (!fp) { 
		fp = fopen(filename,"w+");
		if (!fp) error("report_results: LogFile failed to open");
		else warning("Generating Logfile...");
		fclose(fp);
	}
	else {
		fprintf(stderr,"GPU DAXPY benchmark@%s dev=%d found: [min,max] = [%d,%d] with step(%d) and %d max_benchmarks\n", MACHINE, dev_id, Nmin, Nmax, N_step, samples_max);
		fclose(fp);	
		exit(1); 
	}
	return filename;		  	
}

void report_run(char* filename, short dev_id, size_t N, double cublas_t_av, double cublas_t_min, double cublas_t_max){

	FILE* fp = fopen(filename,"a");
	if (!fp) error("report_run: LogFile failed to open");
   	fprintf(fp,"%d, %e,%e,%e\n", N, cublas_t_av, cublas_t_min, cublas_t_max);
        fclose(fp); 
}

int main(const int argc, const char *argv[]) {

  	double alpha;
  	alpha = 1.1234;

  	int ctr = 1, dev_id, samples_max, bench_num = 0;

	char machine[256];
	size_t Nmin, Nmax, N_step;
	size_t incx = 1, incy = 1;

	switch (argc) {
	case (7):
		sprintf(machine , "%s", argv[ctr++]);
		dev_id = atoi(argv[ctr++]);
		Nmin = atoi(argv[ctr++]);
		Nmax = atoi(argv[ctr++]);
		N_step = atoi(argv[ctr++]);
		samples_max = atoi(argv[ctr++]);
		break;
	default:
		error("Incorrect input arguments. Usage: ./correct_run machine dev_id Nmin Nmax N_step max_benchmarks");
  	}

	if (strcmp(MACHINE, machine)) error("daxpy_microbench_gpu: Running on wrong machine");
	char *filename = check_benchmark(dev_id, Nmin, Nmax, N_step, samples_max);

	/// Set device 
	cudaSetDevice(dev_id);

	cublasHandle_t handle0;
 	cudaStream_t host_stream;

  	cudaStreamCreate(&host_stream);
	assert(CUBLAS_STATUS_SUCCESS == cublasCreate(&handle0));
	assert(CUBLAS_STATUS_SUCCESS == cublasSetStream(handle0, host_stream));

	double cpu_timer = csecond();

	double *x_dev, *y_dev;
  	vec_alloc((void**)&x_dev, Nmax * sizeof(double), dev_id);
  	vec_alloc((void**)&y_dev, Nmax * sizeof(double), dev_id);
	cudaCheckErrors();

	cpu_timer  = csecond() - cpu_timer ;
	fprintf(stderr, "Allocated Device memory t_aloc = %lf ms\n", cpu_timer  * 1000);

	cpu_timer = csecond();

	Dinit_cuRAND(x_dev, Nmax, 42);
	Dinit_cuRAND(y_dev, Nmax, 42*42);

	cudaCheckErrors();
	cpu_timer  = csecond() - cpu_timer ;	
	fprintf(stderr, "Initialized Device Mem: t_init = %lf ms\n", cpu_timer  * 1000);

	fprintf(stderr, "\nTile details: x(inc=%d) y(inc=%d) -> Nmax = %d\n", 1, 1, Nmax);

	fprintf(stderr, "Constants: alpha = %lf\n", alpha);

	// Warmup 
	for ( int itt = 0; itt <10; itt++){
		assert(CUBLAS_STATUS_SUCCESS == cublasDaxpy(handle0, Nmax, &alpha, x_dev, incx, y_dev, incy));
		cudaStreamSynchronize(host_stream);
	}
	cudaCheckErrors();
	double cublas_t_av, cublas_t_min , cublas_t_max; 
	size_t bench_ctr = 0;
	for (size_t N = Nmin; N < Nmax + 1; N+=N_step) {
		if(bench_ctr > samples_max) break; 
		fprintf(stderr,"Running CUBLAS GPU-> N = %d\n", N);
		cublas_t_av = cublas_t_max = 0;
		cublas_t_min = 1e9;
		for (int itt = 0; itt < ITER; itt ++) {
			cpu_timer = csecond();
			assert(CUBLAS_STATUS_SUCCESS == cublasDaxpy(handle0, N, &alpha, x_dev, incx, y_dev, incy));
			cudaStreamSynchronize(host_stream);
			cpu_timer  = csecond() - cpu_timer ;
			cublas_t_av += cpu_timer;
			if (cpu_timer > cublas_t_max) cublas_t_max = cpu_timer; 
			if (cpu_timer < cublas_t_min) cublas_t_min = cpu_timer; 
		}
		cublas_t_av /= ITER;
		fprintf(stderr, "CUBLAS GPU:\n GPU exec time:\t Average=%lf ms, Min = %lf ms, Max = %lf ms\n", cublas_t_av  * 1000, cublas_t_min  * 1000, cublas_t_max  * 1000);
		cudaCheckErrors();

		report_run(filename, dev_id, N, cublas_t_av, cublas_t_min, cublas_t_max); 
		bench_ctr++;
	}
	fprintf(stderr, "Ran %d Benchmarks.Finallizing...\n", bench_ctr);
	return 0;
}
