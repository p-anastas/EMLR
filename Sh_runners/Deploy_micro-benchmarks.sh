#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

###TODO: Change this to the EMLR master directory 
PROJECT_DIR=/home/users/panastas/PhD_stuff/EMLR
cd $PROJECT_DIR

###TODO: Get Input arguments from CMD 
BUILDIR=$1 # The directory for building the microbenchmarks
machine=$2 # The machine name which should match the CmakeLists.txt machine (to avoid using incorrect system source) 
device=$3  # Target device for the GPU benchmarks

## Build the microbenchmarks every time to track changes
cd $BUILDIR
make -j 8
cd $PROJECT_DIR

RESDIR=$PROJECT_DIR/BenchOutputs/$machine

mkdir -p "${RESDIR}/exec_logs"

rm ${RESDIR}/exec_logs/*_microbench_gpu.log
transfer_log="${RESDIR}/exec_logs/transfer_microbench_gpu.log"
dgemm_log="${RESDIR}/exec_logs/dgemm_microbench_gpu.log"
sgemm_log="${RESDIR}/exec_logs/sgemm_microbench_gpu.log"
dgemv_log="${RESDIR}/exec_logs/dgemv_microbench_gpu.log"
daxpy_log="${RESDIR}/exec_logs/daxpy_microbench_gpu.log"

micro_transfer_exec="${BUILDIR}/transfers_microbench_gpu"
micro_dgemm_exec="${BUILDIR}/dgemm_microbench_gpu"
micro_sgemm_exec="${BUILDIR}/sgemm_microbench_gpu"
micro_dgemv_exec="${BUILDIR}/dgemv_microbench_gpu"
micro_daxpy_exec="${BUILDIR}/daxpy_microbench_gpu"

# Transfer micro-benchmark deployment values
BYTESLOW=10240 # 10 KB
BYTESUPPER=104857600 # 100 MB
TRANSFER_LOG_CLASSES=10
TRANSFER_SAMPLES=100

echo "Performing microbenchmarks for transfers..."
mkdir -p "${RESDIR}/cublasSet_Get"
echo "$micro_transfer_exec $machine $device -1 $BYTESLOW $BYTESUPPER $TRANSFER_LOG_CLASSES $TRANSFER_SAMPLES &>> $transfer_log"
$micro_transfer_exec $machine $device -1 $BYTESLOW $BYTESUPPER $TRANSFER_LOG_CLASSES $TRANSFER_SAMPLES &>> $transfer_log
echo "$micro_transfer_exec $machine -1 $device $BYTESLOW $BYTESUPPER $TRANSFER_LOG_CLASSES $TRANSFER_SAMPLES &>> $transfer_log"
$micro_transfer_exec $machine -1 $device $BYTESLOW $BYTESUPPER $TRANSFER_LOG_CLASSES $TRANSFER_SAMPLES &>> $transfer_log
echo "Done"

# General benchmark steps
Mstep=512
Nstep=512
Kstep=512
Dmin=512

# gemm micro-benchmark deployment values
Mmax=8192
Nmax=8192
Kmax=8192
MAX_GEMM_SAMPLES=1000

echo "Performing microbenchmarks for dgemm..."
mkdir -p "${RESDIR}/cublasDgemm"
echo "$micro_dgemm_exec $machine $device $Dmin $Mmax $Nmax $Kmax $Mstep $Nstep $Kstep $MAX_GEMM_SAMPLES &>> $dgemm_log"
$micro_dgemm_exec $machine $device $Dmin $Mmax $Nmax $Kmax $Mstep $Nstep $Kstep $MAX_GEMM_SAMPLES &>> $dgemm_log
echo "Done"

echo "Performing microbenchmarks for sgemm..."
mkdir -p "${RESDIR}/cublasSgemm"
echo "$micro_sgemm_exec $machine $device $Dmin $Mmax $Nmax $Kmax $Mstep $Nstep $Kstep $MAX_GEMM_SAMPLES &>> $sgemm_log"
$micro_sgemm_exec $machine $device $Dmin $Mmax $Nmax $Kmax $Mstep $Nstep $Kstep $MAX_GEMM_SAMPLES &>> $sgemm_log
echo "Done"

# gemv micro-benchmark deployment values
Mmax=16384
Nmax=16384
MAX_GEMV_SAMPLES=1000

echo "Performing microbenchmarks for dgemv..."
mkdir -p "${RESDIR}/cublasDgemv"
echo "$micro_dgemv_exec $machine $device $Dmin $Mmax $Nmax $Mstep $Nstep $MAX_GEMV_SAMPLES &>> $dgemv_log"
$micro_dgemv_exec $machine $device $Dmin $Mmax $Nmax $Mstep $Nstep $MAX_GEMV_SAMPLES &>> $dgemv_log
echo "Done"

# axpy micro-benchmark deployment values
Dmin=262144
Nmax=100000000
Nstep=262144
MAX_AXPY_SAMPLES=1000

echo "Performing microbenchmarks for daxpy..."
mkdir -p "${RESDIR}/cublasDaxpy"
echo "$micro_daxpy_exec $machine $device $Dmin $Nmax $Nstep $MAX_AXPY_SAMPLES &>> $daxpy_log"
$micro_daxpy_exec $machine $device $Dmin $Nmax $Nstep $MAX_AXPY_SAMPLES &>> $daxpy_log
echo "Done"
