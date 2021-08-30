#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
echo "Rversion $(Rscript --version)..."

###TODO: Change this to the EMLR master directory 
PROJECT_DIR=/home/users/panastas/PhD_stuff/EMLR
cd $PROJECT_DIR

###TODO: Get Input arguments from CMD 
machine=$1 # The machine name which should match the CmakeLists.txt machine (to avoid using incorrect system source) 
device=$2  # Target device for the GPU benchmarks

RESDIR=$PROJECT_DIR/BenchOutputs/$machine

mkdir -p "${RESDIR}/exec_logs"

RSRC_DIR=$PROJECT_DIR/R_scripts

rm ${RESDIR}/exec_logs/*_R_models.log
transfer_R_models_log="${RESDIR}/exec_logs/transfer_R_models.log"
dgemm_R_models_log="${RESDIR}/exec_logs/dgemm_R_models.log"
sgemm_R_models_log="${RESDIR}/exec_logs/sgemm_R_models.log"
dgemv_R_models_log="${RESDIR}/exec_logs/dgemv_R_models.log"
daxpy_R_models_log="${RESDIR}/exec_logs/daxpy_R_models.log"

micro_transfer_model="${RSRC_DIR}/transfer_models.r"
stepwise_model="${RSRC_DIR}/Stepwise_model_AIC.r"

# Transfer micro-benchmark deployment values
BYTESLOW=10240 # 10 KB
BYTESUPPER=104857600 # 100 MB
TRANSFER_LOG_CLASSES=10
TRANSFER_SAMPLES=100

echo "Performing linear regression for transfers..."
echo "Rscript $micro_transfer_model $PROJECT_DIR $machine $device $BYTESLOW $BYTESUPPER $TRANSFER_LOG_CLASSES $TRANSFER_SAMPLES &> $transfer_R_models_log"
Rscript $micro_transfer_model $PROJECT_DIR $machine $device $BYTESLOW $BYTESUPPER $TRANSFER_LOG_CLASSES $TRANSFER_SAMPLES &> $transfer_R_models_log
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

echo "Performing stepwise multiple regression for Dgemm..."
echo "Rscript $stepwise_model $PROJECT_DIR $machine $device Dgemm $Dmin $Mmax $Nmax $Kmax $Mstep $Nstep $Kstep $MAX_GEMM_SAMPLES &> $dgemm_R_models_log"
Rscript $stepwise_model "$PROJECT_DIR" $machine $device Dgemm $Dmin $Mmax $Nmax $Kmax $Mstep $Nstep $Kstep $MAX_GEMM_SAMPLES &> $dgemm_R_models_log
echo "Done"

echo "Performing stepwise multiple regression for Sgemm..."
echo "Rscript $stepwise_model $PROJECT_DIR $machine $device Sgemm $Dmin $Mmax $Nmax $Kmax $Mstep $Nstep $Kstep $MAX_GEMM_SAMPLES &> $sgemm_R_models_log"
Rscript $stepwise_model $PROJECT_DIR $machine $device Sgemm $Dmin $Mmax $Nmax $Kmax $Mstep $Nstep $Kstep $MAX_GEMM_SAMPLES &> $sgemm_R_models_log
echo "Done"

# gemv micro-benchmark deployment values
Mmax=16384
Nmax=16384
MAX_GEMV_SAMPLES=1000

echo "Performing stepwise multiple regression for Dgemv..."
echo "Rscript $stepwise_model $PROJECT_DIR $machine $device Dgemv $Dmin $Mmax $Nmax $Mstep $Nstep $MAX_GEMV_SAMPLES &> $dgemv_R_models_log"
Rscript $stepwise_model $PROJECT_DIR $machine  $device Dgemv $Dmin $Mmax $Nmax $Mstep $Nstep $MAX_GEMV_SAMPLES &> $dgemv_R_models_log
echo "Done"

# axpy micro-benchmark deployment values
Dmin=262144
Nstep=262144
Nmax=100000000
MAX_AXPY_SAMPLES=1000

echo "Performing stepwise multiple regression for Daxpy..."
echo "Rscript $stepwise_model $PROJECT_DIR $machine $device Daxpy $Dmin $Nmax $Nstep $MAX_AXPY_SAMPLES &> $daxpy_R_models_log"
Rscript $stepwise_model $PROJECT_DIR $machine $device Daxpy $Dmin $Nmax $Nstep $MAX_AXPY_SAMPLES &> $daxpy_R_models_log
echo "Done"

exit 0 











