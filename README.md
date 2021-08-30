# EMLR
A set of **E**mpirical **M**icrobenchmarks for **L**inear **R**egression intended for BLAS time prediction.
The backend microbencmarks are performed using cuBLAS, but can be easily configured for any GPU library with the BLAS layout. 

*This code has been fully integrated into CoCoPeLia(https://github.com/p-anastas/CoCoPeLia-Framework) and is now deprecated*

# Requirements

- CUDA (Tested with 7.0 and 9.2, but should work with older/newer versions)
- CUBLAS library (default = installed along with CUDA)
- NVIDIA GPU with support for double operations. 
- Cmake 3.10 or greater

# Deployment

**Build:**
- Modify the library paths at CmakeLists.txt for your system.
- *mkdir build && cd build*
- *cmake ../ && make -j 4*

**Run Microbenchmarks:**
- Modify *$PROJECT_DIR* in Deploy_micro-benchmarks.sh.
- Run *Deploy_micro-benchmarks.sh buildir machinename cuda_dev_id*
- The microbenchmarks might take a while depending on your machine characteristics (~hours). 

**Create Models with R:**
- Modify *$PROJECT_DIR* in Generate_R_models.sh.
- Run *Generate_R_models.sh machinename cuda_dev_id*.
