cmake_minimum_required (VERSION 3.10)
#set(CMAKE_VERBOSE_MAKEFILE on)

###TODO:Define specific compiler, flags, includes, libs for the target system (IF needed).

## 1) CUDA + CUBLAS
set (CUDA_PREFIX "/usr/local/cuda-9.2")

# 	a) Compiler
set (CMAKE_CUDA_COMPILER ${CUDA_PREFIX}/bin/nvcc)

# 	b) Compile flags
set (CMAKE_CUDA_FLAGS "-w -O3 -arch=sm_70") #35 for Tesla K40, 70 for V100, 61 for GTX 1060/70
message("CMAKE_CUDA_FLAGS: " ${CMAKE_CUDA_FLAGS})

# 	c) Includes
set (CUDA_INCLUDE_DIRS ${CUDA_PREFIX}/include ${CUDA_PREFIX}/samples/common/inc)
include_directories(${CUDA_INCLUDE_DIRS})
message("CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})

# 	d) Link Libraries
set (CUDA_LD ${CUDA_PREFIX}/lib64) # or /lib depending on system
set (CUDA_LINK "-L${CUDA_LD} -lcudart -lcublas -lcurand") #  
message( "CUDA_LINK: " ${CUDA_LINK} )


## 2) C++ 

#	a) auto

#	b) compile flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -lm -std=gnu89 -Wuninitialized")
message( "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})

#	c) auto

#	d) auto

### Initialize the project 
project (EMLR CXX CUDA)
message( "PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR} )
message( "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR} )
message( "CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR} )

###TODO: Define custom compile flags.

## 3) Name of the target machine to avoid confusion between testbeds.
#add_definitions(-DMACHINE="testbedI_Tes-K40")
add_definitions(-DMACHINE="testbedII_Tes-V100")

## 4) Pass the EMLR directory as a string 
add_definitions(-DPROJECTDIR="${PROJECT_SOURCE_DIR}")

## 5) Define Benchmark Itteration number for each problem size. 
add_definitions(-DITER=100)

# EMLR header/include files
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include)
message( "My header files: " ${CMAKE_CURRENT_SOURCE_DIR}/include )
set (LINK_LIB "${CUDA_LINK} -lrt -ldl")
message( "LINK_LIB: " ${LINK_LIB})

file(GLOB LIBFILES ${PROJECT_SOURCE_DIR}/lib/*.c* )
add_library(MyLib ${LIBFILES})
message( "MyLib: " ${MyLib} )
target_link_libraries(MyLib ${LINK_LIB})

add_executable (daxpy_microbench_gpu ${PROJECT_SOURCE_DIR}/src/daxpy_microbench_gpu.cu )
add_executable (dgemv_microbench_gpu ${PROJECT_SOURCE_DIR}/src/dgemv_microbench_gpu.cu )
add_executable (dgemm_microbench_gpu ${PROJECT_SOURCE_DIR}/src/dgemm_microbench_gpu.cu )
add_executable (sgemm_microbench_gpu ${PROJECT_SOURCE_DIR}/src/sgemm_microbench_gpu.cu )
add_executable (transfers_microbench_gpu ${PROJECT_SOURCE_DIR}/src/transfers_microbench_gpu.cu )

target_link_libraries(transfers_microbench_gpu ${LINK_LIB} MyLib)
target_link_libraries(dgemm_microbench_gpu ${LINK_LIB} MyLib)
target_link_libraries(sgemm_microbench_gpu ${LINK_LIB} MyLib)
target_link_libraries(dgemv_microbench_gpu ${LINK_LIB} MyLib)
target_link_libraries(daxpy_microbench_gpu ${LINK_LIB} MyLib)
