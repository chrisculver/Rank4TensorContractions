#!/bin/bash

rm src/*.o
rm a.out

baseDir=/g/g15/culver5/Code/laph_build_and_run
codeDir=${baseDir}/code
Ieigen=${codeDir}/source/eigen
Cuda_root=/usr/tce/packages/cuda/cuda-11.1.1


nvcc -std=c++14 -dc -Iinclude -I${Cuda_root}/include -L${Cuda_root}/lib64 -lcudart -lcublas src/trace.cu -o src/trace.o
nvcc -std=c++14 -dc -Iinclude -I${Ieigen} -I${Cuda_root}/include/ -L${Cuda_root}/lib64 -lcudart -lcublas src/gpu_kernel.cu -o src/gpu_kernel.o
nvcc -std=c++14 -Iinclude -I${Ieigen} main.cpp -L${Cuda_root}/lib64 -lcudart -lcublas src/trace.o src/gpu_kernel.o
