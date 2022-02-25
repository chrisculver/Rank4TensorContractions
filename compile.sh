#!/bin/bash

rm src/*.o
rm a.out

baseDir=/g/g15/culver5/Code/laph_build_and_run
codeDir=${baseDir}/code
Ieigen=${codeDir}/source/eigen
Cuda_root=/usr/tce/packages/cuda/cuda-11.1.1
cuTensorDir=/usr/WS1/culver5/libs/libcutensor-linux-openpower-1.4.0.6

nvcc -std=c++14 -dc -Iinclude -I${Cuda_root}/include -L${Cuda_root}/lib64 -lcudart -lcublas src/trace.cu -o src/trace.o
nvcc -std=c++14 -dc -Iinclude -I${Ieigen} -I${Cuda_root}/include/ -I${cuTensorDir}/include/ -L${Cuda_root}/lib64 -L${cuTensorDir}/lib/11.0 -lcudart -lcublas -lcutensor src/gpu_kernel.cu -o src/gpu_kernel.o
nvcc -std=c++14 -dc -Iinclude -I${Ieigen} -I${Cuda_root}/include/ -I${cuTensorDir}/include/ -L${Cuda_root}/lib64 -L${cuTensorDir}/lib/11.0 -lcudart -lcublas -lcutensor src/tensor_kernel.cu -o src/tensor_kernel.o
nvcc -std=c++14 -Iinclude -I${Ieigen} main.cpp -L${Cuda_root}/lib64 -L${cuTensorDir}/lib/11.0 -lcudart -lcublas -lcutensor src/trace.o src/gpu_kernel.o src/tensor_kernel.o
