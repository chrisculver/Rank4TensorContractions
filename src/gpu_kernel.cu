#include "gpu_kernel.h"
#include "trace.h"
#include "timer.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

void single_index_contract(std::complex<double> *res, std::complex<double> *bpropMat, std::complex<double> *bsinkMat, long int dim)
{
  //cublas setup - taken from examples online
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  stat = cublasCreate(&handle);
  cublasSetStream(handle, stream);
  // cublas multiply can do C=alphaA*B + betaC or something like that
  std::complex<double> alpha(1.,0.);
  std::complex<double> beta(0.,0.);
  cuDoubleComplex *_alpha = reinterpret_cast<cuDoubleComplex*>(&alpha);
  cuDoubleComplex *_beta = reinterpret_cast<cuDoubleComplex*>(&beta);
  int block_size = 32;
  dim3 threads(block_size, block_size);
  dim3 grid(dim/threads.x, dim/threads.y);

  long int bTensor_size = dim*dim*dim*dim*sizeof(std::complex<double>);
  long int resTensor_size = dim*dim*dim*dim*dim*dim*sizeof(std::complex<double>);

  cuDoubleComplex *d_A, *d_B, *d_C;
  cudaMalloc((void **) &d_A, bTensor_size);
  cudaMalloc((void **) &d_B, bTensor_size);
  cudaMalloc((void **) &d_C, resTensor_size);
 
  cudaMemcpy(d_A, bpropMat, bTensor_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, bsinkMat, bTensor_size, cudaMemcpyHostToDevice);
 
  Timer<> gpu_timer("Zgemm time");
  cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
              //Left index size of A, Right index size of B, summed index
              dim*dim*dim, dim*dim*dim, dim, 
              _alpha, d_A, dim*dim*dim, 
                     d_B, dim, 
              _beta, d_C, dim*dim*dim);
  gpu_timer.print<std::chrono::microseconds>("us");
 
  cudaMemcpy(res, d_C, resTensor_size, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  
  cublasDestroy(handle);
  cudaStreamSynchronize(0);
  cudaStreamDestroy(stream);
}


void all_index_contract(std::complex<double> *res, std::complex<double> *bpropMat, std::complex<double> *bsinkMat, long int dim)
{
  Timer<> setup_timer("setup all_index_contract");
  
  //cublas setup - taken from examples online
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  stat = cublasCreate(&handle);
  cublasSetStream(handle, stream);
  // cublas multiply can do C=alphaA*B + betaC or something like that
  std::complex<double> alpha(1.,0.);
  std::complex<double> beta(0.,0.);
  cuDoubleComplex *_alpha = reinterpret_cast<cuDoubleComplex*>(&alpha);
  cuDoubleComplex *_beta = reinterpret_cast<cuDoubleComplex*>(&beta);
  int block_size = 32;
  dim3 threads(block_size, block_size);
  dim3 grid(dim/threads.x, dim/threads.y);

  long int bTensor_size = dim*dim*dim*dim*sizeof(std::complex<double>);
  long int resTensor_size = dim*dim*dim*dim*dim*dim*sizeof(std::complex<double>);

  cuDoubleComplex *d_A, *d_B, *d_C, *d_res;
  cudaMalloc((void **) &d_A, bTensor_size);
  cudaMalloc((void **) &d_B, bTensor_size);
  cudaMalloc((void **) &d_C, resTensor_size);
  cudaMalloc((void **) &d_res, 2*sizeof(std::complex<double>));
  

  //d_res[0] = make_cuDoubleComplex(0.,0.);
  cudaMemcpy(d_A, bpropMat, bTensor_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, bsinkMat, bTensor_size, cudaMemcpyHostToDevice);
  setup_timer.print<std::chrono::microseconds>("us");

  Timer<> gpu_timer("Zgemm time all");
  cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
              //Left index size of A, Right index size of B, summed index
              dim*dim*dim, dim*dim*dim, dim, 
              _alpha, d_A, dim*dim*dim, 
                     d_B, dim, 
              _beta, d_C, dim*dim*dim);
  cudaDeviceSynchronize();
  gpu_timer.print<std::chrono::microseconds>("us");
 
  Timer<> trace_timer("Trace time all");
  trace_rank6<<<1,1>>>(d_res, d_C, dim);
  cudaDeviceSynchronize();
  trace_timer.print<std::chrono::microseconds>("us");

  cudaMemcpy(res, d_res, sizeof(std::complex<double>), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_res);
  
  cublasDestroy(handle);
  cudaStreamSynchronize(0);
  cudaStreamDestroy(stream);
}


void contract(std::complex<double> *res, std::complex<double> *bpropMat, std::complex<double> *bsinkMat, long int dim)
{
  Timer<> setup_timer("setup contract");
  
  //cublas setup - taken from examples online
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  stat = cublasCreate(&handle);
  cublasSetStream(handle, stream);
  // cublas multiply can do C=alphaA*B + betaC or something like that
  std::complex<double> alpha(1.,0.);
  std::complex<double> beta(0.,0.);
  cuDoubleComplex *_alpha = reinterpret_cast<cuDoubleComplex*>(&alpha);
  cuDoubleComplex *_beta = reinterpret_cast<cuDoubleComplex*>(&beta);
  int block_size = 32;
  dim3 threads(block_size, block_size);
  dim3 grid(dim/threads.x, dim/threads.y);

  long int bTensor_size = dim*dim*dim*dim*sizeof(std::complex<double>);
  long int resTensor_size = dim*dim*dim*dim*dim*dim*sizeof(std::complex<double>);

  cuDoubleComplex *d_A, *d_B, *d_C, *d_res;
  cudaMalloc((void **) &d_A, bTensor_size);
  cudaMalloc((void **) &d_B, bTensor_size);
  cudaMalloc((void **) &d_C, resTensor_size);
  cudaMalloc((void **) &d_res, 2*sizeof(std::complex<double>));
  

  //d_res[0] = make_cuDoubleComplex(0.,0.);
  cudaMemcpy(d_A, bpropMat, bTensor_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, bsinkMat, bTensor_size, cudaMemcpyHostToDevice);
  setup_timer.print<std::chrono::microseconds>("us");
 
  Timer<> contract_timer("contract time");
  contract<<<1,1>>>(d_res, d_A, d_B, dim);
  cudaDeviceSynchronize();
  contract_timer.print<std::chrono::microseconds>("us");

  cudaMemcpy(res, d_res, sizeof(std::complex<double>), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_res);
  
  cublasDestroy(handle);
  cudaStreamSynchronize(0);
  cudaStreamDestroy(stream);
}
