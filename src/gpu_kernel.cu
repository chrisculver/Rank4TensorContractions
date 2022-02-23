#include "gpu_kernel.h"
#include "trace.h"
#include "timer.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cutensor.h>

#include <unordered_map>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

//cTensor Error Handler
#define HANDLE_ERROR(x) { \
  const auto err = x;     \
  if( err != CUTENSOR_STATUS_SUCCESS ) \
  { \
    printf("Error: %s in line %d\n", cutensorGetErrorString(err), __LINE__); \
    exit(-1); \
  } \
}


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
  gpu_timer.stop<std::chrono::microseconds>("us");
 
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
  setup_timer.stop<std::chrono::microseconds>("us");

  Timer<> gpu_timer("Zgemm time all");
  cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
              //Left index size of A, Right index size of B, summed index
              dim*dim*dim, dim*dim*dim, dim, 
              _alpha, d_A, dim*dim*dim, 
                     d_B, dim, 
              _beta, d_C, dim*dim*dim);
  cudaDeviceSynchronize();
  gpu_timer.stop<std::chrono::microseconds>("us");
 
  Timer<> trace_timer("Trace time all");
  trace_rank6<<<1,1>>>(d_res, d_C, dim);
  cudaDeviceSynchronize();
  trace_timer.stop<std::chrono::microseconds>("us");

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
  setup_timer.stop<std::chrono::microseconds>("us");
 
  Timer<> contract_timer("contract time");
  contract<<<1,1>>>(d_res, d_A, d_B, dim);
  cudaDeviceSynchronize();
  contract_timer.stop<std::chrono::microseconds>("us");

  cudaMemcpy(res, d_res, sizeof(std::complex<double>), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_res);
  
  cublasDestroy(handle);
  cudaStreamSynchronize(0);
  cudaStreamDestroy(stream);
}

void contract4(std::complex<double> *res, std::complex<double> *A, std::complex<double> *B, std::complex<double> *C, std::complex<double> *D, long int dim)
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

  cuDoubleComplex *d_A, *d_B, *d_C, *d_D, *d_AB, *d_CD, *d_res;
  cudaMalloc((void **) &d_A, bTensor_size);
  cudaMalloc((void **) &d_B, bTensor_size);
  cudaMalloc((void **) &d_C, bTensor_size);
  cudaMalloc((void **) &d_D, bTensor_size);
  cudaMalloc((void **) &d_AB, bTensor_size);
  cudaMalloc((void **) &d_CD, bTensor_size);
  cudaMalloc((void **) &d_res, 2*sizeof(std::complex<double>));
   

  //d_res[0] = make_cuDoubleComplex(0.,0.);
  cudaMemcpy(d_A, A, bTensor_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, bTensor_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, bTensor_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, D, bTensor_size, cudaMemcpyHostToDevice);
  cudaMemset(d_AB, 0, bTensor_size);
  cudaMemset(d_CD, 0, bTensor_size);
  cudaMemset(d_AB, 0, bTensor_size);
  cudaMemset(d_res, 0, 2*sizeof(std::complex<double>));
  setup_timer.stop<std::chrono::microseconds>("us");
 
  Timer<> contract_timer("contract time");
  contractAB<<<1,1>>>(d_AB, d_A, d_B, dim);
  contractCD<<<1,1>>>(d_CD, d_C, d_D, dim);
  contract<<<1,1>>>(d_res, d_AB, d_CD, dim);
  cudaDeviceSynchronize();
  contract_timer.stop<std::chrono::microseconds>("us");

  cudaMemcpy(res, d_res, sizeof(std::complex<double>), cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_res);
  
  cublasDestroy(handle);
  cudaStreamSynchronize(0);
  cudaStreamDestroy(stream);
}


void cuTensorContract(std::complex<double> *res, std::complex<double> *A,std::complex<double> *B, long int dim)
{
  // allocate device memory and copy tensors
  long int bTensor_size = dim*dim*dim*dim*sizeof(std::complex<double>);

  cuDoubleComplex *d_A, *d_B, *d_C, *d_D;
  cudaMalloc((void **) &d_A, bTensor_size);
  cudaMalloc((void **) &d_B, bTensor_size);
  cudaMalloc((void **) &d_C, dim*dim*sizeof(std::complex<double>));
  cudaMalloc((void **) &d_D, sizeof(std::complex<double>));

  cudaMemcpy(d_A, A, bTensor_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, bTensor_size, cudaMemcpyHostToDevice);
  
  cudaMemset(d_C, 0, dim*dim*sizeof(std::complex<double>));
  cudaMemset(d_D, 0, sizeof(std::complex<double>));


  //types of cuTensor
  Timer<> cutensor_setup("cutensor setup time");
  cudaDataType_t tensType = CUDA_C_64F;
  cutensorComputeType_t computeType = CUTENSOR_COMPUTE_64F;

  typedef float floatTypeCompute;

  cuDoubleComplex alpha = make_cuDoubleComplex(1.0,0.0);
  cuDoubleComplex beta = make_cuDoubleComplex(0.0,0.0);

  //modes of tensors
  std::vector<int> modeC{'a','b'};
  std::vector<int> modeA{'a','j','k','l'};
  std::vector<int> modeB{'l','k','j','b'};

  int nmodeA = modeA.size();
  int nmodeB = modeB.size();
  int nmodeC = modeC.size();

  //extents of modes
  std::unordered_map<int, int64_t> extent;
  extent['i']=dim;
  extent['j']=dim;
  extent['k']=dim;
  extent['l']=dim;
  extent['a']=dim;
  extent['b']=dim;

  std::vector<int64_t> extentA, extentB, extentC;
  for(auto mode: modeA)
    extentA.push_back(extent[mode]);
  for(auto mode: modeB)
    extentB.push_back(extent[mode]);
  for(auto mode: modeC)
    extentC.push_back(extent[mode]); 

  //create tensor descriptors 
  cutensorHandle_t handle;
  cutensorInit(&handle);

  cutensorTensorDescriptor_t descA, descB, descC;
  HANDLE_ERROR( cutensorInitTensorDescriptor(
      &handle,
      &descA,
      nmodeA,
      extentA.data(),
      NULL,/*stride*/
      tensType,
      CUTENSOR_OP_IDENTITY/*applied to each element*/
    )
  );
  HANDLE_ERROR( cutensorInitTensorDescriptor(
      &handle,
      &descB,
      nmodeB,
      extentB.data(),
      NULL,/*stride*/
      tensType,
      CUTENSOR_OP_IDENTITY/*applied to each element*/
    )
  );
  HANDLE_ERROR( cutensorInitTensorDescriptor(
      &handle,
      &descC,
      nmodeC,
      extentC.data(),
      NULL,/*stride*/
      tensType,
      CUTENSOR_OP_IDENTITY/*applied to each element*/
    )
  );

  //get alignments of A,B,C
  uint32_t alignmentA, alignmentB, alignmentC;
  HANDLE_ERROR( cutensorGetAlignmentRequirement (
    &handle,
    d_A,
    &descA,
    &alignmentA
    )
  );
  HANDLE_ERROR( cutensorGetAlignmentRequirement (
    &handle,
    d_B,
    &descB,
    &alignmentB
    )
  );
  HANDLE_ERROR( cutensorGetAlignmentRequirement (
    &handle,
    d_C,
    &descC,
    &alignmentC
    )
  );

  //create descriptor of contraction
  cutensorContractionDescriptor_t desc;
  HANDLE_ERROR( cutensorInitContractionDescriptor( 
    &handle,
    &desc,
    &descA, modeA.data(), alignmentA,
    &descB, modeB.data(), alignmentB,
    &descC, modeC.data(), alignmentC,
    &descC, modeC.data(), alignmentC,
    computeType
    )
  );


  //determine algorithm
  cutensorContractionFind_t find;
  HANDLE_ERROR( cutensorInitContractionFind(
    &handle, 
    &find,
    CUTENSOR_ALGO_DEFAULT /*will allow internal heuristic to choose best approach*/    
    )
  );
  
  //query workspace
  size_t worksize = 0;
  HANDLE_ERROR( cutensorContractionGetWorkspace(
    &handle,
    &desc,
    &find,
    CUTENSOR_WORKSPACE_RECOMMENDED,
    &worksize
    )
  );

  //allocate workspace
  void *work = nullptr;
  if(worksize > 0)
  {
    if( cudaSuccess != cudaMalloc(&work, worksize) )
    {
      work = nullptr;
      worksize=0;
    }
  }

  //create contraction plan
  cutensorContractionPlan_t plan;
  HANDLE_ERROR( cutensorInitContractionPlan(
    &handle,
    &plan,
    &desc,
    &find,
    worksize
    )
  );

  cutensor_setup.stop<std::chrono::microseconds>("us");
  
  
  cutensorStatus_t err;
  
  Timer<> cutensor_contract("cutensor contract");
  //EXECUTE IT!
  err = cutensorContraction(
      &handle, &plan,
      &alpha, d_A, 
                     d_B,
      &beta, d_C,
                    d_C,
      work, worksize,
      0/*stream*/
  );

  cudaDeviceSynchronize();

  cutensor_contract.stop<std::chrono::microseconds>("us");
  
  if(err != CUTENSOR_STATUS_SUCCESS)
  {
    printf("ERROR: %s\n", cutensorGetErrorString(err));
  }

  trace_matrix<<<1,1>>>(d_D, d_C, dim);

  cudaMemcpy(res, d_D, sizeof(std::complex<double>), cudaMemcpyDeviceToHost);

  if(d_A)
    cudaFree(d_A);
  if(d_B)
    cudaFree(d_B);
  if(d_C)
    cudaFree(d_C);
  if(d_D)
    cudaFree(d_D);
  if(work) 
    cudaFree(work);
}
