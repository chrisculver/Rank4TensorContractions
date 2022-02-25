#include "tensor_kernel.h"
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

/* 
   specialization of CUTensor information for BB->BB scattering tensors
   IMPORTANT THINGS
      *Assume every mode has the same dimensions
      *Manages GPU memory for the tensor
      *Computes the cutensor desc 
      *Computes the cutensor alignment

   You can change which modes of the tensor will be contracted with change_contraction_modes()
*/
class CUTensor
{
  public: 
    cutensorHandle_t &handle;
    cuDoubleComplex *data;
    long int dim;

    std::vector<int> modes;
    std::vector<int64_t> extents;
    cutensorTensorDescriptor_t desc;
    uint32_t alignment;

    CUTensor(std::complex<double> *host_data, cutensorHandle_t &h, long int dim, std::vector<int> m):
      handle(h), modes(m)
    {
      long int tensor_size = pow(dim, modes.size())*sizeof(std::complex<double>);
      cudaMalloc((void **) &data, tensor_size);
      if(host_data)
        cudaMemcpy(data, host_data, tensor_size, cudaMemcpyHostToDevice);
      else
        cudaMemset(data, 0, tensor_size);

      for(auto m: modes)
        extents.push_back(dim);

      init_descriptor();
      init_alignment();


    }

    ~CUTensor()
    {
      if(data)
        cudaFree(data);
    }

    void change_contraction_modes(std::vector<int> m)
    {
      if( m.size() != modes.size() )
      {
        printf("Can't change modes while changing rank of tensor!");
        exit(-1);
      }
      modes=m;
      // I **think** these need to get called to update the descriptor and alignment
      init_descriptor();
      init_alignment();
    }

    void init_descriptor()
    {
      HANDLE_ERROR( cutensorInitTensorDescriptor(
        &handle,
        &desc,
        modes.size(),
        extents.data(),
        NULL,//stride
        CUDA_C_64F,
        CUTENSOR_OP_IDENTITY//applied to each element
        )
      );
    }

    void init_alignment()
    {
      HANDLE_ERROR( cutensorGetAlignmentRequirement (
        &handle,
        data,
        &desc,
        &alignment
        )
      );
    }
};



void cuTensorContract(std::complex<double> *res, std::complex<double> *A,std::complex<double> *B, long int dim)
{
  // allocate device memory and copy tensors
  cuDoubleComplex *d_D;
  cudaMalloc((void **) &d_D, sizeof(std::complex<double>));
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


  //create tensor descriptors 
  cutensorHandle_t handle;
  cutensorInit(&handle);

  CUTensor dA(A, handle, dim, modeA);
  CUTensor dB(B, handle, dim, modeB);
  CUTensor dC(nullptr, handle, dim, modeC);
  

  //create descriptor of contraction
  cutensorContractionDescriptor_t desc;
  HANDLE_ERROR( cutensorInitContractionDescriptor( 
    &handle,
    &desc,
    &(dA.desc), dA.modes.data(), dA.alignment,
    &(dB.desc), dB.modes.data(), dB.alignment,
    &(dC.desc), dC.modes.data(), dC.alignment,
    &(dC.desc), dC.modes.data(), dC.alignment,
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
      &alpha, dA.data, 
                     dB.data,
      &beta, dC.data,
                    dC.data,
      work, worksize,
      0/*stream*/
  );

  cudaDeviceSynchronize();

  cutensor_contract.stop<std::chrono::microseconds>("us");
  
  if(err != CUTENSOR_STATUS_SUCCESS)
  {
    printf("ERROR: %s\n", cutensorGetErrorString(err));
  }

  trace_matrix<<<1,1>>>(d_D, dC.data, dim);

  cudaMemcpy(res, d_D, sizeof(std::complex<double>), cudaMemcpyDeviceToHost);

  if(d_D)
    cudaFree(d_D);
  if(work) 
    cudaFree(work);
}



void cuTensorContract4(std::complex<double> *res, std::complex<double> *A,std::complex<double> *B, 
                       std::complex<double> *C, std::complex<double> *D, long int dim)
{
  // allocate device memory and copy tensors
  long int bTensor_size = dim*dim*dim*dim*sizeof(std::complex<double>);

  cuDoubleComplex *d_A, *d_B, *d_C, *d_D, *d_AB, *d_CD, *d_Mat, *d_tr;
  cudaMalloc((void **) &d_A, bTensor_size);
  cudaMalloc((void **) &d_B, bTensor_size);
  cudaMalloc((void **) &d_C, bTensor_size);
  cudaMalloc((void **) &d_D, bTensor_size);
  cudaMalloc((void **) &d_AB, bTensor_size);
  cudaMalloc((void **) &d_CD, bTensor_size);
  cudaMalloc((void **) &d_Mat, dim*dim*sizeof(std::complex<double>));
  cudaMalloc((void **) &d_tr, sizeof(std::complex<double>));

  cudaMemcpy(d_A, A, bTensor_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, bTensor_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, bTensor_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, D, bTensor_size, cudaMemcpyHostToDevice);
  
  cudaMemset(d_AB, 0, bTensor_size);
  cudaMemset(d_CD, 0, bTensor_size);
  cudaMemset(d_Mat, 0, dim*dim*sizeof(std::complex<double>));
  cudaMemset(d_tr, 0, sizeof(std::complex<double>));


  //types of cuTensor
  Timer<> cutensor_setup("cutensor setup time");
  cudaDataType_t tensType = CUDA_C_64F;
  cutensorComputeType_t computeType = CUTENSOR_COMPUTE_64F;

  typedef float floatTypeCompute;

  cuDoubleComplex alpha = make_cuDoubleComplex(1.0,0.0);
  cuDoubleComplex beta = make_cuDoubleComplex(0.0,0.0);

  //modes of tensors
  std::vector<int> modeA{'a','b','i','j'};
  std::vector<int> modeB{'j','i','c','d'};
  std::vector<int> modeC{'a','b','i','j'};
  std::vector<int> modeD{'i','j','c','d'};
  std::vector<int> modeAB{'a','b','c','d'};
  std::vector<int> modeCD{'a','b','c','d'};
  std::vector<int> modeCD2{'d','c','b','e'};
  std::vector<int> modeMat{'a','e'};

  int nmodeA = modeA.size();
  int nmodeB = modeB.size();
  int nmodeC = modeC.size();
  int nmodeD = modeD.size();
  int nmodeAB = modeAB.size();
  int nmodeCD = modeCD.size();
  int nmodeCD2 = modeCD2.size();
  int nmodeMat = modeMat.size();

  //extents of modes
  std::unordered_map<int, int64_t> extent;
  extent['i']=dim;
  extent['j']=dim;
  extent['a']=dim;
  extent['b']=dim;
  extent['c']=dim;
  extent['d']=dim;
  extent['e']=dim;

  std::vector<int64_t> extentA, extentB, extentC, extentD, extentAB, extentCD, extentCD2, extentMat;
  for(auto mode: modeA)
    extentA.push_back(extent[mode]);
  for(auto mode: modeB)
    extentB.push_back(extent[mode]);
  for(auto mode: modeC)
    extentC.push_back(extent[mode]); 
  for(auto mode: modeD)
    extentD.push_back(extent[mode]); 
  for(auto mode: modeAB)
    extentAB.push_back(extent[mode]); 
  for(auto mode: modeCD)
    extentCD.push_back(extent[mode]); 
  for(auto mode: modeCD2)
    extentCD2.push_back(extent[mode]); 
  for(auto mode: modeMat)
    extentMat.push_back(extent[mode]); 

  //create tensor descriptors 
  cutensorHandle_t handle;
  cutensorInit(&handle);

  cutensorTensorDescriptor_t descA, descB, descC, descD, descAB, descCD, descCD2, descMat;
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
  HANDLE_ERROR( cutensorInitTensorDescriptor(
      &handle,
      &descD,
      nmodeD,
      extentD.data(),
      NULL,/*stride*/
      tensType,
      CUTENSOR_OP_IDENTITY/*applied to each element*/
    )
  );
  HANDLE_ERROR( cutensorInitTensorDescriptor(
      &handle,
      &descAB,
      nmodeAB,
      extentAB.data(),
      NULL,/*stride*/
      tensType,
      CUTENSOR_OP_IDENTITY/*applied to each element*/
    )
  );
  HANDLE_ERROR( cutensorInitTensorDescriptor(
      &handle,
      &descCD,
      nmodeCD,
      extentCD.data(),
      NULL,/*stride*/
      tensType,
      CUTENSOR_OP_IDENTITY/*applied to each element*/
    )
  );
  HANDLE_ERROR( cutensorInitTensorDescriptor(
      &handle,
      &descCD2,
      nmodeCD2,
      extentCD2.data(),
      NULL,/*stride*/
      tensType,
      CUTENSOR_OP_IDENTITY/*applied to each element*/
    )
  );
  HANDLE_ERROR( cutensorInitTensorDescriptor(
      &handle,
      &descMat,
      nmodeMat,
      extentMat.data(),
      NULL,/*stride*/
      tensType,
      CUTENSOR_OP_IDENTITY/*applied to each element*/
    )
  );

  //get alignments of A,B,C
  uint32_t alignmentA, alignmentB, alignmentC, alignmentD, alignmentAB, alignmentCD, alignmentCD2, alignmentMat;
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
  HANDLE_ERROR( cutensorGetAlignmentRequirement (
    &handle,
    d_D,
    &descD,
    &alignmentD
    )
  );
  HANDLE_ERROR( cutensorGetAlignmentRequirement (
    &handle,
    d_AB,
    &descAB,
    &alignmentAB
    )
  );
  HANDLE_ERROR( cutensorGetAlignmentRequirement (
    &handle,
    d_CD,
    &descCD,
    &alignmentCD
    )
  );
  HANDLE_ERROR( cutensorGetAlignmentRequirement (
    &handle,
    d_CD,
    &descCD2,
    &alignmentCD2
    )
  );
  HANDLE_ERROR( cutensorGetAlignmentRequirement (
    &handle,
    d_Mat,
    &descMat,
    &alignmentMat
    )
  );

  //create descriptor of contraction
  cutensorContractionDescriptor_t CdescAB, CdescCD, CdescMat;
  HANDLE_ERROR( cutensorInitContractionDescriptor( 
    &handle,
    &CdescAB,
    &descA, modeA.data(), alignmentA,
    &descB, modeB.data(), alignmentB,
    &descAB, modeAB.data(), alignmentAB,
    &descAB, modeAB.data(), alignmentAB,
    computeType
    )
  );
  HANDLE_ERROR( cutensorInitContractionDescriptor( 
    &handle,
    &CdescCD,
    &descC, modeC.data(), alignmentC,
    &descD, modeD.data(), alignmentD,
    &descCD, modeCD.data(), alignmentCD,
    &descCD, modeCD.data(), alignmentCD,
    computeType
    )
  );
  HANDLE_ERROR( cutensorInitContractionDescriptor( 
    &handle,
    &CdescMat,
    &descAB, modeAB.data(), alignmentAB,
    &descCD2, modeCD2.data(), alignmentCD2,
    &descMat, modeMat.data(), alignmentMat,
    &descMat, modeMat.data(), alignmentMat,
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
    &CdescAB,
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
  cutensorContractionPlan_t planAB, planCD, planMat;
  HANDLE_ERROR( cutensorInitContractionPlan(
    &handle,
    &planAB,
    &CdescAB,
    &find,
    worksize
    )
  );
  HANDLE_ERROR( cutensorInitContractionPlan(
    &handle,
    &planCD,
    &CdescCD,
    &find,
    worksize
    )
  );
  HANDLE_ERROR( cutensorInitContractionPlan(
    &handle,
    &planMat,
    &CdescMat,
    &find,
    worksize
    )
  );

  cutensor_setup.stop<std::chrono::microseconds>("us");
  cutensorStatus_t err;
  
  Timer<> cutensor_contract("cutensor contract");
  //EXECUTE IT!
  err = cutensorContraction(
      &handle, &planAB,
      &alpha, d_A, 
                     d_B,
      &beta, d_AB,
                    d_AB,
      work, worksize,
      0/*stream*/
  );
  cudaDeviceSynchronize();
  if(err != CUTENSOR_STATUS_SUCCESS)
  {
    printf("ERROR: %s\n", cutensorGetErrorString(err));
  }
  err = cutensorContraction(
      &handle, &planCD,
      &alpha, d_C, 
                     d_D,
      &beta, d_CD,
                    d_CD,
      work, worksize,
      0/*stream*/
  );
  cudaDeviceSynchronize();
  if(err != CUTENSOR_STATUS_SUCCESS)
  {
    printf("ERROR: %s\n", cutensorGetErrorString(err));
  }
  err = cutensorContraction(
      &handle, &planMat,
      &alpha, d_AB, 
                     d_CD,
      &beta, d_Mat,
                    d_Mat,
      work, worksize,
      0/*stream*/
  );
  cudaDeviceSynchronize();
  if(err != CUTENSOR_STATUS_SUCCESS)
  {
    printf("ERROR: %s\n", cutensorGetErrorString(err));
  }

  cutensor_contract.stop<std::chrono::microseconds>("us");
  

  trace_matrix<<<1,1>>>(d_tr, d_Mat, dim);

  cudaMemcpy(res, d_tr, sizeof(std::complex<double>), cudaMemcpyDeviceToHost);

  if(d_A)
    cudaFree(d_A);
  if(d_B)
    cudaFree(d_B);
  if(d_C)
    cudaFree(d_C);
  if(d_D)
    cudaFree(d_D);
  if(d_AB)
    cudaFree(d_AB);
  if(d_CD)
    cudaFree(d_CD);
  if(d_Mat)
    cudaFree(d_Mat);
  if(d_tr)
    cudaFree(d_tr);
  if(work) 
    cudaFree(work);
}
