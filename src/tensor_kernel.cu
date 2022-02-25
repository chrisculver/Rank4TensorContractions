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
  cuDoubleComplex *d_tr;
  cudaMalloc((void **) &d_tr, sizeof(std::complex<double>));
  
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

  //create tensor descriptors 
  cutensorHandle_t handle;
  cutensorInit(&handle);
  
  CUTensor dA(A, handle, dim, modeA);
  CUTensor dB(B, handle, dim, modeB);
  CUTensor dC(C, handle, dim, modeC);
  CUTensor dD(D, handle, dim, modeD);
  CUTensor dAB(nullptr, handle, dim, modeAB);
  CUTensor dCD(nullptr, handle, dim, modeCD);
  CUTensor dMat(nullptr, handle, dim, modeMat);

  //create descriptor of contraction
  cutensorContractionDescriptor_t CdescAB, CdescCD, CdescMat;
  HANDLE_ERROR( cutensorInitContractionDescriptor( 
    &handle,
    &CdescAB,
    &(dA.desc), dA.modes.data(), dA.alignment,
    &(dB.desc), dB.modes.data(), dB.alignment,
    &(dAB.desc), dAB.modes.data(), dAB.alignment,
    &(dAB.desc), dAB.modes.data(), dAB.alignment,
    computeType
    )
  );
  HANDLE_ERROR( cutensorInitContractionDescriptor( 
    &handle,
    &CdescCD,
    &(dC.desc), dC.modes.data(), dC.alignment,
    &(dD.desc), dD.modes.data(), dD.alignment,
    &(dCD.desc), dCD.modes.data(), dCD.alignment,
    &(dCD.desc), dCD.modes.data(), dCD.alignment,
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
 
  // Multiple workspaces????

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

  cutensor_setup.stop<std::chrono::microseconds>("us");
  cutensorStatus_t err;
  
  Timer<> cutensor_contract("cutensor contract");
  //EXECUTE IT!
  err = cutensorContraction(
      &handle, &planAB,
      &alpha, dA.data, 
                     dB.data,
      &beta, dAB.data,
                    dAB.data,
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
      &alpha, dC.data, 
                     dD.data,
      &beta, dCD.data,
                    dCD.data,
      work, worksize,
      0/*stream*/
  );
  cudaDeviceSynchronize();
  if(err != CUTENSOR_STATUS_SUCCESS)
  {
    printf("ERROR: %s\n", cutensorGetErrorString(err));
  }

  
  //TODO try swapping modes of dCD
 
  dCD.change_contraction_modes(modeCD2);

  HANDLE_ERROR( cutensorInitContractionDescriptor( 
    &handle,
    &CdescMat,
    &(dAB.desc), dAB.modes.data(), dAB.alignment,
    &(dCD.desc), dCD.modes.data(), dCD.alignment,
    &(dMat.desc), dMat.modes.data(), dMat.alignment,
    &(dMat.desc), dMat.modes.data(), dMat.alignment,
    computeType
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
  
  


  err = cutensorContraction(
      &handle, &planMat,
      &alpha, dAB.data, 
                     dCD.data, //dCD2 doesn't have the data
      &beta, dMat.data,
                    dMat.data,
      work, worksize,
      0/*stream*/
  );
  cudaDeviceSynchronize();
  if(err != CUTENSOR_STATUS_SUCCESS)
  {
    printf("ERROR: %s\n", cutensorGetErrorString(err));
  }

  cutensor_contract.stop<std::chrono::microseconds>("us");
  

  trace_matrix<<<1,1>>>(d_tr, dMat.data, dim);

  cudaMemcpy(res, d_tr, sizeof(std::complex<double>), cudaMemcpyDeviceToHost);

  if(d_tr)
    cudaFree(d_tr);
  if(work) 
    cudaFree(work);
}
