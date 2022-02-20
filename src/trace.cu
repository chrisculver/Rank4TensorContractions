#include "trace.h"

__global__ void trace_rank6(cuDoubleComplex *res, cuDoubleComplex *A, int dim)
{
  int index = threadIdx.x;
  int stride = blockDim.x;

  for(int i=0; i<dim; ++i)
  for(int j=0; j<dim; ++j)
  for(int k=0; k<dim; ++k)
  {
    res[0]=cuCadd(res[0],A[i*dim*dim*dim*dim*dim + j*dim*dim*dim*dim + k*dim*dim*dim + k*dim*dim + j*dim + i]);
  }
}


__global__ void contract(cuDoubleComplex *res, cuDoubleComplex *A, cuDoubleComplex *B, int dim)
{
  for(int i=0; i<dim; ++i)
  for(int j=0; j<dim; ++j)
  for(int k=0; k<dim; ++k)
  for(int l=0; l<dim; ++l)
  {
    res[0]=cuCadd(res[0], cuCmul(A[i*dim*dim*dim + j*dim*dim + k*dim + l], B[l*dim*dim*dim + k*dim*dim + j*dim + i]));
  }
}
