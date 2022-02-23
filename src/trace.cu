#include "trace.h"


__global__ void trace_matrix(cuDoubleComplex *res, cuDoubleComplex *A, int dim)
{
  for(int i=0; i<dim; ++i)
    res[0]=cuCadd(res[0], A[i*dim + i]);
}


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


__global__ void contractAB(cuDoubleComplex *res, cuDoubleComplex *A, cuDoubleComplex *B, int dim)
{
  for(size_t a=0; a<dim; ++a)
  for(size_t b=0; b<dim; ++b)
  for(size_t c=0; c<dim; ++c)
  for(size_t d=0; d<dim; ++d)
  {
    for(size_t i=0; i<dim; ++i)
    for(size_t j=0; j<dim; ++j)
    {
      res[a*dim*dim*dim + b*dim*dim + c*dim + d]=cuCadd(res[a*dim*dim*dim + b*dim*dim + c*dim + d], cuCmul(A[a*dim*dim*dim + b*dim*dim + i*dim + j], B[j*dim*dim*dim + i*dim*dim + c*dim + d]));
    }
  }
}

__global__ void contractCD(cuDoubleComplex *res, cuDoubleComplex *C, cuDoubleComplex *D, int dim)
{
  for(size_t a=0; a<dim; ++a)
  for(size_t b=0; b<dim; ++b)
  for(size_t c=0; c<dim; ++c)
  for(size_t d=0; d<dim; ++d)
  {
    for(size_t i=0; i<dim; ++i)
    for(size_t j=0; j<dim; ++j)
    {
      res[a*dim*dim*dim + b*dim*dim + c*dim + d]=cuCadd(res[a*dim*dim*dim + b*dim*dim + c*dim + d], cuCmul(C[a*dim*dim*dim + b*dim*dim + i*dim + j], D[i*dim*dim*dim + j*dim*dim + c*dim + d]));
    }
  }
}
