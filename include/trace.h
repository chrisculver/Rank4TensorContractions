#ifndef TRACE_H
#define TRACE_H

#include <cuComplex.h>

__global__ void trace_matrix(cuDoubleComplex *res, cuDoubleComplex *A, int dim);
__global__ void trace_rank6(cuDoubleComplex *res, cuDoubleComplex *A, int dim);
__global__ void contract(cuDoubleComplex *res, cuDoubleComplex *A, cuDoubleComplex *B, int dim);
__global__ void contractAB(cuDoubleComplex *res, cuDoubleComplex *A, cuDoubleComplex *B, int dim);
__global__ void contractCD(cuDoubleComplex *res, cuDoubleComplex *C, cuDoubleComplex *D, int dim);

#endif
