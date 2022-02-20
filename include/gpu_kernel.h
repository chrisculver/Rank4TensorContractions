#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

#include <complex>
#include <vector>

void single_index_contract(std::complex<double> *res, std::complex<double> *bpropMat, std::complex<double> *bsinkMat, long int dim);
void all_index_contract(std::complex<double> *res, std::complex<double> *bpropMat, std::complex<double> *bsinkMat, long int dim);
void contract(std::complex<double> *res, std::complex<double> *bpropMat, std::complex<double> *bsinkMat, long int dim);
void contract4(std::complex<double> *res, std::complex<double> *A, std::complex<double> *B, std::complex<double> *C, std::complex<double> *D, long int dim);

#endif
