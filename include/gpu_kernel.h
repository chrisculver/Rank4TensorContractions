#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

#include <complex>
#include <vector>

void single_index_contract(std::complex<double> *res, std::complex<double> *bpropMat, std::complex<double> *bsinkMat, long int dim);
void all_index_contract(std::complex<double> *res, std::complex<double> *bpropMat, std::complex<double> *bsinkMat, long int dim);
void contract(std::complex<double> *res, std::complex<double> *bpropMat, std::complex<double> *bsinkMat, long int dim);
void cuTensorContract(std::complex<double> *res, std::complex<double> *a, std::complex<double> *b, long int dim);
void contract4(std::complex<double> *res, std::complex<double> *a, std::complex<double> *b, std::complex<double> *c, std::complex<double> *d, long int dim);
void cuTensorContract4(std::complex<double> *res, std::complex<double> *a, std::complex<double> *b, std::complex<double> *c, std::complex<double> *d, long int dim);

#endif
