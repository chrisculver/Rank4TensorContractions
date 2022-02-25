#ifndef TENSOR_KERNEL_H
#define TENSOR_KERNEL_H

#include <complex>
#include <vector>

void cuTensorContract(std::complex<double> *res, std::complex<double> *a, std::complex<double> *b, long int dim);
void cuTensorContract4(std::complex<double> *res, std::complex<double> *a, std::complex<double> *b, std::complex<double> *c, std::complex<double> *d, long int dim);

#endif
