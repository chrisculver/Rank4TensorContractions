#include "eigen_contractor.h"
#include "gpu_kernel.h"
#include "timer.h"

#include <unsupported/Eigen/CXX11/Tensor>

#include <iostream>
#include <complex>

using namespace std;
using cd = complex<double>;
using Tensor4 = Eigen::Tensor<std::complex<double>,4>;

cd compute_two_tensors_gpu(const Tensor4 &tensA, const Tensor4 &tensB);
cd full_trace_gpu(const Tensor4 &tensA, const Tensor4 &tensB);
cd full_contract_gpu(const Tensor4 &tensA, const Tensor4 &tensB);
cd explicit_contract(const Tensor4 &A, const Tensor4 &B);
cd explicit_contract(const Tensor4 &A, const Tensor4 &B, const Tensor4 &C, const Tensor4 &D);

int main()
{
  const int N=12;

  Tensor4 A(N,N,N,N),B(N,N,N,N),C(N,N,N,N),D(N,N,N,N); 
  A.setRandom();
  B.setRandom();
  C.setRandom();
  D.setRandom(); 


  std::vector<std::vector<int>> diagram;
  diagram = std::vector<std::vector<int>>{{0,3},{1,2},{2,1},{3,0}};
 
  Timer<> cpu_timer("Brute for CPU contract");
  std::cout << explicit_contract(A,B) << std::endl;
  cpu_timer.stop<std::chrono::microseconds>("us");
  cout << endl;
  Timer<> ecpu_timer("Eigen CPU contract A,B");  
  cd cpu_res = contract_two_tensors(A,B,diagram);
  std::cout << cpu_res << std::endl;
  ecpu_timer.stop<std::chrono::microseconds>("us");
  
  cout << endl;
  Timer<> gpu_timer("GPU contract one index");
  std::cout << compute_two_tensors_gpu(A,B) << std::endl;
  gpu_timer.stop<std::chrono::microseconds>("us");

  cout << endl;
  Timer<> gpu_trace_timer("GPU full trace");
  std::cout << full_trace_gpu(A,B) << std::endl;
  gpu_trace_timer.stop<std::chrono::microseconds>("us");
  
  
  cout << endl;
  Timer<> gpu_c_timer("GPU full contract");
  std::cout << full_contract_gpu(A,B) << std::endl;
  gpu_c_timer.stop<std::chrono::microseconds>("us");


  cout << endl << endl << "============Contracting 4 rank4  tensors===========" << endl;
  Timer<> cpu4_timer("Brute force CPU");
  cout << explicit_contract(A,B,C,D) << endl;
  cpu4_timer.stop<std::chrono::milliseconds>("ms");
  cout << endl;

  Timer<> ecpu4_timer("Eigen CPU contract ABCD");
  cd ecpu_res = contract_four_tensors(A,B,C,D);
  cout << ecpu_res << endl;
  ecpu4_timer.stop<std::chrono::microseconds>("us");
  cout << endl;

  return 0;
}




cd compute_two_tensors_gpu(const Tensor4 &tensA, const Tensor4 &tensB)
{
  long int dim = tensA.dimensions()[0];
  long int tdim = dim*dim*dim*dim;

  cd *A = new cd[tdim];
  cd *B = new cd[tdim];
  cd res(0.,0.);


  for(size_t i=0; i<dim; ++i)
  for(size_t j=0; j<dim; ++j)
  for(size_t k=0; k<dim; ++k)
  for(size_t l=0; l<dim; ++l)
  {
    A[i*dim*dim*dim + j*dim*dim + k*dim + l] = tensA(l,k,j,i);
    B[i*dim*dim*dim + j*dim*dim + k*dim + l] = tensB(l,k,j,i);
  }

  std::complex<double> *GPUres = (std::complex<double> *)malloc(((long int)sizeof(std::complex<double>))*(dim*dim*dim*dim*dim*dim));
  
  Timer<> gpu_timer("GPU single index");
  single_index_contract(GPUres, A, B, dim);
  gpu_timer.stop<std::chrono::microseconds>("us");



  Timer<> cpu_trace("CPU trace");
  for(int i=0; i<dim; ++i)
  for(int j=0; j<dim; ++j)
  for(int k=0; k<dim; ++k)
    res+=GPUres[i*dim*dim*dim*dim*dim + j*dim*dim*dim*dim + k*dim*dim*dim + k*dim*dim + j*dim + i];
  
  cpu_trace.stop<std::chrono::microseconds>("us");
  
  
  
  free(GPUres);
  
  delete A;
  delete B;

  return res;
}

cd explicit_contract(const Tensor4 &A, const Tensor4 &B)
{
  long int dim = A.dimensions()[0];
  cd res(0.,0.);
  
  for(size_t i=0; i<dim; ++i)
  for(size_t j=0; j<dim; ++j)
  for(size_t k=0; k<dim; ++k)
  for(size_t l=0; l<dim; ++l)
  {
    res+=A(i,j,k,l)*B(l,k,j,i);
  }

  return res;
}


cd explicit_contract(const Tensor4 &A, const Tensor4 &B, const Tensor4 &C, const Tensor4 &D)
{
  long int dim = A.dimensions()[0];
  cd res(0.,0.);
 
  Tensor4 AB(dim,dim,dim,dim), CD(dim,dim,dim,dim);
  AB.setZero();
  CD.setZero();

  for(size_t a=0; a<dim; ++a)
  for(size_t b=0; b<dim; ++b)
  for(size_t c=0; c<dim; ++c)
  for(size_t d=0; d<dim; ++d)
  {
    for(size_t i=0; i<dim; ++i)
    for(size_t j=0; j<dim; ++j)
    {
      AB(a,b,c,d)+=A(a,b,i,j)*B(j,i,c,d);
      CD(a,b,c,d)+=C(a,b,i,j)*D(i,j,c,d);
    }
  }

  for(size_t a=0; a<dim; ++a)
  for(size_t b=0; b<dim; ++b)
  for(size_t c=0; c<dim; ++c)
  for(size_t d=0; d<dim; ++d)
  {
    res+=AB(a,b,c,d)*CD(d,c,b,a);
  }

  return res;
}




cd full_trace_gpu(const Tensor4 &tensA, const Tensor4 &tensB)
{
  long int dim = tensA.dimensions()[0];
  long int tdim = dim*dim*dim*dim;

  cd *A = new cd[tdim];
  cd *B = new cd[tdim];
  cd res(0.,0.);


  for(size_t i=0; i<dim; ++i)
  for(size_t j=0; j<dim; ++j)
  for(size_t k=0; k<dim; ++k)
  for(size_t l=0; l<dim; ++l)
  {
    A[i*dim*dim*dim + j*dim*dim + k*dim + l] = tensA(l,k,j,i);
    B[i*dim*dim*dim + j*dim*dim + k*dim + l] = tensB(l,k,j,i);
  }

  std::complex<double> *GPUres = (std::complex<double> *)malloc(((long int)sizeof(std::complex<double>))*(1));
 
  all_index_contract(GPUres, A, B, dim);
 
  res=GPUres[0];

  free(GPUres);
  
  delete A;
  delete B;
  
  return res;
}


cd full_contract_gpu(const Tensor4 &tensA, const Tensor4 &tensB)
{
  long int dim = tensA.dimensions()[0];
  long int tdim = dim*dim*dim*dim;

  cd *A = new cd[tdim];
  cd *B = new cd[tdim];
  cd res(0.,0.);


  for(size_t i=0; i<dim; ++i)
  for(size_t j=0; j<dim; ++j)
  for(size_t k=0; k<dim; ++k)
  for(size_t l=0; l<dim; ++l)
  {
    A[i*dim*dim*dim + j*dim*dim + k*dim + l] = tensA(l,k,j,i);
    B[i*dim*dim*dim + j*dim*dim + k*dim + l] = tensB(l,k,j,i);
  }

  std::complex<double> *GPUres = (std::complex<double> *)malloc(((long int)sizeof(std::complex<double>))*(1));
 
  contract(GPUres, A, B, dim);
 
  res=GPUres[0];

  free(GPUres);
  
  delete A;
  delete B;
  
  return res;
}
