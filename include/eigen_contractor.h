#ifndef EIGEN_CONTRACTOR_H
#define EIGEN_CONTRACTOR_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#include <vector>
#include <complex>


std::complex<double> contract_two_tensors(
    const Eigen::Tensor<std::complex<double>, 4> &bProp,
    const Eigen::Tensor<std::complex<double>, 4> &bSink,
    std::vector<std::vector<int>> indices
    )
{
  Eigen::array<Eigen::IndexPair<int>, 4> idx = {
    Eigen::IndexPair<int>(indices[0][0],indices[0][1]),
    Eigen::IndexPair<int>(indices[1][0],indices[1][1]),
    Eigen::IndexPair<int>(indices[2][0],indices[2][1]),
    Eigen::IndexPair<int>(indices[3][0],indices[3][1])
  };

  Eigen::Tensor<std::complex<double>,0> res=bProp.contract(bSink, idx);

  return res(0);
}

#endif
