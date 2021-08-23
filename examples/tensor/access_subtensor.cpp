//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#include <boost/numeric/ublas/tensor.hpp>
// #include <boost/multiprecision/cpp_bin_float.hpp>

#include <ostream>

//NOLINTNEXTLINE
int main()
{
  namespace ublas = boost::numeric::ublas;

  try {
    using value   = float;
    using layout  = ublas::layout::first_order; // storage format
    using tensor  = ublas::tensor_dynamic<value,layout>;
    using span    = ublas::span<>;
    using subtensor = typename tensor::subtensor_type;
    auto uplus1 = [](auto const& a){return a + value(1); };
    auto bplus  = std::plus <value>{};
    auto bminus = std::minus<value>{};
    constexpr auto ones  = ublas::ones<value,layout>{};


    // creates a three-dimensional tensor with extents 3,4 and 2
    // tensor A stores single-precision floating-point number according
    // to the first-order storage format

    tensor t1 = ones(3,3,2);
    value cnt(0);
    for (auto i = 0u; i < t1.size(0); i++) {
      for (auto j = 0u; j < t1.size(1); j++) {
        for (auto k = 0u; k < t1.size(2); k++) {
          t1(i,j,k) = cnt;
          cnt+= value(1);
        }
      }
    }
    auto A = t1 (span(1,1,2), span(0,2,2), span());

    auto B = subtensor(A);

    tensor t2 = ones(2,2,2) + A + B;
    auto t3 = ublas::inner_prod(B, t2);
    // // // formatted output
    std::cout << "% --------------------------- " << std::endl << std::endl;
    std::cout << "t1=" << t1 << ";" << std::endl << std::endl;
    std::cout << "B=" << B << ";" << std::endl << std::endl;
    std::cout << "t2=" << t2 << ";" << std::endl << std::endl;
    std::cout << "t3=" << t3 << ";" << std::endl << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Cought exception " << e.what();
    std::cerr << " in the main function of access-tensor." << std::endl;
  }
}
