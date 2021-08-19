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
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <ostream>

int main()
{
  namespace ublas = boost::numeric::ublas;
  using value     = float;
  using tensor    = ublas::tensor_dynamic<value>;
  using matrix    = ublas::matrix<value>;
  using vector    = ublas::vector<value>;
  using shape     = tensor::extents_type;
  using span      = ublas::span<>;

  try {


    auto t1 = tensor{3,4,2};
    auto A = t1(span(), span(), span());

    tensor B = A = 2;

    if (A == B) {
      std::cout << "Equal" << std::endl;
    }

    // formatted output
    std::cout << "% --------------------------- " << std::endl;
    std::cout << "% --------------------------- " << std::endl << std::endl;
    std::cout << "A=" << A << ";" << std::endl << std::endl;

    B = 1;
    A = B + 1;

    std::cout << "% --------------------------- " << std::endl;
    std::cout << "% --------------------------- " << std::endl << std::endl;
    std::cout << "A=" << A << ";" << std::endl << std::endl;

    auto n = shape{3,4};
    auto D = matrix(n[0],n[1],1);
    auto e = vector(n[1],1);
    auto f = vector(n[0],2);

    // Calling constructor with
    // vector expression templates
    tensor C = 2*f;
    // formatted output
    std::cout << "% --------------------------- " << std::endl;
    std::cout << "% --------------------------- " << std::endl << std::endl;
    std::cout << "C=" << C << ";" << std::endl << std::endl;


    // Calling overloaded operators
    // and mixing simple tensor and matrix expression templates
    tensor F = 3*C + 4*prod(2*D,e);

    // formatted output
    std::cout << "% --------------------------- " << std::endl;
    std::cout << "% --------------------------- " << std::endl << std::endl;
    std::cout << "F=" << F << ";" << std::endl << std::endl;

  }  catch (const std::exception& e) {
    std::cerr << "Cought exception " << e.what();
    std::cerr << "in the main function of simple expression." << std::endl;
  }

}
