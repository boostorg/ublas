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
#include <boost/multiprecision/cpp_bin_float.hpp>

#include <ostream>

int main()
{
    using namespace boost::numeric::ublas;
    using namespace boost::multiprecision;

    {
      using value_t   = float;
      using format_t  = boost::numeric::ublas::layout::first_order; // storage format
      using tensor_t  = boost::numeric::ublas::dynamic_tensor<value_t,format_t>;

      // creates a three-dimensional tensor with extents 3,4 and 2
      // tensor A stores single-precision floating-point number according
      // to the first-order storage format

      auto A = tensor_t{3,4,2};

      // initializes the tensor with increasing values along the first-index
      // using a single index.
      auto vf = 1.0f;
      for(auto i = 0u; i < A.size(); ++i, vf += 1.0f)
          A[i] = vf;

      // formatted output
      std::cout << "% --------------------------- " << std::endl;
      std::cout << "% --------------------------- " << std::endl << std::endl;
      std::cout << "A=" << A << ";" << std::endl << std::endl;
    }


    {
      using value_t   = std::complex<boost::multiprecision::cpp_bin_float_double_extended>;
      using format_t  = boost::numeric::ublas::layout::last_order; // storage format
      using tensor_t  = boost::numeric::ublas::dynamic_tensor<value_t,format_t>;
      using shape_t   = typename tensor_t::extents_type;

      // creates a four-dimensional tensor with extents 5,4,3 and 2
      // tensor A stores complex floating-point extended double precision numbers
      // according to the last-order storage format
      // and initializes it with the default value.

      auto B = tensor_t(shape_t{5,4,3,2},value_t{});

      // initializes the tensor with increasing values along the last-index
      // using a single-index
      auto vc = value_t(0,0);
      for(auto i = 0u; i < B.size(); ++i, vc += value_t(1,1))
          B[i] = vc;

      // formatted output
      std::cout << "% --------------------------- " << std::endl;
      std::cout << "% --------------------------- " << std::endl << std::endl;
      std::cout << "B=" << B << ";" << std::endl << std::endl;


      auto C = tensor_t(B.extents());
      // computes the complex conjugate of elements of B
      // using multi-index notation.
      for(auto i = 0u; i < B.size(0); ++i)
          for(auto j = 0u; j < B.size(1); ++j)
              for(auto k = 0u; k < B.size(2); ++k)
                  for(auto l = 0u; l < B.size(3); ++l)
                      C.at(i,j,k,l) = std::conj(B.at(i,j,k,l));

      std::cout << "% --------------------------- " << std::endl;
      std::cout << "% --------------------------- " << std::endl << std::endl;
      std::cout << "C=" << C << ";" << std::endl << std::endl;



      // computes the complex conjugate of elements of B
      // using iterators.
      auto D = tensor_t(B.extents());
      std::transform(B.begin(), B.end(), D.begin(), [](auto const& b){ return std::conj(b); });
      std::cout << "% --------------------------- " << std::endl;
      std::cout << "% --------------------------- " << std::endl << std::endl;
      std::cout << "D=" << D << ";" << std::endl << std::endl;

      // reshaping tensors.
      auto new_extents = B.extents().base();
      std::next_permutation( new_extents.begin(), new_extents.end() );
      D.reshape( extents<>(new_extents)  );
      std::cout << "% --------------------------- " << std::endl;
      std::cout << "% --------------------------- " << std::endl << std::endl;
      std::cout << "newD=" << D << ";" << std::endl << std::endl;
    }
}
