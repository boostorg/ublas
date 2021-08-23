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
    std::cout << "% --------------------------- " << std::endl;
    auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor>( B, uplus1 );
    auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor>( A, uplus1 );
    for (auto& x: uexpr1.e.extents()) {
      std::cout << x << " ";
    }
    std::cout << std::endl;
    for (auto& x: uexpr2.e.extents()) {
      std::cout << x << " ";
    }

    std::cout << std::endl;
    std::cout << "% --------------------------- " << std::endl;

    // bexpr_uexpr = (s1+1) + (2+s2)
    auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor>( uexpr1, uexpr2, bplus );

    // bexpr_bexpr_uexpr = ((s1+1) + (2+s2)) - s2
    auto bexpr_bexpr_uexpr1 = ublas::detail::make_binary_tensor_expression<tensor>( bexpr_uexpr, B, bminus );

    auto ext = ublas::detail::retrieve_extents(bexpr_bexpr_uexpr1);
    for (auto& x: ext) {
      std::cout << x << " ";
    }
    std::cout << std::endl;

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

  // try {
  //   using value   = std::complex<boost::multiprecision::cpp_bin_float_double_extended>;
  //   using layout  = ublas::layout::last_order; // storage format
  //   using tensor  = ublas::tensor_dynamic<value,layout>;
  //   using shape   = typename tensor::extents_type;
  //   using span    = ublas::span<>;
  //   constexpr auto zeros = ublas::zeros<value,layout>{};


  //   // creates a four-dimensional tensor with extents 5,4,3 and 2
  //   // tensor A stores complex floating-point extended double precision numbers
  //   // according to the last-order storage format
  //   // and initializes it with the default value.

  //   //NOLINTNEXTLINE
  //   tensor t1 = zeros(5,4,3,2);
  //   auto B = t1(span(), span(), span(), span());

  //   // initializes the tensor with increasing values along the last-index
  //   // using a single-index
  //   auto vc = value(0,0);
  //   for(auto i = 0u; i < B.size(); ++i, vc += value(1,1))
  //     B[i] = vc;

  //   // formatted output
  //   std::cout << "% --------------------------- " << std::endl;
  //   std::cout << "% --------------------------- " << std::endl << std::endl;
  //   std::cout << "B=" << B << ";" << std::endl << std::endl;

  //   auto C = tensor(B.extents());
  //   // computes the complex conjugate of elements of B
  //   // using multi-index notation.
  //   for(auto i = 0u; i < B.size(0); ++i)
  //     for(auto j = 0u; j < B.size(1); ++j)
  //       for(auto k = 0u; k < B.size(2); ++k)
  //         for(auto l = 0u; l < B.size(3); ++l)
  //           C.at(i,j,k,l) = std::conj(B.at(i,j,k,l));

  //   std::cout << "% --------------------------- " << std::endl;
  //   std::cout << "% --------------------------- " << std::endl << std::endl;
  //   std::cout << "C=" << C << ";" << std::endl << std::endl;


  //   // // computes the complex conjugate of elements of B
  //   // // using iterators.
  //   auto D = tensor(B.extents());
  //   // // std::transform(B.begin(), B.end(), D.begin(), [](auto const& b){ return std::conj(b); });
  //   // std::cout << "% --------------------------- " << std::endl;
  //   // std::cout << "% --------------------------- " << std::endl << std::endl;
  //   // std::cout << "D=" << D << ";" << std::endl << std::endl;

  //   // reshaping tensors.
  //   auto new_extents = B.extents().base();
  //   std::next_permutation( new_extents.begin(), new_extents.end() );
  //   auto E = reshape( D, shape(new_extents)  );
  //   std::cout << "% --------------------------- " << std::endl;
  //   std::cout << "% --------------------------- " << std::endl << std::endl;
  //   std::cout << "E=" << E << ";" << std::endl << std::endl;


  // } catch (const std::exception& e) {
  //   std::cerr << "Cought exception " << e.what();
  //   std::cerr << "in the main function of access-tensor." << std::endl;
  // }
}
