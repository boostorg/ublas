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
//  And we acknowledge the support from all contributors.


#include <boost/numeric/ublas/tensor.hpp>
#include <boost/test/unit_test.hpp>


#include <algorithm>
#include <iostream>


#include "utility.hpp"

BOOST_AUTO_TEST_SUITE ( test_einstein_notation,
                      *boost::unit_test::depends_on("test_multi_index") )


using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;

//using test_types = zip<int>::with_t<boost::numeric::ublas::layout::first_order>;

BOOST_AUTO_TEST_CASE_TEMPLATE( test_einstein_multiplication, value,  test_types )
{
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;
    using tensor_type  = boost::numeric::ublas::dynamic_tensor<value_type,layout_type>;
    // NOLINTNEXTLINE(google-build-using-namespace)
    using namespace boost::numeric::ublas::index;
    
    {
        auto A = tensor_type{5,3};
        auto B = tensor_type{3,4};
        //  auto C = tensor_type{4,5,6};

        for(auto j = 0u; j < A.extents().at(1); ++j){
            for(auto i = 0u; i < A.extents().at(0); ++i){
                A.at( i,j ) = value_type( static_cast< inner_type_t<value_type> >(i+1) );
            }
        }

        for(auto j = 0u; j < B.extents().at(1); ++j){
          for(auto i = 0u; i < B.extents().at(0); ++i){
                B.at( i,j ) = value_type( static_cast< inner_type_t<value_type> >(i+1) );
          }
        }



        auto AB = A(_,_e) * B(_e,_);

        //  std::cout << "A = " << A << std::endl;
        //  std::cout << "B = " << B << std::endl;
        //  std::cout << "AB = " << AB << std::endl;

        for(auto j = 0u; j < AB.extents().at(1); ++j){
          for(auto i = 0u; i < AB.extents().at(0); ++i){
            auto e0  = B.extents().at(0);
            auto sum = std::div(e0*(e0+1),2);
            BOOST_CHECK_EQUAL( AB.at(i,j) , A.at(i,0)*value_type(sum.quot) );
          }
        }


    }


    {
        auto A = tensor_type{4,5,3};
        auto B = tensor_type{3,4,2};

        for(auto k = 0u; k < A.extents().at(2); ++k){
            for(auto j = 0u; j < A.extents().at(1); ++j){
                for(auto i = 0u; i < A.extents().at(0); ++i){
                    A.at( i,j,k ) = value_type( static_cast< inner_type_t<value_type> >(i+1) );
                }
            }
        }

        for(auto k = 0u; k < B.extents().at(2); ++k){
          for(auto j = 0u; j < B.extents().at(1); ++j){
                for(auto i = 0u; i < B.extents().at(0); ++i){
                    B.at( i,j,k ) = value_type( static_cast< inner_type_t<value_type> >(i+1) );
                }
          }
        }



        auto AB = A(_d,_,_f) * B(_f,_d,_);

        //  std::cout << "A = " << A << std::endl;
        //  std::cout << "B = " << B << std::endl;
        //  std::cout << "AB = " << AB << std::endl;
        // n*(n+1)/2;
        auto const nf = ( B.extents().at(0) * (B.extents().at(0)+1) / 2 );
        auto const nd = ( A.extents().at(0) * (A.extents().at(0)+1) / 2 );

        for(auto j = 0u; j < AB.extents().at(1); ++j){
          for(auto i = 0u; i < AB.extents().at(0); ++i){
                BOOST_CHECK_EQUAL( AB.at( i,j ) ,  value_type( static_cast< inner_type_t<value_type> >(nf * nd) ) );
          }
        }


    }


    {
        auto A = tensor_type{4,3};
        auto B = tensor_type{3,4,2};

        for(auto j = 0u; j < A.extents().at(1); ++j){
            for(auto i = 0u; i < A.extents().at(0); ++i){
                A.at( i,j ) = value_type( static_cast< inner_type_t<value_type> >(i+1) );
            }
        }


        for(auto k = 0u; k < B.extents().at(2); ++k){
          for(auto j = 0u; j < B.extents().at(1); ++j){
            for(auto i = 0u; i < B.extents().at(0); ++i){
                    B.at( i,j,k ) = value_type( static_cast< inner_type_t<value_type> >(i+1) );
            }
          }
        }

        auto AB = A(_d,_f) * B(_f,_d,_);

        // n*(n+1)/2;
        auto const nf = ( B.extents().at(0) * (B.extents().at(0)+1) / 2 );
        auto const nd = ( A.extents().at(0) * (A.extents().at(0)+1) / 2 );

        for(auto i = 0u; i < AB.extents().at(0); ++i){
            BOOST_CHECK_EQUAL ( AB.at( i  ) ,  value_type( static_cast< inner_type_t<value_type> >(nf * nd) ) );
        }

    }
}

BOOST_AUTO_TEST_SUITE_END()
