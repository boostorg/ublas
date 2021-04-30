//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//
//  And we acknowledge the support from all contributors.


#include <iostream>
#include <algorithm>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <boost/test/unit_test.hpp>

#include "utility.hpp"

// BOOST_AUTO_TEST_SUITE ( test_tensor_functions, * boost::unit_test::depends_on("test_tensor_contraction") )
BOOST_AUTO_TEST_SUITE ( test_tensor_functions)


using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;

//using test_types = zip<int>::with_t<boost::numeric::ublas::layout::first_order>;


struct fixture
{
    using dynamic_extents_type = boost::numeric::ublas::extents<>;
    fixture()
      : extents {
          dynamic_extents_type{1,1}, // 1
          dynamic_extents_type{2,3}, // 2
          dynamic_extents_type{2,3,1}, // 3
          dynamic_extents_type{4,2,3}, // 4
          dynamic_extents_type{4,2,3,5}} // 5
    {
    }

    std::vector<dynamic_extents_type> extents;
};




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_prod_vector, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;
    using tensor_type  = ublas::dynamic_tensor<value_type,layout_type>;
    using vector_type  = typename tensor_type::vector_type;


    for(auto const& n : extents){

        auto a = tensor_type(n, value_type{2});

        for(auto m = 0u; m < n.size(); ++m){

            auto b = vector_type  (n[m], value_type{1} );

            auto c = ublas::prod(a, b, m+1);

            for(auto i = 0u; i < c.size(); ++i)
                BOOST_CHECK_EQUAL( c[i] , value_type( static_cast< inner_type_t<value_type> >(n[m]) ) * a[i] );

        }
    }
  auto n = extents[4];
  auto a = tensor_type(n, value_type{2});
  auto b = vector_type(n[0], value_type{1});

  auto zero_rank_empty_tensor = tensor_type{};
  auto empty = vector_type{};

  BOOST_CHECK_THROW(prod(a, b, 0), std::length_error);
  BOOST_CHECK_THROW(prod(a, b, 9), std::length_error);
  BOOST_CHECK_THROW(prod(zero_rank_empty_tensor, b, 1), std::length_error);
  BOOST_CHECK_THROW(prod(a, empty, 2), std::length_error);

}

BOOST_AUTO_TEST_CASE( test_tensor_prod_vector_exception )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = float;
    using layout_type  = ublas::layout::first_order;
    using d_tensor_type  = ublas::dynamic_tensor<value_type,layout_type>;
    using vector_type  = typename d_tensor_type::vector_type;

    auto t1 = d_tensor_type{ublas::extents<>{},1.f};
    auto v1 = vector_type{3,value_type{1}};

    BOOST_REQUIRE_THROW(prod(t1,v1,0),std::length_error);
    BOOST_REQUIRE_THROW(prod(t1,v1,1),std::length_error);
    BOOST_REQUIRE_THROW(prod(t1,v1,3),std::length_error);
}




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_prod_matrix, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;
    using tensor_type  = ublas::dynamic_tensor<value_type,layout_type>;
    using matrix_type  = typename tensor_type::matrix_type;


    for(auto const& n : extents) {

        auto a = tensor_type(n, value_type{2});

        for(auto m = 0u; m < n.size(); ++m){

            auto b  = matrix_type  ( n[m], n[m], value_type{1} );

            auto c = ublas::prod(a, b, m+1);

            for(auto i = 0u; i < c.size(); ++i)
                BOOST_CHECK_EQUAL( c[i] , value_type( static_cast< inner_type_t<value_type> >(n[m]) ) * a[i] );

        }
    }

  auto n = extents[4];
  auto a = tensor_type(n, value_type{2});
  auto b = matrix_type(n[0], n[0], value_type{1});

  auto zero_rank_empty_tensor = tensor_type{};
  auto empty = matrix_type{};

  BOOST_CHECK_THROW(prod(a, b, 0), std::length_error);
  BOOST_CHECK_THROW(prod(a, b, 9), std::length_error);
  BOOST_CHECK_THROW(prod(zero_rank_empty_tensor, b, 1), std::length_error);
  BOOST_CHECK_THROW(prod(a, empty, 2), std::length_error);
}

BOOST_AUTO_TEST_CASE( test_tensor_prod_matrix_exception )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = float;
    using layout_type  = ublas::layout::first_order;
    using d_extents_type = ublas::extents<>;
    using d_tensor_type  = ublas::dynamic_tensor<value_type,layout_type>;
    using matrix_type  = typename d_tensor_type::matrix_type;

    auto t1 = d_tensor_type{d_extents_type{},1.f};
    auto m1 = matrix_type{3,3,value_type{1}};


    BOOST_REQUIRE_THROW(prod(t1,m1,0),std::length_error);
    BOOST_REQUIRE_THROW(prod(t1,m1,1),std::length_error);
    BOOST_REQUIRE_THROW(prod(t1,m1,3),std::length_error);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_prod_tensor_1, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;
    using tensor_type  = ublas::dynamic_tensor<value_type,layout_type>;

    // left-hand and right-hand side have the
    // the same number of elements

    for(auto const& na : extents) {

        auto a  = tensor_type( na, value_type{2} );
        auto b  = tensor_type( na, value_type{3} );

        auto const pa = a.rank();

        // the number of contractions is changed.
        for( auto q = 0ul; q <= pa; ++q) { // pa

            auto phi = std::vector<std::size_t> ( q );

            std::iota(phi.begin(), phi.end(), 1ul);

            auto c = ublas::prod(a, b, phi);

            auto acc = value_type(1);
            for(auto i = 0ul; i < q; ++i)
                acc *= value_type( static_cast< inner_type_t<value_type> >( a.extents().at(phi.at(i)-1) ) );

            for(auto i = 0ul; i < c.size(); ++i)
                BOOST_CHECK_EQUAL( c[i] , acc * a[0] * b[0] );

        }
    }
}

BOOST_AUTO_TEST_CASE( test_tensor_prod_tensor_1_exception )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = float;
    using layout_type  = ublas::layout::first_order;
    using d_extents_type = ublas::extents<>;
    using d_tensor_type  = ublas::dynamic_tensor<value_type,layout_type>;

    auto t1 = d_tensor_type{};
    std::vector<std::size_t> phia = {1,2,3};
    std::vector<std::size_t> phib = {1,2,3,4,5};


    auto t3 = d_tensor_type{d_extents_type{1,2},1.f};
    auto t4 = d_tensor_type{d_extents_type{1,2},1.f};
    BOOST_REQUIRE_THROW(prod(t3,t4,phia,phib),std::runtime_error);


    auto t5 = d_tensor_type{d_extents_type{1,2,3,4},1.f};
    auto t6 = d_tensor_type{d_extents_type{1,2},1.f};
    BOOST_REQUIRE_THROW(prod(t5,t6,phia,phib),std::runtime_error);


    auto t7 = d_tensor_type{d_extents_type{1,2,3,4,5},1.f};
    auto t8 = d_tensor_type{d_extents_type{1,2,3,4,5},1.f};
    BOOST_REQUIRE_THROW(prod(t7,t8,phia,phib),std::runtime_error);

    std::vector<std::size_t> phia_2 = {1,2,3,5,4};
    std::vector<std::size_t> phib_2 = {1,2,3,4,5};
    auto t9 = d_tensor_type{d_extents_type{1,2,3,4,5,6},1.f};
    auto t10 = d_tensor_type{d_extents_type{1,2,3,4,5,6},1.f};
    BOOST_REQUIRE_THROW(prod(t9,t10,phia_2,phib_2),std::runtime_error);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_prod_tensor_2, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;
    using tensor_type  = ublas::dynamic_tensor<value_type,layout_type>;


    auto compute_factorial = [](auto const& p){
        auto f = 1ul;
        for(auto i = 1u; i <= p; ++i)
            f *= i;
        return f;
    };

    auto permute_extents = [](auto const& pi, auto const& na){
        auto nb = na;
        assert(pi.size() == na.size());
        for(auto j = 0u; j < pi.size(); ++j)
            nb[pi[j]-1] = na[j];
        return nb;
    };


    // left-hand and right-hand side have the
    // the same number of elements

    for(auto const& na : extents) {

        auto a  = tensor_type( na, value_type{2} );
        auto const pa = a.rank();


        auto pi   = std::vector<std::size_t>(pa);
        auto fac = compute_factorial(pa);
        std::iota( pi.begin(), pi.end(), 1 );

        for(auto f = 0ul; f < fac; ++f)
        {
            auto nb = permute_extents( pi, na  );
            auto b  = tensor_type( nb, value_type{3} );

            // the number of contractions is changed.
            for( auto q = 0ul; q <= pa; ++q) { // pa

                auto phia = std::vector<std::size_t> ( q );  // concatenation for a
                auto phib = std::vector<std::size_t> ( q );  // concatenation for b

                std::iota(phia.begin(), phia.end(), 1ul);
                std::transform(  phia.begin(), phia.end(), phib.begin(),
                                 [&pi] ( std::size_t i ) { return pi.at(i-1); } );

                auto c = ublas::prod(a, b, phia, phib);

                auto acc = value_type(1);
                for(auto i = 0ul; i < q; ++i)
                    acc *= value_type( static_cast< inner_type_t<value_type> >( a.extents().at(phia.at(i)-1) ) );

                for(auto i = 0ul; i < c.size(); ++i)
                    BOOST_CHECK_EQUAL( c[i] , acc * a[0] * b[0] );

            }

            std::next_permutation(pi.begin(), pi.end());
        }
    }

    auto phia = std::vector<std::size_t >(3);
    auto sphia = std::vector<std::size_t>(2);

    BOOST_CHECK_THROW(ublas::prod(tensor_type{}, tensor_type({2,1,2}), phia, phia), std::runtime_error);
        BOOST_CHECK_THROW(ublas::prod(tensor_type({1,2,3}), tensor_type(), phia, phia), std::runtime_error);
        BOOST_CHECK_THROW(ublas::prod(tensor_type({1,2,4}), tensor_type({2,1}), phia, phia), std::runtime_error);
        BOOST_CHECK_THROW(ublas::prod(tensor_type({1,2}), tensor_type({2,1,2}), phia, phia), std::runtime_error);
        BOOST_CHECK_THROW(ublas::prod(tensor_type({1,2}), tensor_type({2,1,3}), sphia, phia), std::runtime_error);
        BOOST_CHECK_THROW(ublas::prod(tensor_type({1,2}), tensor_type({2,2}), phia, sphia), std::runtime_error);
        BOOST_CHECK_THROW(ublas::prod(tensor_type({1,2}), tensor_type({4,4}), sphia, phia), std::runtime_error);
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_inner_prod, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;
    using tensor_type  = ublas::dynamic_tensor<value_type,layout_type>;


    for(auto const& n : extents) {

        auto a  = tensor_type(n, value_type(2));
        auto b  = tensor_type(n, value_type(1));

        auto c = ublas::inner_prod(a, b);
        auto r = std::inner_product(a.begin(),a.end(), b.begin(),value_type(0));

        BOOST_CHECK_EQUAL( c , r );

    }
  BOOST_CHECK_THROW(ublas::inner_prod(tensor_type({1,2,3}), tensor_type({1,2,3,4})), std::length_error); // rank different
  BOOST_CHECK_THROW(ublas::inner_prod(tensor_type(), tensor_type()), std::length_error); //empty tensor
  BOOST_CHECK_THROW(ublas::inner_prod(tensor_type({1,2,3}), tensor_type({3,2,1})), std::length_error); // different extent
}


BOOST_AUTO_TEST_CASE( test_tensor_inner_prod_exception )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = float;
    using layout_type  = ublas::layout::first_order;
    using d_extents_type = ublas::extents<>;
    using d_tensor_type  = ublas::dynamic_tensor<value_type,layout_type>;

    auto t1 = d_tensor_type{d_extents_type{1,2},1.f};
    auto t2 = d_tensor_type{d_extents_type{1,2,3},1.f};
    BOOST_REQUIRE_THROW( ublas::inner_prod(t1, t2), std::length_error);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_norm, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;
    using tensor_type  = ublas::dynamic_tensor<value_type,layout_type>;


    for(auto const& n : extents) {

        auto a  = tensor_type(n);

        auto one = value_type(1);
        auto v = one;
        for(auto& aa: a)
            aa = v, v += one;


        auto c = ublas::inner_prod(a, a);
        auto r = std::inner_product(a.begin(),a.end(), a.begin(),value_type(0));

        tensor_type var = (a+a)/value_type(2); // std::complex<float>/int not allowed as expression is captured
        auto r2 = ublas::norm( var );

        BOOST_CHECK_THROW(ublas::norm(tensor_type{}), std::runtime_error);

        BOOST_CHECK_EQUAL( c , r );
        BOOST_CHECK_EQUAL( std::sqrt( c ) , r2 );

    }
}

BOOST_FIXTURE_TEST_CASE( test_tensor_real_imag_conj, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = float;
    using complex_type = std::complex<value_type>;
    using layout_type  = ublas::layout::first_order;

    using tensor_complex_type  = ublas::dynamic_tensor<complex_type, layout_type>;
    using tensor_type  = ublas::dynamic_tensor<value_type,layout_type>;

    for(auto const& n : extents) {

        auto a   = tensor_type(n);
        auto r0  = tensor_type(n);
        auto r00 = tensor_complex_type(n);


        auto one = value_type(1);
        auto v = one;
        for(auto& aa: a)
            aa = v, v += one;

        tensor_type b = (a+a) / value_type( 2 );
        tensor_type r1 = ublas::real( (a+a) / value_type( 2 )  );
        std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::real( l );  }   );
        BOOST_CHECK( (bool) (r0 == r1) );

        tensor_type r2 = ublas::imag( (a+a) / value_type( 2 )  );
        std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::imag( l );  }   );
        BOOST_CHECK( (bool) (r0 == r2) );
        
        tensor_complex_type r3 = ublas::conj( (a+a) / value_type( 2 )  );
        std::transform(  b.begin(), b.end(), r00.begin(), [](auto const& l){ return std::conj( l );  }   );
        BOOST_CHECK( (bool) (r00 == r3) );

    }

    for(auto const& n : extents) {

        auto a   = tensor_complex_type(n);

        auto r00 = tensor_complex_type(n);
        auto r0  = tensor_type(n);


        auto one = complex_type(1,1);
        auto v = one;
        for(auto& aa: a)
            aa = v, v = v + one;

        tensor_complex_type b = (a+a) / complex_type( 2,2 );


        tensor_type r1 = ublas::real( (a+a) / complex_type( 2,2 )  );
        std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::real( l );  }   );
        BOOST_CHECK( (bool) (r0 == r1) );

        tensor_type r2 = ublas::imag( (a+a) / complex_type( 2,2 )  );
        std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::imag( l );  }   );
        BOOST_CHECK( (bool) (r0 == r2) );

        tensor_complex_type r3 = ublas::conj( (a+a) / complex_type( 2,2 )  );
        std::transform(  b.begin(), b.end(), r00.begin(), [](auto const& l){ return std::conj( l );  }   );
        BOOST_CHECK( (bool) (r00 == r3) );



    }



}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_outer_prod, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;
    using tensor_type  = ublas::dynamic_tensor<value_type,layout_type>;

    for(auto const& n1 : extents) {
        auto a  = tensor_type(n1, value_type(2));
        for(auto const& n2 : extents) {

            auto b  = tensor_type(n2, value_type(1));
            auto c  = ublas::outer_prod(a, b);

            for(auto const& cc : c)
                BOOST_CHECK_EQUAL( cc , a[0]*b[0] );
        }
    }
}

template<class V>
void init(std::vector<V>& a)
{
    auto v = V(1);
    for(auto i = 0u; i < a.size(); ++i, ++v){
        a[i] = v;
    }
}

template<class V>
void init(std::vector<std::complex<V>>& a)
{
    auto v = std::complex<V>(1,1);
    for(auto i = 0u; i < a.size(); ++i){
        a[i] = v;
        v.real(v.real()+1);
        v.imag(v.imag()+1);
    }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_trans, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;
    using tensor_type  = ublas::dynamic_tensor<value_type,layout_type>;

    auto fak = [](auto const& p){
        auto f = 1ul;
        for(auto i = 1u; i <= p; ++i)
            f *= i;
        return f;
    };

    auto inverse = [](auto const& pi){
        auto pi_inv = pi;
        for(auto j = 0u; j < pi.size(); ++j)
            pi_inv[pi[j]-1] = j+1;
        return pi_inv;
    };

    for(auto const& n : extents)
    {
        auto const p = n.size();
        auto const s = product(n);
        auto aref = tensor_type(n);
        auto v    = value_type{};
        for(auto i = 0u; i < s; ++i, v+=1)
            aref[i] = v;
        auto a    = aref;


        auto pi = std::vector<std::size_t>(p);
        std::iota(pi.begin(), pi.end(), 1);
        a = ublas::trans( a, pi );
        bool res1 = a == aref;
        BOOST_CHECK( res1 );


        auto const pfak = fak(p);
        auto i = 0u;
        for(; i < pfak-1; ++i) {
            std::next_permutation(pi.begin(), pi.end());
            a = ublas::trans( a, pi );
        }
        std::next_permutation(pi.begin(), pi.end());
        for(; i > 0; --i) {
            std::prev_permutation(pi.begin(), pi.end());
            auto pi_inv = inverse(pi);
            a = ublas::trans( a, pi_inv );
        }
        bool res2 = a == aref; // it was an expression. so evaluate into bool
        BOOST_CHECK( res2 );

    }
}

BOOST_AUTO_TEST_SUITE_END()
