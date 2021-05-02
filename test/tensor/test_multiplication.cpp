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


#include <iostream>
#include <algorithm>
#include <vector>

#include <boost/numeric/ublas/tensor/multiplication.hpp>
#include <boost/numeric/ublas/tensor/fixed_rank_extents.hpp>
#include <boost/numeric/ublas/tensor/static_extents.hpp>
#include <boost/numeric/ublas/tensor/dynamic_extents.hpp>
#include <boost/numeric/ublas/tensor/strides.hpp>
#include "utility.hpp"

#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_SUITE (test_tensor_contraction)


using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;

//using test_types = zip<int>::with_t<boost::numeric::ublas::layout::first_order>;


struct fixture
{
    using extents_type = boost::numeric::ublas::extents<>;
    fixture()
      : extents {
          extents_type{1,1}, // 1
          extents_type{1,2}, // 2
          extents_type{2,1}, // 3
          extents_type{2,3}, // 4
          extents_type{5,4}, // 5
          extents_type{2,3,1}, // 6
          extents_type{4,1,3}, // 7
          extents_type{1,2,3}, // 8
          extents_type{4,2,3}, // 9
          extents_type{4,2,3,5}} // 10
    {
    }
    std::vector<extents_type> extents;
};



BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_mtv, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;
    using vector_type  = std::vector<value_type>;
    using extents_type = ublas::extents<>;
    using strides_type = ublas::strides_t<extents_type,layout_type>;
    using extents_type_base = typename extents_type::base_type;
    using size_type = typename extents_type_base::value_type;


    for(auto const& na : extents) {

      if(ublas::size(na) > 2)
            continue;

      auto a = vector_type(ublas::product(na), value_type{2});
        auto wa = strides_type(na);
        for(auto m = size_type(0); m < ublas::size(na); ++m){
            auto nb = extents_type {na[m],1};
            auto wb = strides_type (nb);
            auto b  = vector_type  (product(nb), value_type{1} );

            auto nc_base = extents_type_base(std::max(ublas::size(na)-1, size_type{2}), 1);

            for(auto i = 0u, j = 0u; i < ublas::size(na); ++i)
                if(i != m)
                    nc_base[j++] = na[i];

            auto nc = extents_type (nc_base);
            auto wc = strides_type (nc);
            auto c  = vector_type  (ublas::product(nc), value_type{0});

            ublas::detail::recursive::mtv(
                  m,
                  c.data(), ublas::data(nc), wc.data(),
                  a.data(), ublas::data(na), wa.data(),
                  b.data());


            for(auto i = 0u; i < c.size(); ++i)
                BOOST_CHECK_EQUAL( c[i] , value_type( static_cast< inner_type_t<value_type> >(na[m]) ) * a[i] );

        }
    }
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_mtm, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_type   = typename value::first_type;
  using layout_type  = typename value::second_type;
  using vector_type  = std::vector<value_type>;
  using extents_type = ublas::extents<>;
  using strides_type = ublas::strides_t<extents_type,layout_type>;
  // using extents_type_base = typename extents_type::base_type;


  for(auto const& na : extents) {

    if(ublas::size(na) != 2)
      continue;

    auto a  = vector_type  (ublas::product(na), value_type{2});
    auto wa = strides_type (na);

    auto nb = extents_type {na[1],na[0]};
    auto wb = strides_type (nb);
    auto b  = vector_type  (ublas::product(nb), value_type{1} );

    auto nc = extents_type {na[0],nb[1]};
    auto wc = strides_type (nc);
    auto c  = vector_type  (ublas::product(nc));


    ublas::detail::recursive::mtm(
      c.data(), ublas::data(nc), wc.data(),
      a.data(), ublas::data(na), wa.data(),
      b.data(), ublas::data(nb), wb.data());


    for(auto i = 0u; i < c.size(); ++i)
      BOOST_CHECK_EQUAL( c[i] , value_type( static_cast< inner_type_t<value_type> >(na[1]) ) * a[0] );


}
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ttv, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;
    using vector_type  = std::vector<value_type>;
    using extents_type = ublas::extents<>;
    using strides_type = ublas::strides_t<extents_type,layout_type>;
    using extents_type_base = typename extents_type::base_type;
    using size_type = typename extents_type_base::value_type;


    for(auto const& na : extents) {

      auto a = vector_type(ublas::product(na), value_type{2});
        auto wa = strides_type(na);
        for(auto m = size_type(0); m < ublas::size(na); ++m){
            auto b  = vector_type  (na[m], value_type{1} );
            auto nb = extents_type {na[m],1};
            auto wb = strides_type (nb);

            auto nc_base = extents_type_base(std::max(ublas::size(na)-1, size_type(2)),1);

            for(auto i = 0ul, j = 0ul; i < ublas::size(na); ++i)
                if(i != m)
                    nc_base[j++] = na[i];

            auto nc = extents_type (nc_base);
            auto wc = strides_type (nc);
            auto c  = vector_type  (ublas::product(nc), value_type{0});

            ublas::ttv(m+1, ublas::size(na),
                       c.data(), ublas::data(nc), wc.data(),
                       a.data(), ublas::data(na), wa.data(),
                       b.data(), ublas::data(nb), wb.data());


            for(auto i = 0u; i < c.size(); ++i)
                BOOST_CHECK_EQUAL( c[i] , value_type(na[m]) * a[i] );

        }
    }
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ttm, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;
    using vector_type  = std::vector<value_type>;
    using extents_type = ublas::extents<>;
    using strides_type = ublas::strides_t<extents_type,layout_type>;
    using size_type = typename extents_type::value_type;


    for(auto const& na : extents) {

      auto a = vector_type(ublas::product(na), value_type{2});
        auto wa = strides_type(na);
        for(auto m = size_type(0); m < ublas::size(na); ++m){
            const auto nb = extents_type {na[m], na[m] };
            const auto b  = vector_type  (ublas::product(nb), value_type{1} );
            const auto wb = strides_type (nb);


            const auto& nc = na;
            const auto wc = strides_type (nc);
            auto c  = vector_type  (ublas::product(nc), value_type{0});

            ublas::ttm(m+1, ublas::size(na),
                       c.data(), ublas::data(nc), wc.data(),
                       a.data(), ublas::data(na), wa.data(),
                       b.data(), ublas::data(nb), wb.data());

            for(auto i = 0u; i < c.size(); ++i)
                BOOST_CHECK_EQUAL( c[i] , value_type( static_cast< inner_type_t<value_type> >(na[m]) ) * a[i] );

        }
    }
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ttt_permutation, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;
    using vector_type  = std::vector<value_type>;
    using extents_type = ublas::extents<>;
    using strides_type = ublas::strides_t<extents_type,layout_type>;
    using size_type    = typename strides_type::value_type;


    auto compute_factorial = [](auto const& p){
        auto f = 1ul;
        for(auto i = 1u; i <= p; ++i)
            f *= i;
        return f;
    };


    auto compute_inverse_permutation = [](auto const& pi){
        auto pi_inv = pi;
        for(auto j = 0u; j < pi.size(); ++j)
            pi_inv[pi[j]-1] = j+1;
        return pi_inv;
    };

    auto permute_extents = [](auto const& pi, auto const& na){
        auto nb = na;
        assert(pi.size() == ublas::size(na));
        for(auto j = 0u; j < pi.size(); ++j)
            nb[j] = na[pi[j]-1];
        return nb;
    };


    // left-hand and right-hand side have the
    // the same number of elements

    // computing the inner product with
    // different permutation tuples for
    // right-hand side

    for(auto const& na : extents) {

        auto wa = strides_type(na);
        auto a  = vector_type(ublas::product(na), value_type{2});
        auto pa  = ublas::size(na);
        auto pia = std::vector<size_type>(pa);
        std::iota( pia.begin(), pia.end(), 1 );

        auto pib     = pia;
        auto pib_inv = compute_inverse_permutation(pib);

        auto f = compute_factorial(pa);

        // for the number of possible permutations
        // only permutation tuple pib is changed.
        for(auto i = 0u; i < f; ++i) {

            auto nb = permute_extents( pib, na  );
            auto wb = strides_type(nb);
            auto b  = vector_type(ublas::product(nb), value_type{3});
            auto pb = ublas::size(nb);

            // the number of contractions is changed.
            for( auto q = size_type(0); q <= pa; ++q) {

                auto r  = pa - q;
                auto s  = pb - q;

                auto pc = r+s > 0 ? std::max(r+s,size_type(2)) : size_type(2);

                auto nc_base = std::vector<size_type>( pc , 1 );

                for(auto j = 0u; j < r; ++j)
                    nc_base[ j ] = na[ pia[j]-1 ];

                for(auto j = 0u; j < s; ++j)
                    nc_base[ r + j ] = nb[ pib_inv[j]-1 ];

                auto nc = extents_type ( nc_base );
                auto wc = strides_type ( nc );
                auto c  = vector_type  ( ublas::product(nc), value_type(0) );

                ublas::ttt(pa,pb,q,
                           pia.data(), pib_inv.data(),
                           c.data(), ublas::data(nc), wc.data(),
                           a.data(), ublas::data(na), wa.data(),
                           b.data(), ublas::data(nb), wb.data());


                auto acc = value_type(1);
                for(auto j = r; j < pa; ++j)
                    acc *= value_type( static_cast< inner_type_t<value_type> >(na[pia[j]-1]) );

                for(auto j = 0ul; j < c.size(); ++j)
                    BOOST_CHECK_EQUAL( c[j] , acc * a[0] * b[0] );

            }

            std::next_permutation(pib.begin(), pib.end());
            pib_inv = compute_inverse_permutation(pib);
        }
    }
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ttt, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;
    using vector_type  = std::vector<value_type>;
    using extents_type = ublas::extents<>;
    using strides_type = ublas::strides_t<extents_type,layout_type>;
    using size_type    = typename strides_type::value_type;

    // left-hand and right-hand side have the
    // the same number of elements

    // computing the inner product with
    // different permutation tuples for
    // right-hand side

    for(auto const& na : extents) {

        auto wa = strides_type(na);
        auto a  = vector_type(ublas::product(na), value_type{2});
        auto pa = ublas::size(na);

        auto nb = na;
        auto wb = strides_type(nb);
        auto b  = vector_type(ublas::product(nb), value_type{3});
        auto pb = ublas::size(nb);

        //  std::cout << "na = ";
        //  std::copy(na.begin(), na.end(), std::ostream_iterator<size_type>(std::cout, " "));
        //  std::cout << std::endl;

        //  std::cout << "nb = ";
        //  std::copy(nb.begin(), nb.end(), std::ostream_iterator<size_type>(std::cout, " "));
        //  std::cout << std::endl;


        // the number of contractions is changed.
        for( auto q = size_type(0); q <= pa; ++q) { // pa

            auto r  = pa - q;
            auto s  = pb - q;

            auto pc = r+s > 0 ? std::max(r+s, size_type(2)) : size_type(2);

            auto nc_base = std::vector<size_type>( pc , 1 );

            for(auto i = 0u; i < r; ++i)
                nc_base[ i ] = na[ i ];

            for(auto i = 0u; i < s; ++i)
                nc_base[ r + i ] = nb[ i ];

            auto nc = extents_type ( nc_base );
            auto wc = strides_type ( nc );
            auto c  = vector_type  ( ublas::product(nc), value_type{0} );

            //   std::cout << "nc = ";
            //   std::copy(nc.begin(), nc.end(), std::ostream_iterator<size_type>(std::cout, " "));
            //   std::cout << std::endl;

            ublas::ttt(pa,pb,q,
                       c.data(), ublas::data(nc), wc.data(),
                       a.data(), ublas::data(na), wa.data(),
                       b.data(), ublas::data(nb), wb.data());


            auto acc = value_type(1);
            for(auto i = r; i < pa; ++i)
                acc *= value_type( static_cast< inner_type_t<value_type> >(na[i]) );

            for(auto i = 0u; i < c.size(); ++i)
                BOOST_CHECK_EQUAL( c[i] , acc * a[0] * b[0] );

        }

    }
}





BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_inner, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;
    using strides_type = ublas::strides_t<ublas::extents<>,layout_type>;
    using vector_type  = std::vector<value_type>;


    for(auto const& n : extents) {

        auto a = vector_type(ublas::product(n), value_type{2});
        auto b = vector_type(ublas::product(n), value_type{3});
        auto w = strides_type(n);

        auto c = ublas::inner(ublas::size(n), ublas::data(n), a.data(), w.data(), b.data(), w.data(), value_type(0));
        auto cref = std::inner_product(a.begin(), a.end(), b.begin(), value_type(0));


        BOOST_CHECK_EQUAL( c , cref );

    }

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_outer, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;
    using extents_type = ublas::extents<>;
    using strides_type = ublas::strides_t<extents_type,layout_type>;
    using vector_type  = std::vector<value_type>;


    for(auto const& na : extents) {

        auto a = vector_type(ublas::product(na), value_type{2});
        auto wa = strides_type(na);

        for(auto const& nb : extents) {

            auto b = vector_type(ublas::product(nb), value_type{3});
            auto wb = strides_type(nb);

            auto c = vector_type(ublas::product(nb)*ublas::product(na));
            auto nc_base = typename extents_type::base_type(ublas::size(na)+ublas::size(nb));

            for(auto i = 0u; i < ublas::size(na); ++i)
                nc_base[i] = na[i];
            for(auto i = 0u; i < ublas::size(nb); ++i)
              nc_base[i+ublas::size(na)] = nb[i];

            auto nc = extents_type(nc_base);
            auto wc = strides_type(nc);

            ublas::outer(c.data(), ublas::size(nc), ublas::data(nc), wc.data(),
                         a.data(), ublas::size(na), ublas::data(na), wa.data(),
                         b.data(), ublas::size(nb), ublas::data(nb), wb.data());

            for(auto const& cc : c)
                BOOST_CHECK_EQUAL( cc , a[0]*b[0] );
        }

    }

}


BOOST_AUTO_TEST_SUITE_END()
