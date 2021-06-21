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


#include <iostream>
#include <algorithm>
#include <vector>

#include <boost/numeric/ublas/tensor/multiplication.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>
#include "utility.hpp"

#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_SUITE (test_tensor_contraction)


//using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;

using test_types = zip<float>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;

//using test_types = zip<int>::with_t<boost::numeric::ublas::layout::first_order>;


struct fixture
{
  using extents_t = boost::numeric::ublas::extents<>;
  const std::vector<extents_t> extents =
  {
    extents_t{1,1}, // 1
    extents_t{1,2}, // 2
    extents_t{2,1}, // 3
    extents_t{2,3}, // 4
    extents_t{5,4}, // 5
    extents_t{2,3,1}, // 6
    extents_t{4,1,3}, // 7
    extents_t{1,2,3}, // 8
    extents_t{4,2,3}, // 9
    extents_t{4,2,3,5} // 10
  };
};



BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_mtv, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_t   = typename value::first_type;
  using layout_t  = typename value::second_type;
  using vector_t  = std::vector<value_t>;
  using extents_t = ublas::extents<>;
  using extents_base_t = typename extents_t::base_type;


  for(auto const& na : extents) {

    if(ublas::is_scalar(na) || ublas::is_vector(na) || ublas::is_tensor(na))
      continue;

    auto const n1 = na[0];
    auto const n2 = na[1];

    auto a = vector_t(ublas::product(na), value_t{2});
    auto wa = ublas::to_strides(na,layout_t{});

    for(auto m = std::size_t{0}; m < 2; ++m){

      auto nb = extents_t {na[m],std::size_t{1}};
      auto wb = ublas::to_strides(nb,layout_t{});
      auto b  = vector_t  (ublas::product(nb), value_t{1} );

      // [n1 n2 1 ... 1] x1 [n1 1] -> [n2 1 ... 1]
      // [n1 n2 1 ... 1] x2 [n2 1] -> [n1 1 ... 1]

      auto nc_base = extents_base_t(std::max(std::size_t(ublas::size(na)-1u), std::size_t{2}), 1);
      nc_base[0] = m==0 ? n2 : n1;

      auto nc = extents_t (nc_base);
      auto wc = ublas::to_strides(nc,layout_t{});
      auto c  = vector_t  (ublas::product(nc), value_t{0});

      assert( (m==0u) || (m==1u));

      ublas::detail::recursive::mtv(m, n1,n2, c.data(), size_t(1), a.data(), wa[0], wa[1], b.data(), size_t(1));

      auto v = value_t(na[m]);
      BOOST_CHECK(std::equal(c.begin(),c.end(),a.begin(), [v](auto cc, auto aa){return cc == v*aa;}));

    }
  }
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_mtm, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_t   = typename value::first_type;
  using layout_t  = typename value::second_type;
  using vector_t  = std::vector<value_t>;
  using extents_t = ublas::extents<>;

  for(auto const& na : extents) {

    if(ublas::size(na) != 2)
      continue;

    auto a  = vector_t  (ublas::product(na), value_t{2});
    auto wa = ublas::to_strides(na,layout_t{});

    auto nb = extents_t {na[1],na[0]};
    auto wb = ublas::to_strides(nb,layout_t{});
    auto b  = vector_t  (ublas::product(nb), value_t{1} );

    auto nc = extents_t {na[0],nb[1]};
    auto wc = ublas::to_strides(nc,layout_t{});
    auto c  = vector_t  (ublas::product(nc));


    ublas::detail::recursive::mtm(
      c.data(), nc.data(), wc.data(),
      a.data(), na.data(), wa.data(),
      b.data(), nb.data(), wb.data());

    auto v = value_t(na[1])*a[0];
    BOOST_CHECK(std::all_of(c.begin(),c.end(), [v](auto cc){return cc == v;}));

  }
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ttv, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_t   = typename value::first_type;
  using layout_t  = typename value::second_type;
  using vector_t  = std::vector<value_t>;
  using extents_t = ublas::extents<>;
  using extents_base_t = typename extents_t::base_type;

  for(auto const& na : extents) {

    auto a = vector_t(ublas::product(na), value_t{2});
    auto wa = ublas::to_strides(na,layout_t{});
    for(auto m = std::size_t{0}; m < ublas::size(na); ++m){
      auto b  = vector_t  (na[m], value_t{1} );
      auto nb = extents_t {na[m],1};
      auto wb = ublas::to_strides(nb,layout_t{});

      auto nc_base = extents_base_t(std::max(std::size_t{ublas::size(na)-1u}, std::size_t{2}),1);

      for(auto i = 0ul, j = 0ul; i < ublas::size(na); ++i)
        if(i != m)
          nc_base[j++] = na[i];

      auto nc = extents_t (nc_base);
      auto wc = ublas::to_strides(nc,layout_t{});
      auto c  = vector_t  (ublas::product(nc), value_t{0});

      ublas::ttv(m+1, ublas::size(na),
                 c.data(), nc.data(), wc.data(),
                 a.data(), na.data(), wa.data(),
                 b.data(), nb.data(), wb.data());

      auto v = value_t(na[m]);
      BOOST_CHECK(std::equal(c.begin(),c.end(),a.begin(), [v](auto cc, auto aa){return cc == v*aa;}));


    }
  }
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ttm, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_t   = typename value::first_type;
  using layout_t  = typename value::second_type;
  using vector_t  = std::vector<value_t>;
  using extents_t = ublas::extents<>;


  for(auto const& na : extents) {

    auto a = vector_t(ublas::product(na), value_t{2});
    auto wa = ublas::to_strides(na,layout_t{});
    for(auto m = std::size_t{0}; m < ublas::size(na); ++m){
      const auto nb = extents_t {na[m], na[m] };
      const auto b  = vector_t  (ublas::product(nb), value_t{1} );
      const auto wb = ublas::to_strides(nb,layout_t{});


      const auto& nc = na;
      const auto wc = ublas::to_strides(nc,layout_t{});
      auto c  = vector_t  (ublas::product(nc), value_t{0});

      ublas::ttm(m+1, ublas::size(na),
                 c.data(), nc.data(), wc.data(),
                 a.data(), na.data(), wa.data(),
                 b.data(), nb.data(), wb.data());


      auto v = value_t(na[m]);
      BOOST_CHECK(std::equal(c.begin(),c.end(),a.begin(), [v](auto cc, auto aa){return cc == v*aa;}));


//      for(auto i = 0u; i < c.size(); ++i)
//        BOOST_CHECK_EQUAL( c[i] , value_t( static_cast< inner_type_t<value_t> >(na[m]) ) * a[i] );

    }
  }
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ttt_permutation, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_t   = typename value::first_type;
  using layout_t  = typename value::second_type;
  using vector_t  = std::vector<value_t>;
  using extents_t = ublas::extents<>;
  using extents_base_t = typename extents_t::base_type;


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
    auto nb_base = na.base();
    assert(pi.size() == ublas::size(na));
    for(auto j = 0u; j < pi.size(); ++j)
      nb_base[j] = na[pi[j]-1];
    return extents_t(nb_base);
  };


  // left-hand and right-hand side have the
  // the same number of elements

  // computing the inner product with
  // different permutation tuples for
  // right-hand side

  for(auto const& na : extents) {

    auto wa = ublas::to_strides(na,layout_t{});
    auto a  = vector_t(ublas::product(na), value_t{2});
    auto pa  = ublas::size(na);
    auto pia = std::vector<std::size_t>(pa);
    std::iota( pia.begin(), pia.end(), std::size_t{1} );

    auto pib     = pia;
    auto pib_inv = compute_inverse_permutation(pib);

    auto f = compute_factorial(pa);

    // for the number of possible permutations
    // only permutation tuple pib is changed.
    for(auto i = 0u; i < f; ++i) {

      auto nb = permute_extents( pib, na  );
      auto wb = ublas::to_strides(nb,layout_t{});
      auto b  = vector_t(ublas::product(nb), value_t{3});
      auto pb = ublas::size(nb);

      // the number of contractions is changed.
      for(auto q = std::size_t{0}; q <= pa; ++q) {

        auto r  = pa - q;
        auto s  = pb - q;

        auto pc = r+s > 0 ? std::max(std::size_t{r+s},std::size_t{2}) : std::size_t{2};

        auto nc_base = extents_base_t(pc,std::size_t{1});

        for(auto j = 0u; j < r; ++j)
          nc_base[j] = na[pia[j]-1];

        for(auto j = 0u; j < s; ++j)
          nc_base[r+j] = nb[ pib_inv[j]-1 ];

        auto nc = extents_t ( nc_base );
        auto wc = ublas::to_strides(nc,layout_t{});
        auto c  = vector_t  ( ublas::product(nc), value_t(0) );

        ublas::ttt(pa,pb,q,
                   pia.data(), pib_inv.data(),
                   c.data(), nc.data(), wc.data(),
                   a.data(), na.data(), wa.data(),
                   b.data(), nb.data(), wb.data());


        auto acc = std::size_t{1};
        for(auto j = r; j < pa; ++j)
          acc *= na[pia[j]-1];

        auto v = value_t(acc)*a[0]*b[0];

        BOOST_CHECK( std::all_of(c.begin(),c.end(), [v](auto cc){return cc == v; } ) );

      }

      std::next_permutation(pib.begin(), pib.end());
      pib_inv = compute_inverse_permutation(pib);
    }
  }
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ttt, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_t   = typename value::first_type;
  using layout_t  = typename value::second_type;
  using vector_t  = std::vector<value_t>;
  using extents_t = ublas::extents<>;
  using extents_base_t = typename extents_t::base_type;

  // left-hand and right-hand side have the
  // the same number of elements

  // computing the inner product with
  // different permutation tuples for
  // right-hand side

  for(auto const& na : extents) {

    auto wa = ublas::to_strides(na,layout_t{});
    auto a  = vector_t(ublas::product(na), value_t{2});
    auto pa = ublas::size(na);

    auto const& nb = na;
    auto wb = ublas::to_strides(nb,layout_t{});
    auto b  = vector_t(ublas::product(nb), value_t{3});
    auto pb = ublas::size(nb);

    //  std::cout << "na = ";
    //  std::copy(na.begin(), na.end(), std::ostream_iterator<size_type>(std::cout, " "));
    //  std::cout << std::endl;

    //  std::cout << "nb = ";
    //  std::copy(nb.begin(), nb.end(), std::ostream_iterator<size_type>(std::cout, " "));
    //  std::cout << std::endl;


    // the number of contractions is changed.
    for( auto q = std::size_t{0}; q <= pa; ++q) { // pa

      auto r  = pa - q;
      auto s  = pb - q;

      auto pc = r+s > 0 ? std::max(std::size_t{r+s},std::size_t{2}) : std::size_t{2};

      auto nc_base = extents_base_t(pc,std::size_t{1});

      for(auto i = 0u; i < r; ++i)
        nc_base[i] = na[i];

      for(auto i = 0u; i < s; ++i)
        nc_base[r+i] = nb[i];

      auto nc = extents_t ( nc_base );
      auto wc = ublas::to_strides(nc,layout_t{});
      auto c  = vector_t  ( ublas::product(nc), value_t{0} );

      //   std::cout << "nc = ";
      //   std::copy(nc.begin(), nc.end(), std::ostream_iterator<size_type>(std::cout, " "));
      //   std::cout << std::endl;

      ublas::ttt(pa,pb,q,
                 c.data(), nc.data(), wc.data(),
                 a.data(), na.data(), wa.data(),
                 b.data(), nb.data(), wb.data());


      auto acc = std::size_t{1};
      for(auto i = r; i < pa; ++i)
        acc *= na[i];

      auto v = value_t(acc)*a[0]*b[0];

      BOOST_CHECK( std::all_of(c.begin(),c.end(), [v](auto cc){return cc == v; } ) );
    }

  }
}





BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_inner, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_t   = typename value::first_type;
  using layout_t  = typename value::second_type;
  using vector_t  = std::vector<value_t>;

  for(auto const& n : extents) {

    auto a = vector_t(ublas::product(n), value_t{2});
    auto b = vector_t(ublas::product(n), value_t{3});
    auto w = ublas::to_strides(n,layout_t{});

    auto c = ublas::inner(ublas::size(n), n.data(), a.data(), w.data(), b.data(), w.data(), value_t(0));
    auto cref = std::inner_product(a.begin(), a.end(), b.begin(), value_t(0));


    BOOST_CHECK_EQUAL( c , cref );

  }

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_outer, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_t   = typename value::first_type;
  using layout_t  = typename value::second_type;
  using extents_t = ublas::extents<>;
  using vector_t  = std::vector<value_t>;


  for(auto const& na : extents) {

    auto a = vector_t(ublas::product(na), value_t{2});
    auto wa = ublas::to_strides(na,layout_t{});

    for(auto const& nb : extents) {

      auto b = vector_t(ublas::product(nb), value_t{3});
      auto wb = ublas::to_strides(nb,layout_t{});

      auto c = vector_t(ublas::product(nb)*ublas::product(na));
      auto nc_base = typename extents_t::base_type(ublas::size(na)+ublas::size(nb));

      for(auto i = 0u; i < ublas::size(na); ++i)
        nc_base[i] = na[i];
      for(auto i = 0u; i < ublas::size(nb); ++i)
        nc_base[i+ublas::size(na)] = nb[i];

      auto nc = extents_t(nc_base);
      auto wc = ublas::to_strides(nc,layout_t{});

      ublas::outer(c.data(), ublas::size(nc), nc.data(), wc.data(),
                   a.data(), ublas::size(na), na.data(), wa.data(),
                   b.data(), ublas::size(nb), nb.data(), wb.data());

      for(auto const& cc : c)
        BOOST_CHECK_EQUAL( cc , a[0]*b[0] );
    }

  }

}


BOOST_AUTO_TEST_SUITE_END()
