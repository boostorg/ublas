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
#include <boost/numeric/ublas/tensor/algorithms.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>
#include "utility.hpp"

#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_SUITE ( test_tensor_algorithms/*,
                      * boost::unit_test::depends_on("test_shape_dynamic") * boost::unit_test::depends_on("test_strides")*/
                      )

// BOOST_AUTO_TEST_SUITE ( test_tensor_algorithms)


using test_types  = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;
using test_types2 = std::tuple<std::int32_t,std::int64_t,float,double,std::complex<float>>;

struct fixture
{
  using extents_t = boost::numeric::ublas::extents<>;
  const std::vector<extents_t> extents =
  {
      extents_t{1,1}, // 1
      extents_t{1,2}, // 2
      extents_t{2,1}, // 3
      extents_t{2,3}, // 4
      extents_t{2,3,1}, // 5
      extents_t{4,1,3}, // 6
      extents_t{1,2,3}, // 7
      extents_t{4,2,3}, // 8
      extents_t{4,2,3,5}
  };
};




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_algorithms_copy, value,  test_types2, fixture )
{
  namespace ublas    = boost::numeric::ublas;
  using value_type   = value;
  using vector_t     = std::vector<value_type>;


  constexpr auto first_order = ublas::layout::first_order{};
  constexpr auto last_order  = ublas::layout::last_order {};


  for(auto const& n : extents) {

    auto a  = vector_t(product(n));
    auto b  = vector_t(product(n));
    auto c  = vector_t(product(n));

    auto wa = ublas::to_strides(n,first_order);
    auto wb = ublas::to_strides(n,last_order );
    auto wc = ublas::to_strides(n,first_order);

    auto v = value_type{};
    for(auto i = 0ul; i < a.size(); ++i, v+=1){
      a[i]=v;
    }

    ublas::copy( ublas::size(n), n.data(), b.data(), wb.data(), a.data(), wa.data() );
    ublas::copy( ublas::size(n), n.data(), c.data(), wc.data(), b.data(), wb.data() );

    for(auto i = 1ul; i < c.size(); ++i)
      BOOST_CHECK_EQUAL( c[i], a[i] );

    std::size_t const*const p0 = nullptr;
    value_type* c0 = nullptr;

    BOOST_CHECK_THROW( ublas::copy( ublas::size(n), p0,      c.data(), wc.data(), b.data(), wb.data() ), std::runtime_error );
    BOOST_CHECK_THROW( ublas::copy( ublas::size(n), n.data(), c.data(), p0,        b.data(), wb.data() ), std::runtime_error );
    BOOST_CHECK_THROW( ublas::copy( ublas::size(n), n.data(), c.data(), wc.data(), b.data(), p0        ), std::runtime_error );
    BOOST_CHECK_THROW( ublas::copy( ublas::size(n), n.data(), c0,       wc.data(), b.data(), wb.data() ), std::runtime_error );
  }
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_algorithms_copy_exceptions, value,  test_types2, fixture )
{
  namespace ublas    = boost::numeric::ublas;
  using value_type   = value;
  using vector_t     = std::vector<value_type>;
  constexpr auto first_order = ublas::layout::first_order{};


  for(auto const& n : extents) {

    value_type* a  = nullptr;
    auto c  = vector_t(ublas::product(n));

    auto wa = ublas::to_strides(n,first_order);
    auto wc = ublas::to_strides(n,first_order);

    BOOST_REQUIRE_THROW( ublas::copy( ublas::size(n), n.data(), c.data(), wc.data(), a, wa.data() ), std::runtime_error );

  }

  for(auto const& n : extents) {

    value_type* a  = nullptr;
    value_type* c  = nullptr;

    auto wa = ublas::to_strides(n,first_order);
    auto wc = ublas::to_strides(n,first_order);

    BOOST_REQUIRE_THROW( ublas::copy( ublas::size(n), n.data(), c, wc.data(), a, wa.data() ), std::runtime_error );

  }

  for(auto const& n : extents) {

    auto a  = vector_t(product(n));
    value_type* c  = nullptr;

    auto wa = ublas::to_strides(n,first_order);
    auto wc = ublas::to_strides(n,first_order);

    BOOST_REQUIRE_THROW( ublas::copy( ublas::size(n), n.data(), c, wc.data(), a.data(), wa.data() ), std::runtime_error );

  }

  for(auto const& n : extents) {

    auto a  = vector_t(product(n));
    auto c  = vector_t(product(n));


    size_t* wa = nullptr;
    auto wc = ublas::to_strides(n,first_order);

    BOOST_REQUIRE_THROW( ublas::copy( ublas::size(n), n.data(), c.data(), wc.data(), a.data(), wa ), std::runtime_error );

  }

  for(auto const& n : extents) {

    auto a  = vector_t(product(n));
    auto c  = vector_t(product(n));



    size_t* wc = nullptr;
    auto wa = ublas::to_strides(n,first_order);

    BOOST_REQUIRE_THROW( ublas::copy( ublas::size(n), n.data(), c.data(), wc, a.data(), wa.data() ), std::runtime_error );

  }

  for(auto const& n : extents) {

    auto a  = vector_t(product(n));
    auto c  = vector_t(product(n));

    size_t* m = nullptr;
    auto wa = ublas::to_strides(n,first_order);
    auto wc = ublas::to_strides(n,first_order);

    BOOST_REQUIRE_THROW( ublas::copy( ublas::size(n), m, c.data(), wc.data(), a.data(), wa.data() ), std::runtime_error );

  }
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_algorithms_transform, value,  test_types2, fixture )
{
  namespace ublas    = boost::numeric::ublas;
  using value_type   = value;
  using vector_t  = std::vector<value_type>;

  constexpr auto first_order = ublas::layout::first_order{};
  constexpr auto last_order  = ublas::layout::last_order {};


  for(auto const& n : extents) {

    auto a  = vector_t(ublas::product(n));
    auto b  = vector_t(ublas::product(n));
    auto c  = vector_t(ublas::product(n));

    auto wa = ublas::to_strides(n,first_order);
    auto wb = ublas::to_strides(n,last_order );
    auto wc = ublas::to_strides(n,first_order);

    auto v = value_type{};
    for(auto i = 0ul; i < a.size(); ++i, v+=1){
      a[i]=v;
    }

    ublas::transform( ublas::size(n), n.data(), b.data(), wb.data(), a.data(), wa.data(), [](value_type const& a){ return a + value_type(1);} );
    ublas::transform( ublas::size(n), n.data(), c.data(), wc.data(), b.data(), wb.data(), [](value_type const& a){ return a - value_type(1);} );

    auto zero = std::size_t{0};
    ublas::transform(zero, n.data(), c.data(), wc.data(), b.data(), wb.data(), [](value_type const& a){ return a + value_type(1);} );

    value_type* c0 = nullptr;
    const std::size_t* s0 = nullptr;
    std::size_t const*const p0 = nullptr;

    BOOST_CHECK_THROW(ublas::transform( ublas::size(n), n.data(), c0, wb.data(), a.data(), wa.data(), [](value_type const& a){ return a + value_type(1);} ), std::runtime_error);
    BOOST_CHECK_THROW(ublas::transform( ublas::size(n), n.data(), b.data(), s0, a.data(), wa.data(), [](value_type const& a){ return a + value_type(1);} ), std::runtime_error);
    BOOST_CHECK_THROW(ublas::transform( ublas::size(n), p0, b.data(), wb.data(), a.data(), wa.data(), [](value_type const& a){ return a + value_type(1);} ), std::runtime_error);


    for(auto i = 1ul; i < c.size(); ++i)
      BOOST_CHECK_EQUAL( c[i], a[i] );

  }
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_algorithms_transform_exceptions, value,  test_types2, fixture )
{
  namespace ublas    = boost::numeric::ublas;
  using value_type   = value;
  using vector_t  = std::vector<value_type>;

  constexpr auto first_order = ublas::layout::first_order{};

  for(auto const& n : extents) {

    value_type* a  = nullptr;
    auto c  = vector_t(ublas::product(n));

    auto wa = ublas::to_strides(n,first_order);
    auto wc = ublas::to_strides(n,first_order);

    BOOST_REQUIRE_THROW( ublas::transform( ublas::size(n), n.data(), c.data(), wc.data(), a, wa.data(), [](value_type const& a){ return a + value_type(1);} ), std::runtime_error );

  }

  for(auto const& n : extents) {

    value_type* a  = nullptr;
    value_type* c  = nullptr;

    auto wa = ublas::to_strides(n,first_order);
    auto wc = ublas::to_strides(n,first_order);

    BOOST_REQUIRE_THROW( ublas::transform( ublas::size(n), n.data(), c, wc.data(), a, wa.data(), [](value_type const& a){ return a + value_type(1);} ), std::runtime_error );

  }

  for(auto const& n : extents) {

    auto a  = vector_t(ublas::product(n));
    value_type* c  = nullptr;

    auto wa = ublas::to_strides(n,first_order);
    auto wc = ublas::to_strides(n,first_order);

    BOOST_REQUIRE_THROW( ublas::transform( ublas::size(n), n.data(), c, wc.data(), a.data(), wa.data(), [](value_type const& a){ return a + value_type(1);} ), std::runtime_error );

  }

  for(auto const& n : extents) {

    auto a  = vector_t(product(n));
    auto c  = vector_t(product(n));

    size_t* wa = nullptr;
    auto wc = ublas::to_strides(n,first_order);

    BOOST_REQUIRE_THROW( ublas::transform( ublas::size(n), n.data(), c.data(), wc.data(), a.data(), wa, [](value_type const& a){ return a + value_type(1);} ), std::runtime_error );

  }

  for(auto const& n : extents) {

    auto a  = vector_t(ublas::product(n));
    auto c  = vector_t(ublas::product(n));

    size_t* wc = nullptr;
    auto wa = ublas::to_strides(n,first_order);

    BOOST_REQUIRE_THROW( ublas::transform( ublas::size(n), n.data(), c.data(), wc, a.data(), wa.data(), [](value_type const& a){ return a + value_type(1);} ), std::runtime_error );

  }

  for(auto const& n : extents) {

    auto a  = vector_t(product(n));
    auto c  = vector_t(product(n));

    size_t* m = nullptr;
    auto wa = ublas::to_strides(n,first_order);
    auto wc = ublas::to_strides(n,first_order);

    BOOST_REQUIRE_THROW( ublas::transform( ublas::size(n), m, c.data(), wc.data(), a.data(), wa.data(), [](value_type const& a){ return a + value_type(1);} ), std::runtime_error );

  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_algorithms_accumulate, value,  test_types2, fixture )
{
  namespace ublas    = boost::numeric::ublas;
  using value_type   = value;
  using vector_t     = std::vector<value_type>;

  constexpr auto first_order = ublas::layout::first_order{};


  for(auto const& n : extents) {

    auto const s = ublas::product(n);

    auto a  = vector_t(ublas::product(n));
    auto wa = ublas::to_strides(n,first_order);


    auto v = value_type{};
    for(auto i = 0ul; i < a.size(); ++i, v+=value_type(1)){
      a[i]=v;
    }

    auto acc = ublas::accumulate( ublas::size(n), n.data(), a.data(), wa.data(), v);

    auto sum = std::div(s*(s+1),2).quot;

    BOOST_CHECK_EQUAL( acc, value_type( static_cast< inner_type_t<value_type> >( sum ) )  );

    auto zero = std::size_t{0};
    (void)ublas::accumulate(zero, n.data(), a.data(), wa.data(),v);

    value_type* c0 = nullptr;
    std::size_t const*const p0 = nullptr;

    BOOST_CHECK_THROW((void)ublas::accumulate( ublas::size(n), n.data(), c0, wa.data(), v), std::runtime_error);
    BOOST_CHECK_THROW((void)ublas::accumulate( ublas::size(n), n.data(), a.data(), p0, v), std::runtime_error);
    BOOST_CHECK_THROW((void)ublas::accumulate( ublas::size(n), p0, a.data(), wa.data(), v), std::runtime_error);


    auto acc2 = ublas::accumulate( ublas::size(n), n.data(), a.data(), wa.data(), v,
                                  [](auto const& l, auto const& r){return l + r; });

    BOOST_CHECK_EQUAL( acc2, value_type( static_cast< inner_type_t<value_type> >( sum ) )  );

    (void)ublas::accumulate(zero, n.data(), a.data(), wa.data(), v, [](auto const& l, auto const& r){return l + r; });

    BOOST_CHECK_THROW((void)ublas::accumulate( ublas::size(n), n.data(), c0, wa.data(), v,[](auto const& l, auto const& r){return l + r; }), std::runtime_error);
    BOOST_CHECK_THROW((void)ublas::accumulate( ublas::size(n), n.data(), a.data(), p0, v, [](auto const& l, auto const& r){return l + r; }), std::runtime_error);
    BOOST_CHECK_THROW((void)ublas::accumulate( ublas::size(n), p0, a.data(), wa.data(),v, [](auto const& l, auto const& r){return l + r; }), std::runtime_error);

  }
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_algorithms_accumulate_exceptions, value,  test_types2, fixture )
{
  namespace ublas    = boost::numeric::ublas;
  using value_type   = value;
  using vector_t  = std::vector<value_type>;
  constexpr auto first_order = ublas::layout::first_order{};


  for(auto const& n : extents) {
    value_type* a  = nullptr;
    auto wa = ublas::to_strides(n,first_order);
    BOOST_REQUIRE_THROW( (void)ublas::accumulate( ublas::size(n), n.data(), a, wa.data(), value_type{0} ), std::runtime_error );

  }

  for(auto const& n : extents) {
    value_type* a  = nullptr;
    auto wa = ublas::to_strides(n,first_order);
    BOOST_REQUIRE_THROW( (void)ublas::accumulate( ublas::size(n), n.data(), a, wa.data(), value_type{0},[](value_type const& a,value_type const& b){ return a + b;} ), std::runtime_error );
  }

  for(auto const& n : extents) {
    auto a  = vector_t(product(n));
    auto wa = ublas::to_strides(n,first_order);
    size_t p = 0u;
    BOOST_CHECK_EQUAL ( ublas::accumulate( p, n.data(), a.data(), wa.data(), value_type{0} ), value_type{0} );
  }

  for(auto const& n : extents) {
    auto a  = vector_t(product(n));
    auto wa = ublas::to_strides(n,first_order);
    size_t p = 0u;
    BOOST_CHECK_EQUAL( ublas::accumulate( p, n.data(), a.data(), wa.data(), value_type{0}, [](value_type const& a,value_type const& b){ return a + b;} ), value_type{0} );
  }

  for(auto const& n : extents) {
    auto a  = vector_t(product(n));
    size_t* wa = nullptr;
    BOOST_REQUIRE_THROW( (void)ublas::accumulate( ublas::size(n), n.data(), a.data(), wa, value_type{0} ), std::runtime_error );
  }

  for(auto const& n : extents) {
    auto a  = vector_t(product(n));
    auto wa = ublas::to_strides(n,first_order);
    size_t* m = nullptr;
    BOOST_REQUIRE_THROW( (void)ublas::accumulate( ublas::size(n), m, a.data(), wa.data(), value_type{0}, [](value_type const& a,value_type const& b){ return a + b;} ), std::runtime_error );
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


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_algorithms_trans, value,  test_types, fixture )
{
  namespace ublas   = boost::numeric::ublas;
  using value_type  = typename value::first_type;
//  using layout_t    = typename value::second_type;
  using vector_t    = std::vector<value_type>;
  using base_t      = typename extents_t::base_type;
  using permutation_type = std::vector<std::size_t>;

  constexpr auto first_order = ublas::layout::first_order{};


  for(auto const& n : extents) {

    auto p   = ublas::size(n);
    auto s   = ublas::product(n);

    auto pi  = permutation_type(p);
    auto a   = vector_t(s);
    auto b1  = vector_t(s);
    auto b2  = vector_t(s);
    auto c1  = vector_t(s);
    auto c2  = vector_t(s);

    auto wa = ublas::to_strides(n,first_order);

    init(a);

    // so wie last-order.
    for(auto i = std::size_t{0}, j = p; i < ublas::size(n); ++i, --j)
      pi[i] = j;

    auto nc_base = base_t(p);
    for(auto i = 0u; i < p; ++i)
      nc_base[pi[i]-1] = n[i];

    auto nc = extents_t(std::move(nc_base));

    auto wc    = ublas::to_strides(nc,first_order);
    auto wc_pi = base_t(p);
    for(auto i = 0u; i < p; ++i)
      wc_pi[pi[i]-1] = wc[i];

    ublas::copy ( p, n.data(),            c1.data(), wc_pi.data(), a.data(), wa.data());
    ublas::trans( p, n.data(), pi.data(), c2.data(), wc.data(),    a.data(), wa.data() );

    if(!std::is_compound_v<value_type>)
      for(auto i = 0ul; i < s; ++i)
        BOOST_CHECK_EQUAL( c1[i], c2[i] );


    auto nb_base = base_t(p);
    for(auto i = 0u; i < p; ++i)
      nb_base[pi[i]-1] = nc[i];

    auto nb = extents_t(std::move(nb_base));

    auto wb    = ublas::to_strides(nb,first_order);
    auto wb_pi = base_t(p);
    for(auto i = 0u; i < p; ++i)
      wb_pi[pi[i]-1] = wb[i];

    ublas::copy ( p, nc.data(),            b1.data(), wb_pi.data(), c1.data(), wc.data());
    ublas::trans( p, nc.data(), pi.data(), b2.data(), wb.data(),    c2.data(), wc.data() );

    if(!std::is_compound_v<value_type>)
      for(auto i = 0ul; i < s; ++i)
        BOOST_CHECK_EQUAL( b1[i], b2[i] );

    for(auto i = 0ul; i < s; ++i)
      BOOST_CHECK_EQUAL( a[i], b2[i] );

    auto zero = std::size_t{0};
    ublas::trans( zero, n.data(), pi.data(), c2.data(), wc.data(), a.data(), wa.data() );
    ublas::trans( zero, nc.data(), pi.data(), b2.data(), wb.data(), c2.data(), wc.data() );

    value_type *c0 = nullptr;
    std::size_t const*const s0 = nullptr;

    BOOST_CHECK_THROW(ublas::trans( p, n.data(), pi.data(), c0, wc.data(),  a.data(), wa.data()), std::runtime_error);
    BOOST_CHECK_THROW(ublas::trans( p, s0, pi.data(), c2.data(),wc.data(),  a.data(), wa.data()), std::runtime_error);
    BOOST_CHECK_THROW(ublas::trans( p, n.data(), pi.data(), c2.data(), s0,  a.data(), wa.data()), std::runtime_error);
    BOOST_CHECK_THROW(ublas::trans( p, n.data(), s0, c2.data(), wc.data(),  a.data(), wa.data()), std::runtime_error);

  }
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_algorithms_trans_exceptions, value,  test_types, fixture )
{
  namespace ublas        = boost::numeric::ublas;
  using value_type       = typename value::first_type;
  using layout_t         = typename value::second_type;
  using vector_t         = std::vector<value_type>;
  using permutation_type = std::vector<std::size_t>;

  constexpr auto layout = layout_t{};

  std::size_t* nnullptr = nullptr;
  value_type * anullptr = nullptr;

  for(auto const& n : extents) {
    auto p  = ublas::size(n);
    auto s  = ublas::product(n);
    auto pi = permutation_type(p);
    auto a  = vector_t(s);
    auto c  = vector_t(s);
    auto wa = ublas::to_strides(n,layout);
    auto wc = ublas::to_strides(n,layout);
    if(p>1){
      BOOST_REQUIRE_THROW( ublas::trans( p, nnullptr, pi.data(), c.data(), wc.data(),    a.data(), wa.data() ), std::runtime_error );
      BOOST_REQUIRE_THROW( ublas::trans( p, n.data() , nnullptr , c.data(), wc.data(),    a.data(), wa.data() ), std::runtime_error );
      BOOST_REQUIRE_THROW( ublas::trans( p, n.data() , pi.data(), c.data(), nnullptr ,    a.data(), nnullptr  ), std::runtime_error );
      BOOST_REQUIRE_THROW( ublas::trans( p, n.data() , pi.data(), c.data(), wc.data(),    a.data(), nnullptr  ), std::runtime_error );
      BOOST_REQUIRE_THROW( ublas::trans( p, n.data() , pi.data(), c.data(), nnullptr ,    a.data(), wa.data() ), std::runtime_error );
      BOOST_REQUIRE_THROW( ublas::trans( p, n.data() , pi.data(), anullptr, wc.data(),    anullptr, wa.data() ), std::runtime_error );
      BOOST_REQUIRE_THROW( ublas::trans( p, n.data() , pi.data(), c.data(), wc.data(),    anullptr, wa.data() ), std::runtime_error );
      BOOST_REQUIRE_THROW( ublas::trans( p, n.data() , pi.data(), anullptr, wc.data(),    a.data(), wa.data() ), std::runtime_error );
    }

    // ublas::trans( p, n.data(), pi.data(), c.data(), wc.data(),    a.data(), wa.data() );
  }

}


BOOST_AUTO_TEST_SUITE_END()
