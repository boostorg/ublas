//
// 	Copyright (c) 2020, Amit Singh, amitsingh19975@gmail.com
// 	Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//



#include <boost/numeric/ublas/tensor/tensor.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>
#include <boost/test/unit_test.hpp>

#include "utility.hpp"

#include <iostream>
#include <cstdlib>
#include <functional>
#include <tuple>

BOOST_AUTO_TEST_SUITE(test_subtensor_static_rank_expression)

using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;




struct fixture
{
  template<size_t N>
  using extents_t = boost::numeric::ublas::extents<N>;

  static constexpr auto extents = std::make_tuple(
//      extents_t<0>       {},
      extents_t<2>    {1,1},
      extents_t<2>    {1,2},
      extents_t<2>    {2,1},
      extents_t<2>    {2,3},
      extents_t<3>  {2,3,1},
      extents_t<3>  {4,1,3},
      extents_t<3>  {1,2,3},
      extents_t<3>  {4,2,3},
      extents_t<4>{4,2,3,5} );
};


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_subtensor_static_rank_expression_retrieve_extents, value,  test_types, fixture)
{
  namespace ublas  = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;

  auto uplus1 = [](auto const& a){return a + value_t(1); };
  auto uplus2 = [](auto const& a){return value_t(2) + a; };
  auto bplus  = std::plus <value_t>{};
  auto bminus = std::minus<value_t>{};

  for_each_in_tuple(extents, [&](auto const& /*unused*/, auto const& e){



    static constexpr auto size = std::tuple_size_v<std::decay_t<decltype(e)>>;
    using tensor_t = ublas::tensor_static_rank<value_t, size, layout_t>;
    using subtensor = typename tensor_t::subtensor_type;

    auto t = tensor_t(e);
    auto v = value_t{};
    for(auto& tt: t){ tt = v; v+=value_t{1}; }
    auto s = subtensor(t);

    BOOST_CHECK( ublas::detail::retrieve_extents( s ) == e );

    // uexpr1 = s+1
    // uexpr2 = 2+s
    auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_t>( s, uplus1 );
    auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_t>( s, uplus2 );

    BOOST_CHECK( ublas::detail::retrieve_extents( uexpr1 ) == e );
    BOOST_CHECK( ublas::detail::retrieve_extents( uexpr2 ) == e );

    // bexpr_uexpr = (s+1) + (2+s)
    auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( uexpr1, uexpr2, bplus );

    BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_uexpr ) == e );


    // bexpr_bexpr_uexpr = ((s+1) + (2+s)) - s
    auto bexpr_bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( bexpr_uexpr, s, bminus );

    BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr ) == e );

  });

  for_each_in_tuple(extents, [&](auto I, auto const& e1){


    if ( I >= std::tuple_size_v<decltype(extents)> - 1 ){
      return;
    }

    constexpr auto size1 = std::tuple_size_v<std::decay_t<decltype(e1)>>;
    using tensor_type1 = ublas::tensor_static_rank<value_t, size1, layout_t>;
    using subtensor_type1 = typename tensor_type1::subtensor_type;


    for_each_in_tuple(extents, [&,I](auto J, auto const& e2){

      if( J != I + 1 ){
        return;
      }

      static constexpr auto size1 = std::tuple_size_v<std::decay_t<decltype(e1)>>;
      static constexpr auto size2 = std::tuple_size_v<std::decay_t<decltype(e2)>>;
      using tensor_type2 = ublas::tensor_static_rank<value_t, size2, layout_t>;
      using subtensor_type2 = typename tensor_type2::subtensor_type;

      auto v = value_t{};

      tensor_type1 t1(e1);
      for(auto& tt: t1){ tt = v; v+=value_t{1}; }

      tensor_type2 t2(e2);
      for(auto& tt: t2){ tt = v; v+=value_t{2}; }

      auto s1 = subtensor_type1(t1);
      auto s2 = subtensor_type2(t2);

      BOOST_CHECK( ublas::detail::retrieve_extents( s1 ) == ublas::detail::retrieve_extents( t1 ) );
      BOOST_CHECK( ublas::detail::retrieve_extents( s2 ) == ublas::detail::retrieve_extents( t2 ) );
      BOOST_CHECK( ublas::detail::retrieve_extents( s1 ) != ublas::detail::retrieve_extents( s2 ) );

      // uexpr1 = s1+1
      // uexpr2 = 2+s2
      auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_type1>( s1, uplus1 );
      auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_type2>( s2, uplus2 );

      BOOST_CHECK( ublas::detail::retrieve_extents( s1 )     == ublas::detail::retrieve_extents( uexpr1 ) );
      BOOST_CHECK( ublas::detail::retrieve_extents( s2 )     == ublas::detail::retrieve_extents( uexpr2 ) );
      BOOST_CHECK( ublas::detail::retrieve_extents( uexpr1 ) != ublas::detail::retrieve_extents( uexpr2 ) );

      if constexpr( size1 == size2 ){
        // bexpr_uexpr = (s1+1) + (2+s2)
        auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_type1>( uexpr1, uexpr2, bplus );

        BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_uexpr ) == ublas::detail::retrieve_extents(s1) );


        // bexpr_bexpr_uexpr = ((s1+1) + (2+s2)) - s2
        auto bexpr_bexpr_uexpr1 = ublas::detail::make_binary_tensor_expression<tensor_type1>( bexpr_uexpr, s2, bminus );

        BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr1 ) == ublas::detail::retrieve_extents(s2) );

        // bexpr_bexpr_uexpr = s2 - ((s1+1) + (2+s2))
        auto bexpr_bexpr_uexpr2 = ublas::detail::make_binary_tensor_expression<tensor_type1>( s2, bexpr_uexpr, bminus );

        BOOST_CHECK( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr2 ) == ublas::detail::retrieve_extents(s2) );
      }

    });
  });
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_subtensor_static_rank_expression_all_extents_equal, value,  test_types, fixture)
{
  namespace ublas  = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;

  auto uplus1 = [](auto const& a){return a + value_t(1); };
  auto uplus2 = [](auto const& a){return value_t(2) + a; };
  auto bplus  = std::plus <value_t>{};
  auto bminus = std::minus<value_t>{};

  for_each_in_tuple(extents, [&](auto const& /*unused*/, auto& e){
    static constexpr auto size = std::tuple_size_v<std::decay_t<decltype(e)>>;
    using tensor_t = ublas::tensor_static_rank<value_t, size, layout_t>;
    using subtensor = typename tensor_t::subtensor_type;

    auto t = tensor_t(e);
    auto v = value_t{};
    for(auto& tt: t){ tt = v; v+=value_t{1}; }

    auto s = subtensor(t);

    BOOST_CHECK( ublas::detail::all_extents_equal( s , e ) );


    // uexpr1 = s+1
    // uexpr2 = 2+s
    auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_t>( s, uplus1 );
    auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_t>( s, uplus2 );

    BOOST_CHECK( ublas::detail::all_extents_equal( uexpr1, e ) );
    BOOST_CHECK( ublas::detail::all_extents_equal( uexpr2, e ) );

    // bexpr_uexpr = (s+1) + (2+s)
    auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( uexpr1, uexpr2, bplus );

    BOOST_CHECK( ublas::detail::all_extents_equal( bexpr_uexpr, e ) );


    // bexpr_bexpr_uexpr = ((s+1) + (2+s)) - s
    auto bexpr_bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( bexpr_uexpr, s, bminus );

    BOOST_CHECK( ublas::detail::all_extents_equal( bexpr_bexpr_uexpr , e ) );

  });


  for_each_in_tuple(extents, [&](auto I, auto& e1){

    if ( I >= std::tuple_size_v<decltype(extents)> - 1){
      return;
    }

    static constexpr auto size1 = std::tuple_size_v<std::decay_t<decltype(e1)>>;
    using tensor_type1 = ublas::tensor_static_rank<value_t, size1, layout_t>;
    using subtensor_type1 = typename tensor_type1::subtensor_type;

    for_each_in_tuple(extents, [&](auto J, auto& e2){

      if( J != I + 1 ){
        return;
      }


      static constexpr auto size2 = std::tuple_size_v<std::decay_t<decltype(e2)>>;
      using tensor_type2 = ublas::tensor_static_rank<value_t, size2, layout_t>;
      using subtensor_type2 = typename tensor_type2::subtensor_type;

      auto v = value_t{};

      tensor_type1 t1(e1);
      for(auto& tt: t1){ tt = v; v+=value_t{1}; }

      tensor_type2 t2(e2);
      for(auto& tt: t2){ tt = v; v+=value_t{2}; }

      auto s1 = subtensor_type1(t1);
      auto s2 = subtensor_type2(t2);

      BOOST_CHECK( ublas::detail::all_extents_equal( t1, ublas::detail::retrieve_extents(t1) ) );
      BOOST_CHECK( ublas::detail::all_extents_equal( t2, ublas::detail::retrieve_extents(t2) ) );

      // uexpr1 = s1+1
      // uexpr2 = 2+s2
      auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_type1>( s1, uplus1 );
      auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_type2>( s2, uplus2 );

      BOOST_CHECK( ublas::detail::all_extents_equal( uexpr1, ublas::detail::retrieve_extents(uexpr1) ) );
      BOOST_CHECK( ublas::detail::all_extents_equal( uexpr2, ublas::detail::retrieve_extents(uexpr2) ) );

      if constexpr( size1 == size2 ){
        // bexpr_uexpr = (t1+1) + (2+s2)
        auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_type1>( uexpr1, uexpr2, bplus );

        BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_uexpr, ublas::detail::retrieve_extents( bexpr_uexpr  ) ) );

        // bexpr_bexpr_uexpr = ((t1+1) + (2+s2)) - s2
        auto bexpr_bexpr_uexpr1 = ublas::detail::make_binary_tensor_expression<tensor_type1>( bexpr_uexpr, s2, bminus );

        BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr1, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr1  ) ) );

        // bexpr_bexpr_uexpr = s2 - ((t1+1) + (2+s2))
        auto bexpr_bexpr_uexpr2 = ublas::detail::make_binary_tensor_expression<tensor_type1>( s2, bexpr_uexpr, bminus );

        BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr2, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr2  ) ) );


        // bexpr_uexpr2 = (t1+1) + s2
        auto bexpr_uexpr2 = ublas::detail::make_binary_tensor_expression<tensor_type1>( uexpr1, s2, bplus );
        BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_uexpr2, ublas::detail::retrieve_extents( bexpr_uexpr2  ) ) );


        // bexpr_uexpr2 = ((t1+1) + s2) + t1
        auto bexpr_bexpr_uexpr3 = ublas::detail::make_binary_tensor_expression<tensor_type1>( bexpr_uexpr2, t1, bplus );
        BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr3, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr3  ) ) );

        // bexpr_uexpr2 = t1 + (((t1+1) + s2) + t1)
        auto bexpr_bexpr_uexpr4 = ublas::detail::make_binary_tensor_expression<tensor_type1>( t1, bexpr_bexpr_uexpr3, bplus );
        BOOST_CHECK( ! ublas::detail::all_extents_equal( bexpr_bexpr_uexpr4, ublas::detail::retrieve_extents( bexpr_bexpr_uexpr4  ) ) );
      }

    });
  });

}

BOOST_AUTO_TEST_SUITE_END()
