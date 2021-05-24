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



#include <random>
#include <boost/numeric/ublas/tensor.hpp>

#include <boost/test/unit_test.hpp>
#include "utility.hpp"

BOOST_AUTO_TEST_SUITE ( test_tensor_static_rank )

using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;


BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_ctor, value,  test_types)
{
  namespace ublas = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;

  auto a2 = ublas::tensor_static_rank<value_t, 2,layout_t>{1,1};
  BOOST_CHECK_EQUAL(  a2.size() , 1 );
  BOOST_CHECK( !a2.empty() );
  BOOST_CHECK_NE(  a2.data() , nullptr);

  auto a3 = ublas::tensor_static_rank<value_t, 2,layout_t>{2,1};
  BOOST_CHECK_EQUAL(  a3.size() , 2 );
  BOOST_CHECK( !a3.empty() );
  BOOST_CHECK_NE(  a3.data() , nullptr);

  auto a4 = ublas::tensor_static_rank<value_t, 2,layout_t>{1,2};
  BOOST_CHECK_EQUAL(  a4.size() , 2 );
  BOOST_CHECK( !a4.empty() );
  BOOST_CHECK_NE(  a4.data() , nullptr);

  auto a5 = ublas::tensor_static_rank<value_t, 2,layout_t>{2,1};
  BOOST_CHECK_EQUAL(  a5.size() , 2 );
  BOOST_CHECK( !a5.empty() );
  BOOST_CHECK_NE(  a5.data() , nullptr);

  auto a6 = ublas::tensor_static_rank<value_t, 3,layout_t>{4,3,2};
  BOOST_CHECK_EQUAL(  a6.size() , 4*3*2 );
  BOOST_CHECK( !a6.empty() );
  BOOST_CHECK_NE(  a6.data() , nullptr);

  auto a7 = ublas::tensor_static_rank<value_t, 3,layout_t>{4,1,2};
  BOOST_CHECK_EQUAL(  a7.size() , 4*1*2 );
  BOOST_CHECK( !a7.empty() );
  BOOST_CHECK_NE(  a7.data() , nullptr);

}


struct fixture
{
  template<size_t N>
  using extents_t = boost::numeric::ublas::extents<N>;

  std::tuple<
    extents_t<2>, // 1
    extents_t<2>, // 2
    extents_t<3>, // 3
    extents_t<3>, // 4
    extents_t<4>  // 5
    > extents = {  
        extents_t<2>{1,1},
        extents_t<2>{2,3},
        extents_t<3>{4,1,3},
      extents_t<3>{4,2,3},
      extents_t<4>{4,2,3,5}
  };
};


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ctor_extents, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;

  for_each_in_tuple(extents, [](auto const& /*unused*/, auto& e){
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(e)>>;
    auto t = ublas::tensor_static_rank<value_t, size, layout_t>{e};

    BOOST_CHECK_EQUAL (  t.size() , ublas::product(e) );
    BOOST_CHECK_EQUAL (  t.rank() , ublas::size   (e) );
    if(ublas::empty(e)) {
      BOOST_CHECK       ( t.empty()    );
      BOOST_CHECK_EQUAL ( t.data() , nullptr);
    }
    else{
      BOOST_CHECK       ( !t.empty()    );
      BOOST_CHECK_NE    (  t.data() , nullptr);
    }
  });

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_copy_ctor, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;


  for_each_in_tuple(extents, [](auto const& /*unused*/, auto& e){
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(e)>>;
    auto r = ublas::tensor_static_rank<value_t, size, layout_t>{e};

    auto t = r;
    BOOST_CHECK_EQUAL (  t.size() , r.size() );
    BOOST_CHECK_EQUAL (  t.rank() , r.rank() );
    //    BOOST_CHECK ( t.strides() == r.strides() );
    BOOST_CHECK ( t.extents() == r.extents() );

    if(ublas::empty(e)) {
      BOOST_CHECK       ( t.empty()    );
      BOOST_CHECK_EQUAL ( t.data() , nullptr);
    }
    else{
      BOOST_CHECK       ( !t.empty()    );
      BOOST_CHECK_NE    (  t.data() , nullptr);
    }

    for(auto i = 0ul; i < t.size(); ++i)
      BOOST_CHECK_EQUAL( t[i], r[i]  );

  });
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_copy_ctor_layout, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;
  using other_layout_t = std::conditional_t<std::is_same<ublas::layout::first_order,layout_t>::value, ublas::layout::last_order, ublas::layout::first_order>;


  for_each_in_tuple(extents, [](auto const& /*unused*/, auto& e){
    constexpr auto size     = std::tuple_size_v<std::decay_t<decltype(e)>>;
    using tensor_t       = ublas::tensor_static_rank<value_t, size, layout_t>;
    using other_tensor_t = ublas::tensor_static_rank<value_t, size, other_layout_t>;
    auto r = tensor_t(e);
    other_tensor_t t = r;
    tensor_t q = t;

    BOOST_CHECK_EQUAL (  t.size() , r.size() );
    BOOST_CHECK_EQUAL (  t.rank() , r.rank() );
    BOOST_CHECK ( t.extents() == r.extents() );

    BOOST_CHECK_EQUAL (  q.size() , r.size() );
    BOOST_CHECK_EQUAL (  q.rank() , r.rank() );
    //    BOOST_CHECK ( q.strides() == r.strides() );
    BOOST_CHECK ( q.extents() == r.extents() );

    for(auto i = 0ul; i < t.size(); ++i)
      BOOST_CHECK_EQUAL( q[i], r[i]  );

  });
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_copy_move_ctor, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;

  auto check = [](auto const& /*unused*/, auto& e)
  {
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(e)>>;
    using tensor_t = ublas::tensor_static_rank<value_t, size, layout_t>;
    auto r = tensor_t{e};
    auto t = std::move(r);
    BOOST_CHECK_EQUAL (  t.size() , ublas::product(e) );
    BOOST_CHECK_EQUAL (  t.rank() , ublas::size   (e) );

    if(ublas::empty(e)) {
      BOOST_CHECK       ( t.empty()    );
      BOOST_CHECK_EQUAL ( t.data() , nullptr);
    }
    else{
      BOOST_CHECK       ( !t.empty()    );
      BOOST_CHECK_NE    (  t.data() , nullptr);
    }

  };

  for_each_in_tuple(extents,check);
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ctor_extents_init, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;

  std::random_device device{};
  std::minstd_rand0 generator(device());

  using distribution_type = std::conditional_t<std::is_integral_v<value_t>, std::uniform_int_distribution<>, std::uniform_real_distribution<> >;
  auto distribution = distribution_type(1,6);

  for_each_in_tuple(extents, [&](auto const& /*unused*/, auto const& e){
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(e)>>;
    using tensor_t = ublas::tensor_static_rank<value_t, size, layout_t>;

    auto r = value_t( static_cast< inner_type_t<value_t> >(distribution(generator)) );
    auto t = tensor_t(e);
    t = r;
    for(auto i = 0ul; i < t.size(); ++i)
      BOOST_CHECK_EQUAL( t[i], r );

  });
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ctor_extents_array, value,  test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;

  for_each_in_tuple(extents, [](auto const& /*unused*/, auto& e){
    constexpr auto size  = std::tuple_size_v<std::decay_t<decltype(e)>>;
    using tensor_t    = ublas::tensor_static_rank<value_t, size, layout_t>;
    using container_t = typename tensor_t::container_type;

    auto a = container_t(product(e));
    auto v = value_t {};

    for(auto& aa : a){
      aa = v;
      v += value_t{1};
    }
    auto t = tensor_t(e, a);
    v = value_t{};

    for(auto i = 0ul; i < t.size(); ++i, v+=value_t{1})
      BOOST_CHECK_EQUAL( t[i], v);

  });
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_read_write_single_index_access, value,  test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;

  for_each_in_tuple(extents, [](auto const& /*unused*/, auto& e){
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(e)>>;
    using tensor_t = ublas::tensor_static_rank<value_t, size, layout_t>;

    auto t = tensor_t{e};
    auto v = value_t {};
    for(auto i = 0ul; i < t.size(); ++i, v+=value_t{1}){
      t[i] = v;
      BOOST_CHECK_EQUAL( t[i], v );

      t(i) = v;
      BOOST_CHECK_EQUAL( t(i), v );
    }

  });
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_read_write_multi_index_access_at, value,  test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;

  auto check1 = [](const auto& t)
  {
    auto v = value_t{};
    for(auto k = 0ul; k < t.size(); ++k){
      BOOST_CHECK_EQUAL(t[k], v);
      v+=value_t{1};
    }
  };

  auto check2 = [](const auto& t)
  {
    std::array<unsigned,2> k = {0,0};
    auto r = std::is_same<layout_t,ublas::layout::first_order>::value ? 1 : 0;
    auto q = std::is_same<layout_t,ublas::layout::last_order >::value ? 1 : 0;
    auto v = value_t{};
    for(k[r] = 0ul; k[r] < t.size(r); ++k[r]){
      for(k[q] = 0ul; k[q] < t.size(q); ++k[q]){
        BOOST_CHECK_EQUAL(t.at(k[0],k[1]), v);
        v+=value_t{1};
      }
    }
  };

  auto check3 = [](const auto& t)
  {
    std::array<unsigned,3> k = {0,0,0};
    using op_type = std::conditional_t<std::is_same_v<layout_t,ublas::layout::first_order>, std::minus<>, std::plus<>>;
    auto r = std::is_same_v<layout_t,ublas::layout::first_order> ? 2 : 0;
    auto o = op_type{};
    auto v = value_t{};
    for(k[r] = 0ul; k[r] < t.size(r); ++k[r]){
      for(k[o(r,1)] = 0ul; k[o(r,1)] < t.size(o(r,1)); ++k[o(r,1)]){
        for(k[o(r,2)] = 0ul; k[o(r,2)] < t.size(o(r,2)); ++k[o(r,2)]){
          BOOST_CHECK_EQUAL(t.at(k[0],k[1],k[2]), v);
          v+=value_t{1};
        }
      }
    }
  };

  auto check4 = [](const auto& t)
  {
    static constexpr auto order = 4;
    std::array<unsigned,order> k = {0,0,0,0};
    using op_type = std::conditional_t<std::is_same_v<layout_t,ublas::layout::first_order>, std::minus<>, std::plus<>>;
    auto r = std::is_same_v<layout_t,ublas::layout::first_order> ? (order-1) : 0;
    auto o = op_type{};
    auto v = value_t{};
    for(k[o(r,0)] = 0ul; k[o(r,0)] < t.size(o(r,0)); ++k[o(r,0)]){
      for(k[o(r,1)] = 0ul; k[o(r,1)] < t.size(o(r,1)); ++k[o(r,1)]){
        for(k[o(r,2)] = 0ul; k[o(r,2)] < t.size(o(r,2)); ++k[o(r,2)]){
          for(k[o(r,3)] = 0ul; k[o(r,3)] < t.size(o(r,3)); ++k[o(r,3)]){
            BOOST_CHECK_EQUAL(t.at(k[0],k[1],k[2],k[3]), v);
            v+=value_t{1};
          }
        }
      }
    }
  };

  auto check = [check1,check2,check3,check4](auto const& /*unused*/, auto const& e) {
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(e)>>;
    using tensor_t = ublas::tensor_static_rank<value_t, size, layout_t>;
    auto t = tensor_t(e);
    auto v = value_t {};
    for(auto i = 0ul; i < t.size(); ++i){
      t[i] = v;
      v+=value_t{1};
    }

    if constexpr(size == 1) check1(t);
    else if constexpr(size == 2) check2(t);
    else if constexpr(size == 3) check3(t);
    else if constexpr(size == 4) check4(t);

  };

  for_each_in_tuple(extents,check);
}




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_reshape, value,  test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;

  for_each_in_tuple(extents,[&](auto const& /*unused*/, auto const& efrom){

    using efrom_t = std::decay_t<decltype(efrom)>;
    using tensor_t = ublas::tensor_static_rank<value_t, std::tuple_size_v<efrom_t>, layout_t>;

    for_each_in_tuple(extents,[&](auto const& /*unused*/, auto& eto){
      using eto_t = std::decay_t<decltype(eto)>;

      if constexpr( std::tuple_size_v<efrom_t> == std::tuple_size_v<eto_t>  ){

        auto v = value_t {};
        v+=value_t{1};
        auto t = tensor_t(efrom);
        t = v;
        for(auto i = 0ul; i < t.size(); ++i)
          BOOST_CHECK_EQUAL( t[i], v );

        auto t2 = reshape(t,eto);
        for(auto i = 0ul; i < std::min(ublas::product(efrom),ublas::product(eto)); ++i)
          BOOST_CHECK_EQUAL( t2[i], v );

        BOOST_CHECK_EQUAL (  t2.size() , ublas::product(eto) );
        BOOST_CHECK_EQUAL (  t2.rank() , ublas::size   (eto) );
        BOOST_CHECK       ( t2.extents() == eto );

        if(efrom != eto){
          for(auto i = product(efrom); i < t.size(); ++i)
            BOOST_CHECK_EQUAL( t2[i], value_t{} );
        }
      }
    });
  });
}




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_swap, value,  test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;

  for_each_in_tuple(extents,[&](auto const& /*unused*/, auto const& e_t){
    using e_tt = std::decay_t<  decltype(e_t)  >;
    using tensor_t = ublas::tensor_static_rank<value_t, std::tuple_size_v<e_tt>, layout_t>;

    for_each_in_tuple(extents,[&](auto const& /*unused*/, auto& e_r){

      using e_rt = std::decay_t<  decltype(e_r)  >;

      if constexpr( std::tuple_size_v<e_tt> == std::tuple_size_v<e_rt> ){

        auto v = value_t {} + value_t{1};
        auto w = value_t {} + value_t{2};
        auto t = tensor_t(e_t);
        auto r = tensor_t(e_r);

        t = v;
        r = w;

        std::swap( r, t );

        BOOST_CHECK ( std::all_of(t.begin(),t.end(),[w](auto tt){return tt == w; } ) ) ;
        BOOST_CHECK_EQUAL (  t.size() , ublas::product(e_r) );
        BOOST_CHECK_EQUAL (  t.rank() , ublas::size   (e_r) );
        BOOST_CHECK ( t.extents() == e_r );
        BOOST_CHECK ( t.strides() == ublas::to_strides(e_r,layout_t{}) );

        BOOST_CHECK ( std::all_of(r.begin(),r.end(),[v](auto tt){return tt == v; } ) ) ;
        BOOST_CHECK_EQUAL (  r.size() , ublas::product(e_t) );
        BOOST_CHECK_EQUAL (  r.rank() , ublas::size   (e_t) );
        BOOST_CHECK ( r.extents() == e_t );
        BOOST_CHECK ( r.strides() == ublas::to_strides(e_t,layout_t{}) );


      }
    });
  });
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_standard_iterator, value,  test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;

  for_each_in_tuple(extents,[](auto const& /*unused*/, auto const& e){
    using et = std::decay_t<  decltype(e)  >;
    using tensor_t = ublas::tensor_static_rank<value_t, std::tuple_size_v<et>, layout_t>;

    auto v = value_t {} + value_t{1};
    auto t = tensor_t(e);
    t = v;

    BOOST_CHECK_EQUAL( std::distance(t.begin(),  t.end ()), t.size()  );
    BOOST_CHECK_EQUAL( std::distance(t.rbegin(), t.rend()), t.size()  );

    BOOST_CHECK_EQUAL( std::distance(t.cbegin(),  t.cend ()), t.size() );
    BOOST_CHECK_EQUAL( std::distance(t.crbegin(), t.crend()), t.size() );

    if(!t.empty()) {
      BOOST_CHECK(  t.data() ==  std::addressof( *t.begin () )  ) ;
      BOOST_CHECK(  t.data() ==  std::addressof( *t.cbegin() )  ) ;
    }
  });

}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_throw, value, test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
  using value_t  = typename value::first_type;
  using layout_t = typename value::second_type;
  using tensor_t = ublas::tensor_static_rank<value_t, 2, layout_t>;

  auto vec = std::vector<value_t>(1);
  BOOST_CHECK_THROW(tensor_t({5,5},vec), std::length_error);

  // Does not throw but results in a static assertion
//  auto t = tensor_t{{5,5}};
//  auto i = ublas::index::index_type<4>{};
//  BOOST_CHECK_THROW((void)t.operator()(i,i,i), std::runtime_error);

}

BOOST_AUTO_TEST_SUITE_END()
