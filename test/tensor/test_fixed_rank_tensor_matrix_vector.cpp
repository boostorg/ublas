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
#include <random>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/test/unit_test.hpp>

#include "utility.hpp"

BOOST_AUTO_TEST_SUITE ( test_tensor_static_rank_matrix_interoperability )

using test_types = zip<int,float>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;


BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_matrix_copy_ctor, value,  test_types)
{
  namespace ublas  = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout = typename value::second_type;
    using tensor = ublas::tensor_static_rank<value_type, 2,layout>;
    using matrix = typename tensor::matrix_type;

    auto a2 = tensor( matrix(1,1) );
    BOOST_CHECK_EQUAL(  a2.size() , 1 );
    BOOST_CHECK( !a2.empty() );
    BOOST_CHECK_NE(  a2.data() , nullptr);

    auto a3 = tensor( matrix(2,1) );
    BOOST_CHECK_EQUAL(  a3.size() , 2 );
    BOOST_CHECK( !a3.empty() );
    BOOST_CHECK_NE(  a3.data() , nullptr);

    auto a4 = tensor( matrix(1,2) );
    BOOST_CHECK_EQUAL(  a4.size() , 2 );
    BOOST_CHECK( !a4.empty() );
    BOOST_CHECK_NE(  a4.data() , nullptr);

    auto a5 = tensor( matrix(2,3) );
    BOOST_CHECK_EQUAL(  a5.size() , 6 );
    BOOST_CHECK( !a5.empty() );
    BOOST_CHECK_NE(  a5.data() , nullptr);
}


BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_vector_copy_ctor, value,  test_types)
{
  namespace ublas  = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;
    using tensor_type  = ublas::tensor_static_rank<value_type, 2,layout_type>;
    using vector_type = typename tensor_type::vector_type;

    auto a2 = tensor_type( vector_type(1) );
    BOOST_CHECK_EQUAL(  a2.size() , 1 );
    BOOST_CHECK( !a2.empty() );
    BOOST_CHECK_NE(  a2.data() , nullptr);

    auto a3 = tensor_type( vector_type(2) );
    BOOST_CHECK_EQUAL(  a3.size() , 2 );
    BOOST_CHECK( !a3.empty() );
    BOOST_CHECK_NE(  a3.data() , nullptr);

    auto a4 = tensor_type( vector_type(2) );
    BOOST_CHECK_EQUAL(  a4.size() , 2 );
    BOOST_CHECK( !a4.empty() );
    BOOST_CHECK_NE(  a4.data() , nullptr);

    auto a5 = tensor_type( vector_type(3) );
    BOOST_CHECK_EQUAL(  a5.size() , 3 );
    BOOST_CHECK( !a5.empty() );
    BOOST_CHECK_NE(  a5.data() , nullptr);
}


struct fixture
{
    template<size_t N>
    using extents_type = boost::numeric::ublas::extents<N>;

    std::tuple<
        extents_type<2>, // 0
        extents_type<2>, // 1
        extents_type<2>, // 2
        extents_type<2>, // 3
        extents_type<2>  // 4
    > extents = {
        {1,1},
        {1,2},
        {2,1},
        {6,6},
        {9,7},        
    };
};




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_matrix_copy_ctor_extents, value,  test_types, fixture )
{
  namespace ublas  = boost::numeric::ublas;
  using value_type  = typename value::first_type;
  using layout_type = typename value::second_type;

  auto check = [](auto const& /*unused*/, auto& e) {
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(e)>>;
    using tensor = ublas::tensor_static_rank<value_type, size,layout_type>;
    using matrix = typename tensor::matrix_type;

    assert(ublas::size(e)==2);
    tensor t = matrix{e[0],e[1]};
    BOOST_CHECK_EQUAL (  t.size() , ublas::product(e) );
    BOOST_CHECK_EQUAL (  t.rank() , ublas::size   (e) );
    BOOST_CHECK       ( !t.empty()    );
    BOOST_CHECK_NE    (  t.data() , nullptr);
  };

  for_each_in_tuple(extents,check);
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_vector_copy_ctor_extents, value,  test_types, fixture )
{
  namespace ublas  = boost::numeric::ublas;
  using value_type  = typename value::first_type;
  using layout_type = typename value::second_type;


  auto check = [](auto const& /*unused*/, auto& e) {
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(e)>>;
    using tensor = ublas::tensor_static_rank<value_type, size,layout_type>;
    using vector = typename tensor::vector_type;

    assert(ublas::size(e)==2);
    if(ublas::empty(e))
      return;

    tensor t = vector (product(e));
    BOOST_CHECK_EQUAL (  t.size() , ublas::product(e) );
    BOOST_CHECK_EQUAL (  t.rank() , ublas::size   (e) );
    BOOST_CHECK       ( !t.empty()    );
    BOOST_CHECK_NE    (  t.data() , nullptr);
  };

  for_each_in_tuple(extents,check);
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_matrix_copy_assignment, value,  test_types, fixture )
{
  namespace ublas  = boost::numeric::ublas;
  using value_type  = typename value::first_type;
  using layout_type = typename value::second_type;


  for_each_in_tuple(extents, [](auto const& /*unused*/, auto& e) {
    using etype  = std::decay_t<decltype(e)>;
    constexpr auto size = std::tuple_size_v<etype>;
    using tensor = ublas::tensor_static_rank<value_type,size,layout_type>;
    using matrix = typename tensor::matrix_type;

    assert(ublas::size(e) == 2);
    auto t = tensor{e[1],e[0]};
    auto r = matrix(e[0],e[1]);
    std::iota(r.data().begin(),r.data().end(), 1);
    t = r;

    BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0) );
    BOOST_CHECK_EQUAL (  t.extents().at(1) , e.at(1) );
    BOOST_CHECK_EQUAL (  t.size() , ublas::product(e) );
    BOOST_CHECK_EQUAL (  t.rank() , ublas::size   (e) );
    BOOST_CHECK       ( !t.empty()    );
    BOOST_CHECK_NE    (  t.data() , nullptr);

    for(auto j = 0ul; j < t.size(1); ++j){
      for(auto i = 0ul; i < t.size(0); ++i){
        BOOST_CHECK_EQUAL( t.at(i,j), r(i,j)  );
      }
    }
  });

  //for_each_in_tuple(extents,check);
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_vector_copy_assignment, value,  test_types, fixture )
{
  namespace ublas  = boost::numeric::ublas;
  using value_type  = typename value::first_type;
  using layout_type = typename value::second_type;

  auto check = [](auto const& /*unused*/, auto& e) {
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(e)>>;
    using tensor_type = ublas::tensor_static_rank<value_type, size,layout_type>;
    using vector_type = typename tensor_type::vector_type;

    assert(ublas::size(e) == 2);
    auto t = tensor_type{e[1],e[0]};
    auto r = vector_type(e[0]*e[1]);
    std::iota(r.data().begin(),r.data().end(), 1);
    t = r;

    BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0)*e.at(1) );
    BOOST_CHECK_EQUAL (  t.extents().at(1) , 1);
    BOOST_CHECK_EQUAL (  t.size() , ublas::product(e) );
    BOOST_CHECK_EQUAL (  t.rank() , ublas::size   (e) );
    BOOST_CHECK       ( !t.empty()    );
    BOOST_CHECK_NE    (  t.data() , nullptr);

    for(auto i = 0ul; i < t.size(); ++i){
      BOOST_CHECK_EQUAL( t[i], r(i)  );
    }
  };

  for_each_in_tuple(extents,check);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_matrix_move_assignment, value,  test_types, fixture )
{
  namespace ublas  = boost::numeric::ublas;
  using value_type  = typename value::first_type;
  using layout_type = typename value::second_type;


  auto check = [](auto const& /*unused*/, auto& e) {
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(e)>>;
    using tensor_type = ublas::tensor_static_rank<value_type, size,layout_type>;
    using matrix_type = typename tensor_type::matrix_type;

    assert(ublas::size(e) == 2);
    auto t = tensor_type{e[1],e[0]};
    auto r = matrix_type(e[0],e[1]);
    std::iota(r.data().begin(),r.data().end(), 1);
    auto q = r;
    t = std::move(r);

    BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0) );
    BOOST_CHECK_EQUAL (  t.extents().at(1) , e.at(1) );
    BOOST_CHECK_EQUAL (  t.size() , ublas::product(e) );
    BOOST_CHECK_EQUAL (  t.rank() , ublas::size   (e) );
    BOOST_CHECK       ( !t.empty()    );
    BOOST_CHECK_NE    (  t.data() , nullptr);

    for(auto j = 0ul; j < t.size(1); ++j){
      for(auto i = 0ul; i < t.size(0); ++i){
        BOOST_CHECK_EQUAL( t.at(i,j), q(i,j)  );
      }
    }
  };

  for_each_in_tuple(extents,check);
}




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_vector_move_assignment, value,  test_types, fixture )
{
  namespace ublas  = boost::numeric::ublas;
  using value_type  = typename value::first_type;
  using layout_type = typename value::second_type;

  auto check = [](auto const& /*unused*/, auto& e) {
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(e)>>;
    using tensor_type = ublas::tensor_static_rank<value_type, size,layout_type>;
    using vector_type = typename tensor_type::vector_type;

    assert(ublas::size(e) == 2);
    auto t = tensor_type{e[1],e[0]};
    auto r = vector_type(e[0]*e[1]);
    std::iota(r.data().begin(),r.data().end(), 1);
    auto q = r;
    t = std::move(r);

    BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0) * e.at(1));
    BOOST_CHECK_EQUAL (  t.extents().at(1) , 1);
    BOOST_CHECK_EQUAL (  t.size() , ublas::product(e) );
    BOOST_CHECK_EQUAL (  t.rank() , ublas::size   (e) );
    BOOST_CHECK       ( !t.empty()    );
    BOOST_CHECK_NE    (  t.data() , nullptr);

    for(auto i = 0ul; i < t.size(); ++i){
      BOOST_CHECK_EQUAL( t[i], q(i)  );
    }
  };

  for_each_in_tuple(extents,check);
}





BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_matrix_expressions, value,  test_types, fixture )
{
  namespace ublas  = boost::numeric::ublas;
  using value_type  = typename value::first_type;
  using layout_type = typename value::second_type;


  auto check = [](auto const& /*unused*/, auto& e) {
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(e)>>;
    using tensor_type = ublas::tensor_static_rank<value_type, size,layout_type>;
    using matrix_type = typename tensor_type::matrix_type;

    assert(ublas::size(e) == 2);
    auto t = tensor_type{e[1],e[0]};
    auto r = matrix_type(e[0],e[1]);
    std::iota(r.data().begin(),r.data().end(), 1);
    t = r + 3*r;
    tensor_type s = r + 3*r;
    tensor_type q = s + r + 3*r + s; // + 3*r


    BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0) );
    BOOST_CHECK_EQUAL (  t.extents().at(1) , e.at(1) );
    BOOST_CHECK_EQUAL (  t.size() , ublas::product(e) );
    BOOST_CHECK_EQUAL (  t.rank() , ublas::size   (e) );
    BOOST_CHECK       ( !t.empty()    );
    BOOST_CHECK_NE    (  t.data() , nullptr);

    BOOST_CHECK_EQUAL (  s.extents().at(0) , e.at(0) );
    BOOST_CHECK_EQUAL (  s.extents().at(1) , e.at(1) );
    BOOST_CHECK_EQUAL (  s.size() , ublas::product(e) );
    BOOST_CHECK_EQUAL (  s.rank() , ublas::size   (e) );
    BOOST_CHECK       ( !s.empty()    );
    BOOST_CHECK_NE    (  s.data() , nullptr);

    BOOST_CHECK_EQUAL (  q.extents().at(0) , e.at(0) );
    BOOST_CHECK_EQUAL (  q.extents().at(1) , e.at(1) );
    BOOST_CHECK_EQUAL (  q.size() , ublas::product(e) );
    BOOST_CHECK_EQUAL (  q.rank() , ublas::size   (e) );
    BOOST_CHECK       ( !q.empty()    );
    BOOST_CHECK_NE    (  q.data() , nullptr);


    for(auto j = 0ul; j < t.size(1); ++j){
      for(auto i = 0ul; i < t.size(0); ++i){
        BOOST_CHECK_EQUAL( t.at(i,j), 4*r(i,j)  );
        BOOST_CHECK_EQUAL( s.at(i,j), t.at(i,j)  );
        BOOST_CHECK_EQUAL( q.at(i,j), 3*s.at(i,j)  );
      }
    }
  };

  for_each_in_tuple(extents,check);
}






BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_vector_expressions, value,  test_types, fixture )
{
  namespace ublas  = boost::numeric::ublas;
  using value_type  = typename value::first_type;
  using layout_type = typename value::second_type;

  auto check = [](auto const& /*unused*/, auto& e) {
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(e)>>;
    using tensor_type = ublas::tensor_static_rank<value_type, size,layout_type>;
    using vector_type = typename tensor_type::vector_type;

    assert(ublas::size(e) == 2);
    auto t = tensor_type{e[1],e[0]};
    auto r = vector_type(e[0]*e[1]);
    std::iota(r.data().begin(),r.data().end(), 1);
    t = r + 3*r;
    tensor_type s = r + 3*r;
    tensor_type q = s + r + 3*r + s; // + 3*r


    BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0)*e.at(1) );
    BOOST_CHECK_EQUAL (  t.extents().at(1) , 1);
    BOOST_CHECK_EQUAL (  t.size() , ublas::product(e) );
    BOOST_CHECK_EQUAL (  t.rank() , ublas::size   (e) );
    BOOST_CHECK       ( !t.empty()    );
    BOOST_CHECK_NE    (  t.data() , nullptr);

    BOOST_CHECK_EQUAL (  s.extents().at(0) , e.at(0)*e.at(1) );
    BOOST_CHECK_EQUAL (  s.extents().at(1) , 1);
    BOOST_CHECK_EQUAL (  s.size() , ublas::product(e) );
    BOOST_CHECK_EQUAL (  s.rank() , ublas::size   (e) );
    BOOST_CHECK       ( !s.empty()    );
    BOOST_CHECK_NE    (  s.data() , nullptr);

    BOOST_CHECK_EQUAL (  q.extents().at(0) , e.at(0)*e.at(1) );
    BOOST_CHECK_EQUAL (  q.extents().at(1) , 1);
    BOOST_CHECK_EQUAL (  q.size() , ublas::product(e) );
    BOOST_CHECK_EQUAL (  q.rank() , ublas::size   (e) );
    BOOST_CHECK       ( !q.empty()    );
    BOOST_CHECK_NE    (  q.data() , nullptr);



    for(auto i = 0ul; i < t.size(); ++i){
      BOOST_CHECK_EQUAL( t.at(i), 4*r(i)  );
      BOOST_CHECK_EQUAL( s.at(i), t.at(i)  );
      BOOST_CHECK_EQUAL( q.at(i), 3*s.at(i)  );
    }
  };

  for_each_in_tuple(extents,check);
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_matrix_vector_expressions, pair,  test_types, fixture )
{
  namespace ublas  = boost::numeric::ublas;
  using value  = typename pair::first_type;
  using layout = typename pair::second_type;


  auto check = [](auto const& /*unused*/, auto& e) {
    constexpr auto size = std::tuple_size_v<std::decay_t<decltype(e)>>;
    using tensor = ublas::tensor_static_rank<value, size, layout>;
    using matrix = typename tensor::matrix_type;
    using vector = typename tensor::vector_type;

    if(product(e) <= 2)
      return;
    assert(ublas::size(e) == 2);
    auto Q = tensor{e[0],1};
    auto A = matrix(e[0],e[1]);
    auto b = vector(e[1]);
    auto c = vector(e[0]);
    std::iota(b.data().begin(),b.data().end(), 1);
    std::fill(A.data().begin(),A.data().end(), 1);
    std::fill(c.data().begin(),c.data().end(), 2);
    std::fill(Q.begin(),Q.end(), 2);

    tensor T = Q + (ublas::prod(A , b) + 2*c) + 3*Q;

    BOOST_CHECK_EQUAL (  T.extents().at(0) , Q.extents().at(0) );
    BOOST_CHECK_EQUAL (  T.extents().at(1) , Q.extents().at(1));
    BOOST_CHECK_EQUAL (  T.size() , Q.size() );
    BOOST_CHECK_EQUAL (  T.size() , c.size() );
    BOOST_CHECK_EQUAL (  T.rank() , Q.rank() );
    BOOST_CHECK       ( !T.empty()    );
    BOOST_CHECK_NE    (  T.data() , nullptr);

    const auto n   = e[1];
    const auto ab  = value(std::div(n*(n+1),2).quot);
    const auto ref = ab+4*Q(0)+2*c(0);
    BOOST_CHECK( std::all_of(T.begin(),T.end(), [ref](auto cc){ return ref == cc; }) );

//    for(auto i = 0ul; i < T.size(); ++i){
//      auto n = e[1];
//      auto ab = n * (n+1) / 2;
//      BOOST_CHECK_EQUAL( T(i), ab+4*Q(0)+2*c(0)  );
//    }

  };
  for_each_in_tuple(extents,check);
}


BOOST_AUTO_TEST_SUITE_END()
