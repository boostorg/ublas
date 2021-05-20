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



#include <random>
#include <boost/numeric/ublas/tensor.hpp>

#ifndef BOOST_TEST_DYN_LINK
#define BOOST_TEST_DYN_LINK 
#endif
// NOLINTNEXTLINE
#define BOOST_TEST_MODULE Tensor


#include <boost/test/unit_test.hpp>
#include "utility.hpp"

BOOST_AUTO_TEST_SUITE ( test_tensor )

using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;


BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_ctor, value,  test_types)
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;
    using tensor_type  = ublas::tensor_dynamic<value_type,layout_type>;

//    auto a1 = tensor_type{};
//    BOOST_CHECK_EQUAL( a1.size() , 0ul );
//    BOOST_CHECK( a1.empty() );
//    BOOST_CHECK_EQUAL( a1.data() , nullptr);

    auto a2 = tensor_type{1,1};
    BOOST_CHECK_EQUAL(  a2.size() , 1 );
    BOOST_CHECK( !a2.empty() );
    BOOST_CHECK_NE(  a2.data() , nullptr);

    auto a3 = tensor_type{2,1};
    BOOST_CHECK_EQUAL(  a3.size() , 2 );
    BOOST_CHECK( !a3.empty() );
    BOOST_CHECK_NE(  a3.data() , nullptr);

    auto a4 = tensor_type{1,2};
    BOOST_CHECK_EQUAL(  a4.size() , 2 );
    BOOST_CHECK( !a4.empty() );
    BOOST_CHECK_NE(  a4.data() , nullptr);

    auto a5 = tensor_type{2,1};
    BOOST_CHECK_EQUAL(  a5.size() , 2 );
    BOOST_CHECK( !a5.empty() );
    BOOST_CHECK_NE(  a5.data() , nullptr);

    auto a6 = tensor_type{4,3,2};
    BOOST_CHECK_EQUAL(  a6.size() , 4*3*2 );
    BOOST_CHECK( !a6.empty() );
    BOOST_CHECK_NE(  a6.data() , nullptr);

    auto a7 = tensor_type{4,1,2};
    BOOST_CHECK_EQUAL(  a7.size() , 4*1*2 );
    BOOST_CHECK( !a7.empty() );
    BOOST_CHECK_NE(  a7.data() , nullptr);


}


struct fixture
{
    using extents_type = boost::numeric::ublas::extents<>;
    fixture()
      : extents {
          extents_type{1,1}, // 1
          extents_type{1,2}, // 2
          extents_type{2,1}, // 3
          extents_type{2,3}, // 4
          extents_type{2,3,1}, // 5
          extents_type{4,1,3}, // 6
          extents_type{1,2,3}, // 7
          extents_type{4,2,3}, // 8
          extents_type{4,2,3,5}} // 9
    {
    }
    std::vector<extents_type> extents;
};


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ctor_extents, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;
    using tensor_type  = ublas::tensor_dynamic<value_type,layout_type>;

    auto check = [](auto const& e) {
        auto t = tensor_type{e};
        BOOST_CHECK_EQUAL (  t.size() , ublas::product(e) );
        BOOST_CHECK_EQUAL (  t.rank() , ublas::size(e) );
        if(ublas::empty(e)) {
            BOOST_CHECK       ( t.empty()    );
            BOOST_CHECK_EQUAL ( t.data() , nullptr);
        }
        else{
            BOOST_CHECK       ( !t.empty()    );
            BOOST_CHECK_NE    (  t.data() , nullptr);
        }
    };

    for(auto const& e : extents)
        check(e);
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_copy_ctor, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;
    using tensor_type  = ublas::tensor_dynamic<value_type,layout_type>;

    auto check = [](auto const& e)
    {
        auto r = tensor_type{e};
        auto t = r;
        BOOST_CHECK_EQUAL (  t.size() , r.size() );
        BOOST_CHECK_EQUAL (  t.rank() , r.rank() );
        BOOST_CHECK ( t.strides() == r.strides() );
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
    };

    for(auto const& e : extents)
        check(e);
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_copy_ctor_layout, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;
    using tensor_type  = ublas::tensor_dynamic<value_type,layout_type>;
    using other_layout_type = std::conditional_t<std::is_same<ublas::layout::first_order,layout_type>::value, ublas::layout::last_order, ublas::layout::first_order>;
    using other_tensor_type = ublas::tensor_dynamic<value_type, other_layout_type>;


    for(auto const& e : extents)
    {
        auto r = tensor_type{e};
        other_tensor_type t = r;
        tensor_type q = t;

        BOOST_CHECK_EQUAL (  t.size() , r.size() );
        BOOST_CHECK_EQUAL (  t.rank() , r.rank() );
        BOOST_CHECK ( t.extents() == r.extents() );

        BOOST_CHECK_EQUAL (  q.size() , r.size() );
        BOOST_CHECK_EQUAL (  q.rank() , r.rank() );
        BOOST_CHECK ( q.strides() == r.strides() );
        BOOST_CHECK ( q.extents() == r.extents() );

        for(auto i = 0ul; i < t.size(); ++i)
            BOOST_CHECK_EQUAL( q[i], r[i]  );
    }
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_copy_move_ctor, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;
    using tensor_type = ublas::tensor_dynamic<value_type,layout_type>;

    auto check = [](auto const& e)
    {
        auto r = tensor_type{e};
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

    for(auto const& e : extents)
        check(e);
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ctor_extents_init, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;
    using tensor_type = ublas::tensor_dynamic<value_type,layout_type>;

    std::random_device device{};
    std::minstd_rand0 generator(device());

    using distribution_type = std::conditional_t<std::is_integral_v<value_type>, std::uniform_int_distribution<>, std::uniform_real_distribution<> >;
    auto distribution = distribution_type(1,6);

    for(auto const& e : extents){
        auto r = value_type( static_cast< inner_type_t<value_type> >(distribution(generator)) );
        auto t = tensor_type{e,r};
        for(auto i = 0ul; i < t.size(); ++i)
            BOOST_CHECK_EQUAL( t[i], r );
    }
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ctor_extents_array, value,  test_types, fixture)
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;
    using tensor_type = ublas::tensor_dynamic<value_type,layout_type>;
    using container_type  = typename tensor_type::container_type;

    for(auto const& e : extents) {
        auto a = container_type(product(e));
        auto v = value_type {};

        for(auto& aa : a){
            aa = v;
            v += value_type{1};
        }
        auto t = tensor_type{e, a};
        v = value_type{};

        for(auto i = 0ul; i < t.size(); ++i, v+=value_type{1})
            BOOST_CHECK_EQUAL( t[i], v);
    }
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_read_write_single_index_access, value,  test_types, fixture)
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;
    using tensor_type = ublas::tensor_dynamic<value_type,layout_type>;

    for(auto const& e : extents) {
        auto t = tensor_type{e};
        auto v = value_type {};
        for(auto i = 0ul; i < t.size(); ++i, v+=value_type{1}){
            t[i] = v;
            BOOST_CHECK_EQUAL( t[i], v );

            t(i) = v;
            BOOST_CHECK_EQUAL( t(i), v );
        }
    }
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_read_write_multi_index_access_at, value,  test_types, fixture)
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;
    using tensor_type = ublas::tensor_dynamic<value_type,layout_type>;
    auto check1 = [](const tensor_type& t)
    {
        auto v = value_type{};
        for(auto k = 0ul; k < t.size(); ++k){
            BOOST_CHECK_EQUAL(t[k], v);
            v+=value_type{1};
        }
    };

    auto check2 = [](const tensor_type& t)
    {
      std::array<unsigned,2> k = {0,0};
        auto r = std::is_same<layout_type,ublas::layout::first_order>::value ? 1 : 0;
        auto q = std::is_same<layout_type,ublas::layout::last_order >::value ? 1 : 0;
        auto v = value_type{};
        for(k[r] = 0ul; k[r] < t.size(r); ++k[r]){
            for(k[q] = 0ul; k[q] < t.size(q); ++k[q]){
                BOOST_CHECK_EQUAL(t.at(k[0],k[1]), v);
                v+=value_type{1};
            }
        }
    };

    auto check3 = [](const tensor_type& t)
    {
        std::array<unsigned,3> k = {0,0,0};
        using op_type = std::conditional_t<std::is_same_v<layout_type,ublas::layout::first_order>, std::minus<>, std::plus<>>;
        auto r = std::is_same_v<layout_type,ublas::layout::first_order> ? 2 : 0;
        auto o = op_type{};
        auto v = value_type{};
        for(k[r] = 0ul; k[r] < t.size(r); ++k[r]){
            for(k[o(r,1)] = 0ul; k[o(r,1)] < t.size(o(r,1)); ++k[o(r,1)]){
                for(k[o(r,2)] = 0ul; k[o(r,2)] < t.size(o(r,2)); ++k[o(r,2)]){
                    BOOST_CHECK_EQUAL(t.at(k[0],k[1],k[2]), v);
                    v+=value_type{1};
                }
            }
        }
    };

    auto check4 = [](const tensor_type& t)
    {
        std::array<unsigned,4> k = {0,0,0,0};
        using op_type = std::conditional_t<std::is_same_v<layout_type,ublas::layout::first_order>, std::minus<>, std::plus<>>;
        auto r = std::is_same_v<layout_type,ublas::layout::first_order> ? 3 : 0;
        auto o = op_type{};
        auto v = value_type{};
        for(k[r] = 0ul; k[r] < t.size(r); ++k[r]){
            for(k[o(r,1)] = 0ul; k[o(r,1)] < t.size(o(r,1)); ++k[o(r,1)]){
                for(k[o(r,2)] = 0ul; k[o(r,2)] < t.size(o(r,2)); ++k[o(r,2)]){
                    for(k[o(r,3)] = 0ul; k[o(r,3)] < t.size(o(r,3)); ++k[o(r,3)]){
                        BOOST_CHECK_EQUAL(t.at(k[0],k[1],k[2],k[3]), v);
                        v+=value_type{1};
                    }
                }
            }
        }
    };

    auto check = [check1,check2,check3,check4](auto const& e) {
        auto t = tensor_type{e};
        auto v = value_type {};
        for(auto i = 0ul; i < t.size(); ++i){
            t[i] = v;
            v+=value_type{1};
        }

        if(t.rank() == 1) check1(t);
        else if(t.rank() == 2) check2(t);
        else if(t.rank() == 3) check3(t);
        else if(t.rank() == 4) check4(t);

    };

    for(auto const& e : extents)
        check(e);
}




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_reshape, value,  test_types, fixture)
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;
    using tensor_type = ublas::tensor_dynamic<value_type,layout_type>;

    for(auto const& efrom : extents){
        for(auto const& eto : extents){

            auto v = value_type {};
            v+=value_type{1};
            auto t = tensor_type{efrom, v};
            for(auto i = 0ul; i < t.size(); ++i)
                BOOST_CHECK_EQUAL( t[i], v );

            auto r = reshape(t,eto);
            for(auto i = 0ul; i < std::min(ublas::product(efrom),ublas::product(eto)); ++i)
                BOOST_CHECK_EQUAL( r[i], v );

            BOOST_CHECK_EQUAL (  r.size() , ublas::product(eto) );
            BOOST_CHECK_EQUAL (  r.rank() , ublas::size   (eto) );
            BOOST_CHECK ( r.extents() == eto );

            if(efrom != eto){
                for(auto i = product(efrom); i < t.size(); ++i)
                    BOOST_CHECK_EQUAL( r[i], value_type{} );
            }
        }
    }
}




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_swap, value,  test_types, fixture)
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;
    using tensor_type = ublas::tensor_dynamic<value_type,layout_type>;
    for(auto const& e_t : extents){
        for(auto const& e_r : extents) {

            auto v = value_type {} + value_type{1};
            auto w = value_type {} + value_type{2};
            auto t = tensor_type{e_t, v};
            auto r = tensor_type{e_r, w};

            std::swap( r, t );

            for(auto i = 0ul; i < t.size(); ++i)
                BOOST_CHECK_EQUAL( t[i], w );

            BOOST_CHECK_EQUAL (  t.size() , ublas::product(e_r) );
            BOOST_CHECK_EQUAL (  t.rank() , ublas::size   (e_r) );
            BOOST_CHECK ( t.extents() == e_r );

            for(auto i = 0ul; i < r.size(); ++i)
                BOOST_CHECK_EQUAL( r[i], v );

            BOOST_CHECK_EQUAL (  r.size() , ublas::product(e_t) );
            BOOST_CHECK_EQUAL (  r.rank() , ublas::size   (e_t) );
            BOOST_CHECK ( r.extents() == e_t );


        }
    }
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_standard_iterator, value,  test_types, fixture)
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;
    using tensor_type = ublas::tensor_dynamic<value_type,layout_type>;

    for(auto const& e : extents)
    {
        auto v = value_type {} + value_type{1};
        auto t = tensor_type{e, v};

        BOOST_CHECK_EQUAL( std::distance(t.begin(),  t.end ()), t.size()  );
        BOOST_CHECK_EQUAL( std::distance(t.rbegin(), t.rend()), t.size()  );

        BOOST_CHECK_EQUAL( std::distance(t.cbegin(),  t.cend ()), t.size() );
        BOOST_CHECK_EQUAL( std::distance(t.crbegin(), t.crend()), t.size() );

        if(!t.empty()) {
            BOOST_CHECK(  t.data() ==  std::addressof( *t.begin () )  ) ;
            BOOST_CHECK(  t.data() ==  std::addressof( *t.cbegin() )  ) ;
        }
    }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_throw, value, test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
  using value_type  = typename value::first_type;
  using layout_type = typename value::second_type;
  using tensor_type = ublas::tensor_dynamic<value_type, layout_type>;

  std::vector<value_type> vec(2);
  BOOST_CHECK_THROW(tensor_type({5,5},vec), std::invalid_argument);

  auto t = tensor_type{{5,5}};
  auto i = ublas::index::index_type<4>{};
  BOOST_CHECK_THROW((void)t.operator()(i,i,i), std::invalid_argument);

}

BOOST_AUTO_TEST_SUITE_END()
