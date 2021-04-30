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

BOOST_AUTO_TEST_SUITE ( test_static_tensor )

using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;


BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_ctor, value,  test_types)
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;

    auto a1 = ublas::static_tensor<value_type, ublas::static_extents<>,layout_type>{};
    BOOST_CHECK_EQUAL( a1.size() , 0ul );
    BOOST_CHECK( a1.empty() );

    auto a2 = ublas::static_tensor<value_type, ublas::static_extents<1,1>,layout_type>{};
    BOOST_CHECK_EQUAL(  a2.size() , 1 );
    BOOST_CHECK( !a2.empty() );

    auto a3 = ublas::static_tensor<value_type, ublas::static_extents<2,1>,layout_type>{};
    BOOST_CHECK_EQUAL(  a3.size() , 2 );
    BOOST_CHECK( !a3.empty() );

    auto a4 = ublas::static_tensor<value_type, ublas::static_extents<1,2>,layout_type>{};
    BOOST_CHECK_EQUAL(  a4.size() , 2 );
    BOOST_CHECK( !a4.empty() );

    auto a5 = ublas::static_tensor<value_type, ublas::static_extents<2,1>,layout_type>{};
    BOOST_CHECK_EQUAL(  a5.size() , 2 );
    BOOST_CHECK( !a5.empty() );

    auto a6 = ublas::static_tensor<value_type, ublas::static_extents<4,3,2>,layout_type>{};
    BOOST_CHECK_EQUAL(  a6.size() , 4*3*2 );
    BOOST_CHECK( !a6.empty() );

    auto a7 = ublas::static_tensor<value_type, ublas::static_extents<4,1,2>,layout_type>{};
    BOOST_CHECK_EQUAL(  a7.size() , 4*1*2 );
    BOOST_CHECK( !a7.empty() );

}


struct fixture
{
    template<size_t... N>
    using extents_type = boost::numeric::ublas::static_extents<N...>;

    fixture()=default;

    std::tuple<
        extents_type<1,1>,   // 1
        extents_type<2,3>,   // 2
        extents_type<4,1,3>,  // 3
        extents_type<4,2,3>,  // 4
        extents_type<4,2,3,5>   // 5
    > extents;
};


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ctor_extents, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;

    for_each_in_tuple(extents, [](auto const& /*unused*/, auto& e){
        using extents_type = std::decay_t<decltype(e)>;
        auto t = ublas::static_tensor<value_type, extents_type, layout_type>{};

        BOOST_CHECK_EQUAL (  t.size() , product(e) );
        BOOST_CHECK_EQUAL (  t.rank() , e.size() );
        if(e.empty()) {
            BOOST_CHECK       ( t.empty()    );
        }
        else{
            BOOST_CHECK       ( !t.empty()    );
        }
    });

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_copy_ctor, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;


    for_each_in_tuple(extents, [](auto const& /*unused*/, auto& e){
        using extents_type = std::decay_t<decltype(e)>;
        auto r = ublas::static_tensor<value_type, extents_type, layout_type>{0};

        auto t = r;
        BOOST_CHECK_EQUAL (  t.size() , r.size() );
        BOOST_CHECK_EQUAL (  t.rank() , r.rank() );
        BOOST_CHECK ( t.strides() == r.strides() );
        BOOST_CHECK ( t.extents() == r.extents() );

        if(e.empty()) {
            BOOST_CHECK       ( t.empty()    );
        }
        else{
            BOOST_CHECK       ( !t.empty()    );
        }

        for(auto i = 0ul; i < t.size(); ++i)
            BOOST_TEST( t[i] == r[i]);

    });
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_copy_ctor_layout, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;
    using other_layout_type = std::conditional_t<std::is_same<ublas::layout::first_order,layout_type>::value, ublas::layout::last_order, ublas::layout::first_order>;


    for_each_in_tuple(extents, [](auto const& /*unused*/, auto& e){
        using extents_type = std::decay_t<decltype(e)>;
        using tensor_type = ublas::static_tensor<value_type, extents_type, layout_type>;
        auto r = tensor_type{0};
        ublas::static_tensor<value_type, extents_type, other_layout_type> t = r;
        tensor_type q = t;

        BOOST_CHECK_EQUAL (  t.size() , r.size() );
        BOOST_CHECK_EQUAL (  t.rank() , r.rank() );
        BOOST_CHECK ( t.extents() == r.extents() );

        BOOST_CHECK_EQUAL (  q.size() , r.size() );
        BOOST_CHECK_EQUAL (  q.rank() , r.rank() );
        BOOST_CHECK ( q.strides() == r.strides() );
        BOOST_CHECK ( q.extents() == r.extents() );

        for(auto i = 0ul; i < t.size(); ++i)
            BOOST_TEST( q[i] == r[i]);

    });
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_copy_move_ctor, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;

    auto check = [](auto const& /*unused*/, auto& e)
    {
        using extents_type = std::decay_t<decltype(e)>;
        using tensor_type = ublas::static_tensor<value_type, extents_type, layout_type>;
        auto r = tensor_type{};
        auto t = std::move(r);
        BOOST_CHECK_EQUAL (  t.size() , product(e) );
        BOOST_CHECK_EQUAL (  t.rank() , e.size() );

        if(e.empty()) {
            BOOST_CHECK       ( t.empty()    );
        }
        else{
            BOOST_CHECK       ( !t.empty()    );
        }

    };

    for_each_in_tuple(extents,check);
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ctor_extents_init, value,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;

    std::random_device device{};
    std::minstd_rand0 generator(device());

    using distribution_type = std::conditional_t<std::is_integral_v<value_type>, std::uniform_int_distribution<>, std::uniform_real_distribution<> >;
    auto distribution = distribution_type(1,6);

    for_each_in_tuple(extents, [&](auto const& /*unused*/, auto const& e){
        using extents_type = std::decay_t<decltype(e)>;
        using tensor_type = ublas::static_tensor<value_type, extents_type, layout_type>;
        
        auto r = value_type( static_cast< inner_type_t<value_type> >(distribution(generator)) );
        auto t = tensor_type{r};
        for(auto i = 0ul; i < t.size(); ++i)
            BOOST_CHECK_EQUAL( t[i], r );

    });
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_ctor_extents_array, value,  test_types, fixture)
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;

    for_each_in_tuple(extents, [](auto const& /*unused*/, auto& e){
        using extents_type = std::decay_t<decltype(e)>;
        using tensor_type = ublas::static_tensor<value_type, extents_type, layout_type>;
        using array_type  = typename tensor_type::array_type;
        
        auto a = array_type();
        auto v = value_type {};

        for(auto& aa : a){
            aa = v;
            v += value_type{1};
        }
        auto t = tensor_type{a};
        v = value_type{};

        for(auto i = 0ul; i < t.size(); ++i, v+=value_type{1})
            BOOST_CHECK_EQUAL( t[i], v);

    });
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_read_write_single_index_access, value,  test_types, fixture)
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;

    for_each_in_tuple(extents, [](auto const& /*unused*/, auto& e){
        using extents_type = std::decay_t<decltype(e)>;
        using tensor_type = ublas::static_tensor<value_type, extents_type, layout_type>;
        
        auto t = tensor_type{};
        auto v = value_type {};
        for(auto i = 0ul; i < t.size(); ++i, v+=value_type{1}){
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
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;

    auto check1 = [](const auto& t)
    {
        auto v = value_type{};
        for(auto k = 0ul; k < t.size(); ++k){
            BOOST_CHECK_EQUAL(t[k], v);
            v+=value_type{1};
        }
    };

    auto check2 = [](const auto& t)
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

    auto check3 = [](const auto& t)
    {
        std::array<unsigned,3> k= {0,0,0};
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

    auto check4 = [](const auto& t)
    {
        std::array<unsigned,4> k= {0,0,0,0};
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

    auto check = [check1,check2,check3,check4](auto const& /*unused*/, auto const& e) {
        using extents_type = std::decay_t<decltype(e)>;
        using tensor_type = ublas::static_tensor<value_type, extents_type, layout_type>;
        auto t = tensor_type{};
        auto v = value_type {};
        for(auto i = 0ul; i < t.size(); ++i){
            t[i] = v;
            v+=value_type{1};
        }

        if constexpr ( extents_type::_size == 1) check1(t);
        else if constexpr ( extents_type::_size == 2) check2(t);
        else if constexpr ( extents_type::_size == 3) check3(t);
        else if constexpr ( extents_type::_size == 4) check4(t);

    };

    for_each_in_tuple(extents,check);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_standard_iterator, value,  test_types, fixture)
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;

    for_each_in_tuple(extents,[](auto const& /*unused*/, auto& e){
        using extents_type = std::decay_t<decltype(e)>;
        using tensor_type = ublas::static_tensor<value_type, extents_type, layout_type>;
        
        auto v = value_type {} + value_type{1};
        auto t = tensor_type{v};

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
  using value_type  = typename value::first_type;
  using layout_type = typename value::second_type;
  using tensor_type = ublas::static_tensor<value_type, ublas::static_extents<5,5>, layout_type>;

  auto t = tensor_type{};
  auto i = ublas::index::index_type<4>{};
  BOOST_CHECK_THROW((void)t.operator()(i,i,i), std::runtime_error);

}

BOOST_AUTO_TEST_SUITE_END()
