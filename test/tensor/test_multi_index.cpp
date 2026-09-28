//
// 	Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2021, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//


#include <boost/numeric/ublas/tensor/multi_index.hpp>
#include <boost/test/unit_test.hpp>
#include "fixture_utility.hpp"


BOOST_AUTO_TEST_SUITE(test_multi_index, 
    *boost::unit_test::description("Validate Multi-Index")
)


BOOST_AUTO_TEST_CASE(test_index_classes, 
    *boost::unit_test::label("index_value")
    *boost::unit_test::description("Testing index correctness"))
{
    namespace index = boost::numeric::ublas::index;
    
    BOOST_CHECK_EQUAL (  index::_a.value  ,  1  ) ;
    BOOST_CHECK_EQUAL (  index::_b.value  ,  2  ) ;
    BOOST_CHECK_EQUAL (  index::_c.value  ,  3  ) ;
    BOOST_CHECK_EQUAL (  index::_d.value  ,  4  ) ;
    BOOST_CHECK_EQUAL (  index::_e.value  ,  5  ) ;
    BOOST_CHECK_EQUAL (  index::_f.value  ,  6  ) ;
    BOOST_CHECK_EQUAL (  index::_g.value  ,  7  ) ;
    BOOST_CHECK_EQUAL (  index::_h.value  ,  8  ) ;
    BOOST_CHECK_EQUAL (  index::_i.value  ,  9  ) ;
    BOOST_CHECK_EQUAL (  index::_j.value  , 10  ) ;
    BOOST_CHECK_EQUAL (  index::_k.value  , 11  ) ;
    BOOST_CHECK_EQUAL (  index::_l.value  , 12  ) ;
    BOOST_CHECK_EQUAL (  index::_m.value  , 13  ) ;
    BOOST_CHECK_EQUAL (  index::_n.value  , 14  ) ;
    BOOST_CHECK_EQUAL (  index::_o.value  , 15  ) ;
    BOOST_CHECK_EQUAL (  index::_p.value  , 16  ) ;
    BOOST_CHECK_EQUAL (  index::_q.value  , 17  ) ;
    BOOST_CHECK_EQUAL (  index::_r.value  , 18  ) ;
    BOOST_CHECK_EQUAL (  index::_s.value  , 19  ) ;
    BOOST_CHECK_EQUAL (  index::_t.value  , 20  ) ;
    BOOST_CHECK_EQUAL (  index::_u.value  , 21  ) ;
    BOOST_CHECK_EQUAL (  index::_v.value  , 22  ) ;
    BOOST_CHECK_EQUAL (  index::_w.value  , 23  ) ;
    BOOST_CHECK_EQUAL (  index::_x.value  , 24  ) ;
    BOOST_CHECK_EQUAL (  index::_y.value  , 25  ) ;
    BOOST_CHECK_EQUAL (  index::_z.value  , 26  ) ;
}

BOOST_AUTO_TEST_CASE(test_multi_index_class_construction, 
    *boost::unit_test::label("index_construction")
    *boost::unit_test::description("Testing index construction"))
{
    namespace ublas = boost::numeric::ublas;
    namespace index = ublas::index;

    {
        ublas::multi_index<2> idx(index::_a, index::_b);
        BOOST_CHECK_EQUAL(ublas::get<0>(idx), 1ul);
        BOOST_CHECK_EQUAL(ublas::get<1>(idx), 2ul);
    }

    {
        ublas::multi_index<2> idx(index::_d, index::_c);
        BOOST_CHECK_EQUAL(idx[0], 4ul);
        BOOST_CHECK_EQUAL(idx[1], 3ul);
    }
    
   
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("multi_index_class_generation")
    *boost::unit_test::description("Testing multi-index generation for dynamic tensor")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_multi_index_class_generation_tensor_dynamic,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_dynamic<TestTupleType>
)
{
    namespace ublas = boost::numeric::ublas;
    namespace index = ublas::index;

    using value_type= typename TestTupleType::first_type;
    using fixture_t = ublas::tuple_fixture_tensor_dynamic<TestTupleType>;

    constexpr auto check1 = [](auto t, auto idx0, auto idx1){
        BOOST_TEST_CONTEXT("[Dynamic Tensor] multi-index(" << idx0() << ", " << idx1() <<")"){
            t = value_type{2};

            auto a_idx = t(idx0, idx1);

            BOOST_CHECK_EQUAL ( std::addressof( a_idx.first ), std::addressof( t ) ) ;

            BOOST_CHECK_EQUAL (std::get<0>(a_idx.second)(), idx0() ) ;
            BOOST_CHECK_EQUAL (std::get<1>(a_idx.second)(), idx1() ) ;
        }
    };

    constexpr auto check2 = [](auto t, auto idx0, auto idx1, auto idx2){
        BOOST_TEST_CONTEXT("[Dynamic Tensor] multi-index(" << idx0() << ", " << idx1() << idx2() <<")"){
            t = value_type{2};

            auto a_idx = t(idx0, idx1, idx2);

            BOOST_CHECK_EQUAL ( std::addressof( a_idx.first ), std::addressof( t ) ) ;

            BOOST_CHECK_EQUAL (std::get<0>(a_idx.second)(), idx0() ) ;
            BOOST_CHECK_EQUAL (std::get<1>(a_idx.second)(), idx1() ) ;
            BOOST_CHECK_EQUAL (std::get<2>(a_idx.second)(), idx2() ) ;
        }
    };

    check1(fixture_t::t32,  index::_a, index::_c);
    check1(fixture_t::t32,  index::_c, index::_a);
    check2(fixture_t::t432, index::_b, index::_b, index::_a);

}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("multi_index_class_generation")
    *boost::unit_test::description("Testing multi-index generation for static rank tensor")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_multi_index_class_generation_tensor_static_rank,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_static_rank<TestTupleType>
)
{
    namespace ublas = boost::numeric::ublas;
    namespace index = ublas::index;

    using value_type= typename TestTupleType::first_type;
    using fixture_t = ublas::tuple_fixture_tensor_static_rank<TestTupleType>;

    constexpr auto check1 = [](auto t, auto idx0, auto idx1){
        BOOST_TEST_CONTEXT("[Static Rank Tensor] multi-index(" << idx0() << ", " << idx1() <<")"){
            t = value_type{2};

            auto a_idx = t(idx0, idx1);

            BOOST_CHECK_EQUAL ( std::addressof( a_idx.first ), std::addressof( t ) ) ;

            BOOST_CHECK_EQUAL (std::get<0>(a_idx.second)(), idx0() ) ;
            BOOST_CHECK_EQUAL (std::get<1>(a_idx.second)(), idx1() ) ;
        }
    };

    constexpr auto check2 = [](auto t, auto idx0, auto idx1, auto idx2){
        BOOST_TEST_CONTEXT("[Static Rank Tensor] multi-index(" << idx0() << ", " << idx1() << idx2() <<")"){
            t = value_type{2};

            auto a_idx = t(idx0, idx1, idx2);

            BOOST_CHECK_EQUAL ( std::addressof( a_idx.first ), std::addressof( t ) ) ;

            BOOST_CHECK_EQUAL (std::get<0>(a_idx.second)(), idx0() ) ;
            BOOST_CHECK_EQUAL (std::get<1>(a_idx.second)(), idx1() ) ;
            BOOST_CHECK_EQUAL (std::get<2>(a_idx.second)(), idx2() ) ;
        }
    };

    check1(fixture_t::t32,  index::_a, index::_c);
    check1(fixture_t::t32,  index::_c, index::_a);
    check2(fixture_t::t432, index::_b, index::_b, index::_a);

}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("multi_index_class_generation")
    *boost::unit_test::description("Testing multi-index generation for static tensor")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_multi_index_class_generation_tensor_static,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_static<TestTupleType>
)
{
    namespace ublas = boost::numeric::ublas;
    namespace index = ublas::index;

    using value_type= typename TestTupleType::first_type;
    using fixture_t = ublas::tuple_fixture_tensor_static<TestTupleType>;

    constexpr auto check1 = [](auto t, auto idx0, auto idx1){
        BOOST_TEST_CONTEXT("[Static Tensor] multi-index(" << idx0() << ", " << idx1() <<")"){
            t = value_type{2};

            auto a_idx = t(idx0, idx1);

            BOOST_CHECK_EQUAL ( std::addressof( a_idx.first ), std::addressof( t ) ) ;

            BOOST_CHECK_EQUAL (std::get<0>(a_idx.second)(), idx0() ) ;
            BOOST_CHECK_EQUAL (std::get<1>(a_idx.second)(), idx1() ) ;
        }
    };

    constexpr auto check2 = [](auto t, auto idx0, auto idx1, auto idx2){
        BOOST_TEST_CONTEXT("[Static Tensor] multi-index(" << idx0() << ", " << idx1() << idx2() <<")"){
            t = value_type{2};

            auto a_idx = t(idx0, idx1, idx2);

            BOOST_CHECK_EQUAL ( std::addressof( a_idx.first ), std::addressof( t ) ) ;

            BOOST_CHECK_EQUAL (std::get<0>(a_idx.second)(), idx0() ) ;
            BOOST_CHECK_EQUAL (std::get<1>(a_idx.second)(), idx1() ) ;
            BOOST_CHECK_EQUAL (std::get<2>(a_idx.second)(), idx2() ) ;
        }
    };

    check1(fixture_t::t32,  index::_a, index::_c);
    check1(fixture_t::t32,  index::_c, index::_a);
    check2(fixture_t::t432, index::_b, index::_b, index::_a);

}


BOOST_AUTO_TEST_SUITE_END()
