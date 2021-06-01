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


BOOST_AUTO_TEST_SUITE(test_multi_index_utility, 
    *boost::unit_test::description("Validate Multi-Index Utility")
)


BOOST_AUTO_TEST_CASE(test_multi_index_has_index, 
    *boost::unit_test::label("has_index")
    *boost::unit_test::description("Testing has_index trait"))
{
    namespace ublas = boost::numeric::ublas;
    namespace index = ublas::index;
    
    BOOST_TEST_CONTEXT("[Has Index] testing empty tuple"){
        constexpr auto tuple = std::tuple<>{};
        constexpr auto has_a = ublas::has_index<decltype(index::_a),decltype(tuple)>::value;
        constexpr auto has_b = ublas::has_index<decltype(index::_b),decltype(tuple)>::value;
        BOOST_CHECK( !has_a );
        BOOST_CHECK( !has_b );
    }
    
    BOOST_TEST_CONTEXT("[Has Index] testing tuple(_a)"){
        constexpr auto tuple = std::make_tuple(index::_a);
        constexpr auto has_a = ublas::has_index<decltype(index::_a),decltype(tuple)>::value;
        constexpr auto has_b = ublas::has_index<decltype(index::_b),decltype(tuple)>::value;
        BOOST_CHECK(  has_a );
        BOOST_CHECK( !has_b );
    }
    
    BOOST_TEST_CONTEXT("[Has Index] testing tuple(_a, _b, _, _c, _d)"){
        constexpr auto tuple = std::make_tuple(index::_a,index::_b,index::_,index::_c,index::_d);
        constexpr auto has_a = ublas::has_index<decltype(index::_a),decltype(tuple)>::value;
        constexpr auto has_b = ublas::has_index<decltype(index::_b),decltype(tuple)>::value;
        constexpr auto has_c = ublas::has_index<decltype(index::_c),decltype(tuple)>::value;
        constexpr auto has_d = ublas::has_index<decltype(index::_d),decltype(tuple)>::value;
        constexpr auto has_e = ublas::has_index<decltype(index::_e),decltype(tuple)>::value;
        constexpr auto has   = ublas::has_index<decltype(index::_ ),decltype(tuple)>::value;
        BOOST_CHECK(  has_a );
        BOOST_CHECK(  has_b );
        BOOST_CHECK(  has_c );
        BOOST_CHECK(  has_d );
        BOOST_CHECK( !has_e );
        BOOST_CHECK(  has   );
    }

}

BOOST_AUTO_TEST_CASE(test_multi_index_valid, 
    *boost::unit_test::label("is_valid")
    *boost::unit_test::description("Testing if index is valid"))
{
    namespace ublas = boost::numeric::ublas;
    namespace index = ublas::index;
    
    BOOST_TEST_CONTEXT("[Is Valid] testing empty tuple"){
        constexpr auto tuple = std::tuple<>{};
        constexpr auto valid = ublas::valid_multi_index<decltype(tuple)>::value;
        BOOST_CHECK( valid );
    }
    
    BOOST_TEST_CONTEXT("[Is Valid] testing tuple(_a)"){
        constexpr auto tuple = std::make_tuple(index::_a);
        constexpr auto valid = ublas::valid_multi_index<decltype(tuple)>::value;
        BOOST_CHECK( valid );
    }

    BOOST_TEST_CONTEXT("[Is Valid] testing tuple(_a, _, _b)"){
        constexpr auto tuple = std::make_tuple(index::_a,index::_,index::_b);
        constexpr auto valid = ublas::valid_multi_index<decltype(tuple)>::value;
        BOOST_CHECK( valid );
    }

    BOOST_TEST_CONTEXT("[Is Valid] testing tuple(_a, _, _b, _b)"){
        constexpr auto tuple = std::make_tuple(index::_a,index::_,index::_b,index::_b);
        constexpr auto valid = ublas::valid_multi_index<decltype(tuple)>::value;
        BOOST_CHECK( !valid );
    }

    BOOST_TEST_CONTEXT("[Is Valid] testing tuple(_c, _a, _, _b, _b)"){
        constexpr auto tuple = std::make_tuple(index::_c,index::_a,index::_,index::_b,index::_b);
        constexpr auto valid = ublas::valid_multi_index<decltype(tuple)>::value;
        BOOST_CHECK( !valid );
    }

    BOOST_TEST_CONTEXT("[Is Valid] testing tuple(_c, _a, _, _b)"){
        constexpr auto tuple = std::make_tuple(index::_c,index::_a,index::_,index::_b);
        constexpr auto valid = ublas::valid_multi_index<decltype(tuple)>::value;
        BOOST_CHECK( valid );
    }

    BOOST_TEST_CONTEXT("[Is Valid] testing tuple(_, _c, _a, _, _b)"){
        constexpr auto tuple = std::make_tuple(index::_,index::_c,index::_a,index::_,index::_b);
        constexpr auto valid = ublas::valid_multi_index<decltype(tuple)>::value;
        BOOST_CHECK( valid );
    }

    BOOST_TEST_CONTEXT("[Is Valid] testing tuple(_, _c, _a, _, _b, _)"){
        constexpr auto tuple = std::make_tuple(index::_,index::_c,index::_a,index::_,index::_b,index::_);
        constexpr auto valid = ublas::valid_multi_index<decltype(tuple)>::value;
        BOOST_CHECK( valid );
    }

}

BOOST_AUTO_TEST_CASE(test_multi_index_number_equal_indices, 
    *boost::unit_test::label("intersection_indices")
    *boost::unit_test::description("Testing finds the intersection of indices"))
{
    namespace ublas = boost::numeric::ublas;
    namespace index = ublas::index;
    
    BOOST_TEST_CONTEXT("[Intersection of Indices] testing empty tuple"){
        constexpr auto lhs = std::tuple<>{};
        constexpr auto rhs = std::tuple<>{};
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 0ul );
    }
    
    BOOST_TEST_CONTEXT("[Intersection of Indices] testing [LHS]tuple(_a) : [RHS]tuple()"){
        constexpr auto lhs = std::make_tuple(index::_a);
        constexpr auto rhs = std::tuple<>{};
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 0ul );
    }

    BOOST_TEST_CONTEXT("[Intersection of Indices] testing [LHS]tuple(_a) : [RHS]tuple(_a)"){
        constexpr auto lhs = std::make_tuple(index::_a);
        constexpr auto rhs = std::make_tuple(index::_a);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 1ul );
    }

    BOOST_TEST_CONTEXT("[Intersection of Indices] testing [LHS]tuple(_a, _b) : [RHS]tuple(_a)"){
        constexpr auto lhs = std::make_tuple(index::_a,index::_b);
        constexpr auto rhs = std::make_tuple(index::_a);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 1ul );
    }

    BOOST_TEST_CONTEXT("[Intersection of Indices] testing [LHS]tuple(_a) : [RHS]tuple(_a, _b)"){
        constexpr auto lhs = std::make_tuple(index::_b);
        constexpr auto rhs = std::make_tuple(index::_a,index::_b);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 1ul );
    }

    BOOST_TEST_CONTEXT("[Intersection of Indices] testing [LHS]tuple(_a, _b) : [RHS]tuple(_a, _b)"){
        constexpr auto lhs = std::make_tuple(index::_a,index::_b);
        constexpr auto rhs = std::make_tuple(index::_a,index::_b);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 2ul );
    }

    BOOST_TEST_CONTEXT("[Intersection of Indices] testing [LHS]tuple(_b, _a) : [RHS]tuple(_a, _b)"){
        constexpr auto lhs = std::make_tuple(index::_b,index::_a);
        constexpr auto rhs = std::make_tuple(index::_a,index::_b);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 2ul );
    }

    BOOST_TEST_CONTEXT("[Intersection of Indices] testing [LHS]tuple(_b, _a, _c) : [RHS]tuple(_a, _b)"){
        constexpr auto lhs = std::make_tuple(index::_b,index::_a,index::_c);
        constexpr auto rhs = std::make_tuple(index::_a,index::_b);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 2ul );
    }

    BOOST_TEST_CONTEXT("[Intersection of Indices] testing [LHS]tuple(_b, _a, _c) : [RHS]tuple(_a, _b, _d)"){
        constexpr auto lhs = std::make_tuple(index::_b,index::_a,index::_c);
        constexpr auto rhs = std::make_tuple(index::_a,index::_b,index::_d);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 2ul );
    }

    BOOST_TEST_CONTEXT("[Intersection of Indices] testing [LHS]tuple(_b, _a, _d) : [RHS]tuple(_a, _b, _d)"){
        constexpr auto lhs = std::make_tuple(index::_b,index::_a,index::_d);
        constexpr auto rhs = std::make_tuple(index::_a,index::_b,index::_d);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 3ul );
    }

    BOOST_TEST_CONTEXT("[Intersection of Indices] testing [LHS]tuple(_b, _a, _d) : [RHS]tuple(_a, _b, _d, _)"){
        constexpr auto lhs = std::make_tuple(index::_b,index::_a,index::_d);
        constexpr auto rhs = std::make_tuple(index::_a,index::_b,index::_d,index::_);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 3ul );
    }

    BOOST_TEST_CONTEXT("[Intersection of Indices] testing [LHS]tuple(_b, _a, _d, _) : [RHS]tuple(_a, _b, _d, _)"){
        constexpr auto lhs = std::make_tuple(index::_b,index::_a,index::_d,index::_);
        constexpr auto rhs = std::make_tuple(index::_a,index::_b,index::_d,index::_);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 3ul );
    }

    BOOST_TEST_CONTEXT("[Intersection of Indices] testing [LHS]tuple(_b, _a, _d, _) : [RHS]tuple(_, _b, _d, _)"){
        constexpr auto lhs = std::make_tuple(index::_b,index::_a,index::_d,index::_);
        constexpr auto rhs = std::make_tuple(index::_,index::_b,index::_d,index::_);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 2ul );
    }

    BOOST_TEST_CONTEXT("[Intersection of Indices] testing [LHS]tuple(_, _a, _d, _) : [RHS]tuple(_, _b, _d, _)"){
        constexpr auto lhs = std::make_tuple(index::_,index::_a,index::_d,index::_);
        constexpr auto rhs = std::make_tuple(index::_,index::_b,index::_d,index::_);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 1ul );
    }

    BOOST_TEST_CONTEXT("[Intersection of Indices] testing [LHS]tuple(_, _a, _d, _) : [RHS]tuple(_, _b, _d, _, _)"){
        constexpr auto lhs = std::make_tuple(index::_,index::_a,index::_d,index::_);
        constexpr auto rhs = std::make_tuple(index::_,index::_b,index::_d,index::_,index::_);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 1ul );
    }

}

BOOST_AUTO_TEST_CASE(test_multi_index_index_position, 
    *boost::unit_test::label("get_index_pos")
    *boost::unit_test::description("Testing index position"))
{
    namespace ublas = boost::numeric::ublas;
    namespace index = ublas::index;
    
    BOOST_TEST_CONTEXT("[Get Index Position] testing empty tuple"){
        constexpr auto tuple = std::tuple<>{};
        constexpr auto ind   = ublas::index_position<decltype(index::_),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,0ul);
    }
    
    BOOST_TEST_CONTEXT("[Get Index Position] testing get position of (_) in tuple(_)"){
        constexpr auto tuple = std::make_tuple(index::_);
        constexpr auto ind   = ublas::index_position<decltype(index::_),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,0ul);
    }
    
    BOOST_TEST_CONTEXT("[Get Index Position] testing get position of (_a) in tuple(_)"){
        constexpr auto tuple = std::make_tuple(index::_);
        constexpr auto ind   = ublas::index_position<decltype(index::_a),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,1ul);
    }
    
    BOOST_TEST_CONTEXT("[Get Index Position] testing get position of (_) in tuple(_, _a)"){
        constexpr auto tuple = std::make_tuple(index::_,index::_a);
        constexpr auto ind   = ublas::index_position<decltype(index::_),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,0ul);
    }
    
    BOOST_TEST_CONTEXT("[Get Index Position] testing get position of (_a) in tuple(_, _a)"){
        constexpr auto tuple = std::make_tuple(index::_,index::_a);
        constexpr auto ind   = ublas::index_position<decltype(index::_a),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,1ul);
    }

    BOOST_TEST_CONTEXT("[Get Index Position] testing get position of (_b) in tuple(_, _a)"){
        constexpr auto tuple = std::make_tuple(index::_,index::_a);
        constexpr auto ind   = ublas::index_position<decltype(index::_b),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,2ul);
    }

    BOOST_TEST_CONTEXT("[Get Index Position] testing get position of (_c) in tuple(_c, _, _a)"){
        constexpr auto tuple = std::make_tuple(index::_c,index::_,index::_a);
        constexpr auto ind   = ublas::index_position<decltype(index::_c),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,0ul);
    }

    BOOST_TEST_CONTEXT("[Get Index Position] testing get position of (_) in tuple(_c, _, _a, _)"){
        constexpr auto tuple = std::make_tuple(index::_c,index::_,index::_a,index::_);
        constexpr auto ind   = ublas::index_position<decltype(index::_),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,1);
    }

    BOOST_TEST_CONTEXT("[Get Index Position] testing get position of (_a) in tuple(_c, _, _a)"){
        constexpr auto tuple = std::make_tuple(index::_c,index::_,index::_a);
        constexpr auto ind   = ublas::index_position<decltype(index::_a),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,2);
    }

    BOOST_TEST_CONTEXT("[Get Index Position] testing get position of (_d) in tuple(_c, _, _a)"){
        constexpr auto tuple = std::make_tuple(index::_c,index::_,index::_a);
        constexpr auto ind   = ublas::index_position<decltype(index::_d),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,3);
    }

}

BOOST_AUTO_TEST_CASE(test_multi_index_index_position_pairs, 
    *boost::unit_test::label("pos_pairs")
    *boost::unit_test::description("Testing pair positions"))
{
    namespace ublas = boost::numeric::ublas;
    namespace index = ublas::index;
    
    BOOST_TEST_CONTEXT("[Get Pair Position] testing empty tuple"){
        constexpr auto lhs   = std::tuple<>{};
        constexpr auto rhs   = std::tuple<>{};
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_CHECK_EQUAL(array.size(), 0ul );
    }

    BOOST_TEST_CONTEXT("[Get Pair Position] testing [LHS]tuple(_a) : [RHS]tuple()"){
        constexpr auto lhs = std::make_tuple(index::_a);
        constexpr auto rhs = std::tuple<>{};
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_CHECK_EQUAL(array.size(), 0ul );
    }

    BOOST_TEST_CONTEXT("[Get Pair Position] testing [LHS]tuple() : [RHS]tuple(_a)"){
        constexpr auto lhs = std::tuple<>{};
        constexpr auto rhs = std::make_tuple(index::_a);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_CHECK_EQUAL(array.size(), 0ul );
    }

    BOOST_TEST_CONTEXT("[Get Pair Position] testing [LHS]tuple(_b) : [RHS]tuple(_a)"){
        constexpr auto lhs = std::make_tuple(index::_b);
        constexpr auto rhs = std::make_tuple(index::_a);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_CHECK_EQUAL(array.size(), 0ul );
    }



    BOOST_TEST_CONTEXT("[Get Pair Position] testing [LHS]tuple(_a) : [RHS]tuple(_a)"){
        constexpr auto lhs = std::make_tuple(index::_a);
        constexpr auto rhs = std::make_tuple(index::_a);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 1ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 0 );
    }

    BOOST_TEST_CONTEXT("[Get Pair Position] testing [LHS]tuple(_a, _b) : [RHS]tuple(_a)"){
        constexpr auto lhs = std::make_tuple(index::_a,index::_b);
        constexpr auto rhs = std::make_tuple(index::_a);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 1ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 0 );
    }

    BOOST_TEST_CONTEXT("[Get Pair Position] testing [LHS]tuple(_b) : [RHS]tuple(_a, _b)"){
        constexpr auto lhs = std::make_tuple(index::_b);
        constexpr auto rhs = std::make_tuple(index::_a,index::_b);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 1ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 1 );
    }

    BOOST_TEST_CONTEXT("[Get Pair Position] testing [LHS]tuple(_a) : [RHS]tuple(_a)"){
        constexpr auto lhs = std::make_tuple(index::_a);
        constexpr auto rhs = std::make_tuple(index::_a);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 1ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 0 );
    }



    BOOST_TEST_CONTEXT("[Get Pair Position] testing [LHS]tuple(_a, _b) : [RHS]tuple(_a, _b)"){
        constexpr auto lhs = std::make_tuple(index::_a,index::_b);
        constexpr auto rhs = std::make_tuple(index::_a,index::_b);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 2ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 0 );
        BOOST_CHECK_EQUAL(array[1].first , 1 );
        BOOST_CHECK_EQUAL(array[1].second, 1 );
    }

    BOOST_TEST_CONTEXT("[Get Pair Position] testing [LHS]tuple(_b, _a) : [RHS]tuple(_a, _b)"){
        constexpr auto lhs = std::make_tuple(index::_b,index::_a);
        constexpr auto rhs = std::make_tuple(index::_a,index::_b);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 2ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 1 );
        BOOST_CHECK_EQUAL(array[1].first , 1 );
        BOOST_CHECK_EQUAL(array[1].second, 0 );
    }

    BOOST_TEST_CONTEXT("[Get Pair Position] testing [LHS]tuple(_b, _a, _c) : [RHS]tuple(_a, _b)"){
        constexpr auto lhs = std::make_tuple(index::_b,index::_a,index::_c);
        constexpr auto rhs = std::make_tuple(index::_a,index::_b);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 2ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 1 );
        BOOST_CHECK_EQUAL(array[1].first , 1 );
        BOOST_CHECK_EQUAL(array[1].second, 0 );
    }

    BOOST_TEST_CONTEXT("[Get Pair Position] testing [LHS]tuple(_b, _a, _c) : [RHS]tuple(_a, _b, _d)"){
        constexpr auto lhs = std::make_tuple(index::_b,index::_a,index::_c);
        constexpr auto rhs = std::make_tuple(index::_a,index::_b,index::_d);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 2ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 1 );
        BOOST_CHECK_EQUAL(array[1].first , 1 );
        BOOST_CHECK_EQUAL(array[1].second, 0 );
    }

    BOOST_TEST_CONTEXT("[Get Pair Position] testing [LHS]tuple(_b, _a, _d) : [RHS]tuple(_a, _b, _d)"){
        constexpr auto lhs = std::make_tuple(index::_b,index::_a,index::_d);
        constexpr auto rhs = std::make_tuple(index::_a,index::_b,index::_d);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 3ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 1 );
        BOOST_CHECK_EQUAL(array[1].first , 1 );
        BOOST_CHECK_EQUAL(array[1].second, 0 );
        BOOST_CHECK_EQUAL(array[2].first , 2 );
        BOOST_CHECK_EQUAL(array[2].second, 2 );
    }

    BOOST_TEST_CONTEXT("[Get Pair Position] testing [LHS]tuple(_b, _a, _d) : [RHS]tuple(_a, _b, _d, _)"){
        constexpr auto lhs = std::make_tuple(index::_b,index::_a,index::_d);
        constexpr auto rhs = std::make_tuple(index::_a,index::_b,index::_d,index::_);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 3ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 1 );
        BOOST_CHECK_EQUAL(array[1].first , 1 );
        BOOST_CHECK_EQUAL(array[1].second, 0 );
        BOOST_CHECK_EQUAL(array[2].first , 2 );
        BOOST_CHECK_EQUAL(array[2].second, 2 );
    }

    BOOST_TEST_CONTEXT("[Get Pair Position] testing [LHS]tuple(_b, _a, _d, _) : [RHS]tuple(_a, _b, _d, _)"){
        constexpr auto lhs = std::make_tuple(index::_b,index::_a,index::_d,index::_);
        constexpr auto rhs = std::make_tuple(index::_a,index::_b,index::_d,index::_);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 3ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 1 );
        BOOST_CHECK_EQUAL(array[1].first , 1 );
        BOOST_CHECK_EQUAL(array[1].second, 0 );
        BOOST_CHECK_EQUAL(array[2].first , 2 );
        BOOST_CHECK_EQUAL(array[2].second, 2 );
    }

    BOOST_TEST_CONTEXT("[Get Pair Position] testing [LHS]tuple(_b, _a, _d, _) : [RHS]tuple(_a, _b, _d, _)"){
        constexpr auto lhs = std::make_tuple(index::_b,index::_a,index::_d,index::_);
        constexpr auto rhs = std::make_tuple(index::_,index::_b,index::_d,index::_);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 2ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 1 );
        BOOST_CHECK_EQUAL(array[1].first , 2 );
        BOOST_CHECK_EQUAL(array[1].second, 2 );
    }

    BOOST_TEST_CONTEXT("[Get Pair Position] testing [LHS]tuple(_b, _a, _d, _) : [RHS]tuple(_a, _b, _d, _)"){
        constexpr auto lhs = std::make_tuple(index::_,index::_a,index::_d,index::_);
        constexpr auto rhs = std::make_tuple(index::_,index::_b,index::_d,index::_);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 1ul );
        BOOST_CHECK_EQUAL(array[0].first , 2 );
        BOOST_CHECK_EQUAL(array[0].second, 2 );
    }

    BOOST_TEST_CONTEXT("[Get Pair Position] testing [LHS]tuple(_b, _a, _d, _) : [RHS]tuple(_a, _b, _d, _, _)"){
        constexpr auto lhs = std::make_tuple(index::_,index::_a,index::_d,index::_);
        constexpr auto rhs = std::make_tuple(index::_,index::_b,index::_d,index::_,index::_);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 1ul );
        BOOST_CHECK_EQUAL(array[0].first , 2 );
        BOOST_CHECK_EQUAL(array[0].second, 2 );
    }

}

BOOST_AUTO_TEST_SUITE_END()
