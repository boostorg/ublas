//  Copyright (c) 2018-2019 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The author gratefully acknowledge the support of
//  Fraunhofer and Google in producing this work
//  which started as a Google Summer of Code project.
//


#include <boost/numeric/ublas/tensor.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE ( test_multi_index_utility )


BOOST_AUTO_TEST_CASE ( test_multi_index_has_index )
{
    namespace ublas = boost::numeric::ublas;
    namespace i = boost::numeric::ublas::index;

    {
        constexpr auto tuple = std::tuple<>{};
        constexpr auto has_a = ublas::has_index<decltype(i::_a),decltype(tuple)>::value;
        constexpr auto has_b = ublas::has_index<decltype(i::_b),decltype(tuple)>::value;
        BOOST_CHECK( !has_a );
        BOOST_CHECK( !has_b );
    }


    {
        constexpr auto tuple = std::make_tuple(i::_a);
        constexpr auto has_a = ublas::has_index<decltype(i::_a),decltype(tuple)>::value;
        constexpr auto has_b = ublas::has_index<decltype(i::_b),decltype(tuple)>::value;
        BOOST_CHECK(  has_a );
        BOOST_CHECK( !has_b );
    }

    {
      constexpr auto tuple = std::make_tuple(i::_a,i::_b,i::_,i::_c,i::_d);
        constexpr auto has_a = ublas::has_index<decltype(i::_a),decltype(tuple)>::value;
        constexpr auto has_b = ublas::has_index<decltype(i::_b),decltype(tuple)>::value;
        constexpr auto has_c = ublas::has_index<decltype(i::_c),decltype(tuple)>::value;
        constexpr auto has_d = ublas::has_index<decltype(i::_d),decltype(tuple)>::value;
        constexpr auto has_e = ublas::has_index<decltype(i::_e),decltype(tuple)>::value;
        constexpr auto has   = ublas::has_index<decltype(i::_ ),decltype(tuple)>::value;
        BOOST_CHECK(  has_a );
        BOOST_CHECK(  has_b );
        BOOST_CHECK(  has_c );
        BOOST_CHECK(  has_d );
        BOOST_CHECK( !has_e );
        BOOST_CHECK(  has   );
    }
}



BOOST_AUTO_TEST_CASE ( test_multi_index_valid )
{
    namespace ublas = boost::numeric::ublas;
    namespace i = ublas::index;

    {
        constexpr auto tuple = std::tuple<>{};
        constexpr auto valid = ublas::valid_multi_index<decltype(tuple)>::value;
        BOOST_CHECK( valid );
    }


    {
        constexpr auto tuple = std::make_tuple(i::_a);
        constexpr auto valid = ublas::valid_multi_index<decltype(tuple)>::value;
        BOOST_CHECK( valid );
    }

    {
        constexpr auto tuple = std::make_tuple(i::_a,i::_,i::_b);
        constexpr auto valid = ublas::valid_multi_index<decltype(tuple)>::value;
        BOOST_CHECK( valid );
    }

    {
        constexpr auto tuple = std::make_tuple(i::_a,i::_,i::_b,i::_b);
        constexpr auto valid = ublas::valid_multi_index<decltype(tuple)>::value;
        BOOST_CHECK( !valid );
    }

    {
        constexpr auto tuple = std::make_tuple(i::_c,i::_a,i::_,i::_b,i::_b);
        constexpr auto valid = ublas::valid_multi_index<decltype(tuple)>::value;
        BOOST_CHECK( !valid );
    }

    {
        constexpr auto tuple = std::make_tuple(i::_c,i::_a,i::_,i::_b);
        constexpr auto valid = ublas::valid_multi_index<decltype(tuple)>::value;
        BOOST_CHECK( valid );
    }

    {
        constexpr auto tuple = std::make_tuple(i::_,i::_c,i::_a,i::_,i::_b);
        constexpr auto valid = ublas::valid_multi_index<decltype(tuple)>::value;
        BOOST_CHECK( valid );
    }

    {
        constexpr auto tuple = std::make_tuple(i::_,i::_c,i::_a,i::_,i::_b,i::_);
        constexpr auto valid = ublas::valid_multi_index<decltype(tuple)>::value;
        BOOST_CHECK( valid );
    }
}





BOOST_AUTO_TEST_CASE ( test_multi_index_number_equal_indices )
{
    namespace ublas = boost::numeric::ublas;
    namespace i = ublas::index;

    {
        constexpr auto lhs = std::tuple<>{};
        constexpr auto rhs = std::tuple<>{};
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 0 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_a);
        constexpr auto rhs = std::tuple<>{};
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 0 );
    }

    {
        constexpr auto lhs = std::tuple<>{};
        constexpr auto rhs = std::make_tuple(i::_a);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 0 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_b);
        constexpr auto rhs = std::make_tuple(i::_a);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 0 );
    }



    {
        constexpr auto lhs = std::make_tuple(i::_a);
        constexpr auto rhs = std::make_tuple(i::_a);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 1 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_a,i::_b);
        constexpr auto rhs = std::make_tuple(i::_a);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 1 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_b);
        constexpr auto rhs = std::make_tuple(i::_a,i::_b);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 1 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_a);
        constexpr auto rhs = std::make_tuple(i::_a);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 1 );
    }



    {
        constexpr auto lhs = std::make_tuple(i::_a,i::_b);
        constexpr auto rhs = std::make_tuple(i::_a,i::_b);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 2 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_b,i::_a);
        constexpr auto rhs = std::make_tuple(i::_a,i::_b);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 2 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_b,i::_a,i::_c);
        constexpr auto rhs = std::make_tuple(i::_a,i::_b);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 2 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_b,i::_a,i::_c);
        constexpr auto rhs = std::make_tuple(i::_a,i::_b,i::_d);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 2 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_b,i::_a,i::_d);
        constexpr auto rhs = std::make_tuple(i::_a,i::_b,i::_d);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 3 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_b,i::_a,i::_d);
        constexpr auto rhs = std::make_tuple(i::_a,i::_b,i::_d,i::_);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 3 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_b,i::_a,i::_d,i::_);
        constexpr auto rhs = std::make_tuple(i::_a,i::_b,i::_d,i::_);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 3 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_b,i::_a,i::_d,i::_);
        constexpr auto rhs = std::make_tuple(i::_,i::_b,i::_d,i::_);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 2 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_,i::_a,i::_d,i::_);
        constexpr auto rhs = std::make_tuple(i::_,i::_b,i::_d,i::_);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 1 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_,i::_a,i::_d,i::_);
        constexpr auto rhs = std::make_tuple(i::_,i::_b,i::_d,i::_,i::_);
        constexpr auto num = ublas::number_equal_indexes<decltype(lhs), decltype(rhs)>::value;
        BOOST_CHECK_EQUAL( num, 1 );
    }
}







BOOST_AUTO_TEST_CASE ( test_multi_index_index_position )
{
    namespace ublas = boost::numeric::ublas;
    namespace i = ublas::index;

    {
        constexpr auto tuple = std::tuple<>{};
        constexpr auto ind   = ublas::index_position<decltype(i::_),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,0);
    }

    {
      constexpr auto tuple = std::make_tuple(i::_);
        constexpr auto ind   = ublas::index_position<decltype(i::_),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,0);
    }

    {
        constexpr auto tuple = std::make_tuple(i::_);
        constexpr auto ind   = ublas::index_position<decltype(i::_a),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,1);
    }

    {
        constexpr auto tuple = std::make_tuple(i::_,i::_a);
        constexpr auto ind   = ublas::index_position<decltype(i::_),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,0);
    }

    {
        constexpr auto tuple = std::make_tuple(i::_,i::_a);
        constexpr auto ind   = ublas::index_position<decltype(i::_a),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,1);
    }

    {
        constexpr auto tuple = std::make_tuple(i::_,i::_a);
        constexpr auto ind   = ublas::index_position<decltype(i::_b),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,2);
    }




    {
        constexpr auto tuple = std::make_tuple(i::_c,i::_,i::_a);
        constexpr auto ind   = ublas::index_position<decltype(i::_c),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,0);
    }

    {
        constexpr auto tuple = std::make_tuple(i::_c,i::_,i::_a,i::_);
        constexpr auto ind   = ublas::index_position<decltype(i::_),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,1);
    }

    {
        constexpr auto tuple = std::make_tuple(i::_c,i::_,i::_a);
        constexpr auto ind   = ublas::index_position<decltype(i::_a),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,2);
    }

    {
        constexpr auto tuple = std::make_tuple(i::_c,i::_,i::_a);
        constexpr auto ind   = ublas::index_position<decltype(i::_d),decltype(tuple)>::value;
        BOOST_CHECK_EQUAL(ind,3);
    }

}








BOOST_AUTO_TEST_CASE ( test_multi_index_index_position_pairs )
{
    namespace ublas = boost::numeric::ublas;
    namespace i = ublas::index;

    {
        constexpr auto lhs   = std::tuple<>{};
        constexpr auto rhs   = std::tuple<>{};
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_CHECK_EQUAL(array.size(), 0ul );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_a);
        constexpr auto rhs = std::tuple<>{};
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_CHECK_EQUAL(array.size(), 0ul );
    }

    {
        constexpr auto lhs = std::tuple<>{};
        constexpr auto rhs = std::make_tuple(i::_a);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_CHECK_EQUAL(array.size(), 0ul );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_b);
        constexpr auto rhs = std::make_tuple(i::_a);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_CHECK_EQUAL(array.size(), 0ul );
    }



    {
        constexpr auto lhs = std::make_tuple(i::_a);
        constexpr auto rhs = std::make_tuple(i::_a);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 1ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 0 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_a,i::_b);
        constexpr auto rhs = std::make_tuple(i::_a);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 1ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 0 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_b);
        constexpr auto rhs = std::make_tuple(i::_a,i::_b);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 1ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 1 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_a);
        constexpr auto rhs = std::make_tuple(i::_a);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 1ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 0 );
    }



    {
        constexpr auto lhs = std::make_tuple(i::_a,i::_b);
        constexpr auto rhs = std::make_tuple(i::_a,i::_b);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 2ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 0 );
        BOOST_CHECK_EQUAL(array[1].first , 1 );
        BOOST_CHECK_EQUAL(array[1].second, 1 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_b,i::_a);
        constexpr auto rhs = std::make_tuple(i::_a,i::_b);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 2ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 1 );
        BOOST_CHECK_EQUAL(array[1].first , 1 );
        BOOST_CHECK_EQUAL(array[1].second, 0 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_b,i::_a,i::_c);
        constexpr auto rhs = std::make_tuple(i::_a,i::_b);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 2ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 1 );
        BOOST_CHECK_EQUAL(array[1].first , 1 );
        BOOST_CHECK_EQUAL(array[1].second, 0 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_b,i::_a,i::_c);
        constexpr auto rhs = std::make_tuple(i::_a,i::_b,i::_d);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 2ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 1 );
        BOOST_CHECK_EQUAL(array[1].first , 1 );
        BOOST_CHECK_EQUAL(array[1].second, 0 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_b,i::_a,i::_d);
        constexpr auto rhs = std::make_tuple(i::_a,i::_b,i::_d);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 3ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 1 );
        BOOST_CHECK_EQUAL(array[1].first , 1 );
        BOOST_CHECK_EQUAL(array[1].second, 0 );
        BOOST_CHECK_EQUAL(array[2].first , 2 );
        BOOST_CHECK_EQUAL(array[2].second, 2 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_b,i::_a,i::_d);
        constexpr auto rhs = std::make_tuple(i::_a,i::_b,i::_d,i::_);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 3ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 1 );
        BOOST_CHECK_EQUAL(array[1].first , 1 );
        BOOST_CHECK_EQUAL(array[1].second, 0 );
        BOOST_CHECK_EQUAL(array[2].first , 2 );
        BOOST_CHECK_EQUAL(array[2].second, 2 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_b,i::_a,i::_d,i::_);
        constexpr auto rhs = std::make_tuple(i::_a,i::_b,i::_d,i::_);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 3ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 1 );
        BOOST_CHECK_EQUAL(array[1].first , 1 );
        BOOST_CHECK_EQUAL(array[1].second, 0 );
        BOOST_CHECK_EQUAL(array[2].first , 2 );
        BOOST_CHECK_EQUAL(array[2].second, 2 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_b,i::_a,i::_d,i::_);
        constexpr auto rhs = std::make_tuple(i::_,i::_b,i::_d,i::_);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 2ul );
        BOOST_CHECK_EQUAL(array[0].first , 0 );
        BOOST_CHECK_EQUAL(array[0].second, 1 );
        BOOST_CHECK_EQUAL(array[1].first , 2 );
        BOOST_CHECK_EQUAL(array[1].second, 2 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_,i::_a,i::_d,i::_);
        constexpr auto rhs = std::make_tuple(i::_,i::_b,i::_d,i::_);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 1ul );
        BOOST_CHECK_EQUAL(array[0].first , 2 );
        BOOST_CHECK_EQUAL(array[0].second, 2 );
    }

    {
        constexpr auto lhs = std::make_tuple(i::_,i::_a,i::_d,i::_);
        constexpr auto rhs = std::make_tuple(i::_,i::_b,i::_d,i::_,i::_);
        auto array = ublas::index_position_pairs(lhs, rhs);
        BOOST_STATIC_ASSERT(array.size() == 1ul );
        BOOST_CHECK_EQUAL(array[0].first , 2 );
        BOOST_CHECK_EQUAL(array[0].second, 2 );
    }
}



//BOOST_AUTO_TEST_CASE ( test_multi_index_array_to_vector )
//{
//    namespace ublas = boost::numeric::ublas;
//    namespace i = ublas::index;

//    auto check = [](auto const& lhs, auto const& rhs)
//    {
//        auto array = ublas::index_position_pairs(lhs, rhs);

//        auto vector_pair =  ublas::array_to_vector( array );

//        BOOST_CHECK_EQUAL(vector_pair.first .size(), array.size() );
//        BOOST_CHECK_EQUAL(vector_pair.second.size(), array.size() );

//        for(auto i = 0ul; i < array.size(); ++i)
//        {
//            BOOST_CHECK_EQUAL(vector_pair.first [i], array[i].first +1 );
//            BOOST_CHECK_EQUAL(vector_pair.second[i], array[i].second+1 );
//        }

//    };

//    check(std::tuple<>{}        , std::tuple<>{});
//    check(std::make_tuple(i::_a)   , std::tuple<>{});
//    check(std::tuple<>{}        , std::make_tuple(i::_a));
//    check(std::make_tuple(i::_a)   , std::make_tuple(i::_b));
//    check(std::make_tuple(i::_a)   , std::make_tuple(i::_a));
//    check(std::make_tuple(i::_a,i::_b), std::make_tuple(i::_a));
//    check(std::make_tuple(i::_a)   , std::make_tuple(i::_a,i::_b));
//    check(std::make_tuple(i::_a,i::_b), std::make_tuple(i::_a,i::_b));
//    check(std::make_tuple(i::_b,i::_a), std::make_tuple(i::_a,i::_b));
//    check(std::make_tuple(i::_b,i::_a,i::_c), std::make_tuple(i::_a,i::_b,i::_d));
//}



BOOST_AUTO_TEST_SUITE_END()
