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



#include <boost/numeric/ublas/tensor.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include "utility.hpp"

BOOST_AUTO_TEST_SUITE(test_tensor_static_comparison)

using double_extended = boost::multiprecision::cpp_bin_float_double_extended;

using test_types = zip<int,float,double_extended>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;

struct fixture {

    template<size_t... N>
    using extents_type = boost::numeric::ublas::extents<N...>;

    fixture()= default;

    std::tuple<
        extents_type<1,1>,   // 1
        extents_type<2,3>,   // 2
        extents_type<4,1,3>,  // 3
        extents_type<4,2,3>,  // 4
        extents_type<4,2,3,5>   // 5
    > extents;
};


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_comparison, value,  test_types, fixture)
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;

    auto check = [](auto const& /*unused*/, auto& e)
    { 
        using extents_type = std::decay_t<decltype(e)>;
        using tensor_type = ublas::tensor_static<value_type, extents_type, layout_type>;
        auto t  = tensor_type ();
        auto t2 = tensor_type ();
        auto v  = value_type  {};

        std::iota(t.begin(), t.end(), v);
        std::iota(t2.begin(), t2.end(), v+2);

        BOOST_CHECK( t == t  );
        BOOST_CHECK( t != t2 );

        if(t.empty())
            return;

        BOOST_CHECK(!(t < t));
        BOOST_CHECK(!(t > t));
        BOOST_CHECK( t < t2 );
        BOOST_CHECK( t2 > t );
        BOOST_CHECK( t <= t );
        BOOST_CHECK( t >= t );
        BOOST_CHECK( t <= t2 );
        BOOST_CHECK( t2 >= t );
        BOOST_CHECK( t2 >= t2 );
        BOOST_CHECK( t2 >= t );
    };

    for_each_in_tuple(extents,check);

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_comparison_with_tensor_expressions, value,  test_types, fixture)
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;


    auto check = [](auto const& /*unused*/, auto& e)
    { 
        using extents_type = std::decay_t<decltype(e)>;
        using tensor_type = ublas::tensor_static<value_type, extents_type, layout_type>;

        auto t  = tensor_type ();
        auto t2 = tensor_type ();
        auto v  = value_type  {};

        std::iota(t.begin(), t.end(), v);
        std::iota(t2.begin(), t2.end(), v+2);

        BOOST_CHECK( t == t  );
        BOOST_CHECK( t != t2 );

        if(t.empty())
            return;

        BOOST_CHECK( !(t < t) );
        BOOST_CHECK( !(t > t) );
        BOOST_CHECK( t < (t2+t) );
        BOOST_CHECK( (t2+t) > t );
        BOOST_CHECK( t <= (t+t) );
        BOOST_CHECK( (t+t2) >= t );
        BOOST_CHECK( (t2+t2+2) >= t);
        BOOST_CHECK( 2*t2 > t );
        BOOST_CHECK( t < 2*t2 );
        BOOST_CHECK( 2*t2 > t);
        BOOST_CHECK( 2*t2 >= t2 );
        BOOST_CHECK( t2 <= 2*t2);
        BOOST_CHECK( 3*t2 >= t );

    };

    for_each_in_tuple(extents,check);

}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_comparison_with_scalar, value,  test_types, fixture)
{
    namespace ublas = boost::numeric::ublas;
    using value_type  = typename value::first_type;
    using layout_type = typename value::second_type;


    for_each_in_tuple(extents,[](auto const& /*unused*/, auto& e)
    { 
        using extents_type = std::decay_t<decltype(e)>;
        using tensor_type = ublas::tensor_static<value_type, extents_type, layout_type>;

        BOOST_CHECK( tensor_type(value_type{2}) == tensor_type(value_type{2})  );
        BOOST_CHECK( tensor_type(value_type{2}) != tensor_type(value_type{1})  );

        if(ublas::empty(e))
            return;

        BOOST_CHECK( !(tensor_type(2) <  2) );
        BOOST_CHECK( !(tensor_type(2) >  2) );
        BOOST_CHECK(  (tensor_type(2) >= 2) );
        BOOST_CHECK(  (tensor_type(2) <= 2) );
        BOOST_CHECK(  (tensor_type(2) == 2) );
        BOOST_CHECK(  (tensor_type(2) != 3) );

        BOOST_CHECK( !(2 >  tensor_type(2)) );
        BOOST_CHECK( !(2 <  tensor_type(2)) );
        BOOST_CHECK(  (2 <= tensor_type(2)) );
        BOOST_CHECK(  (2 >= tensor_type(2)) );
        BOOST_CHECK(  (2 == tensor_type(2)) );
        BOOST_CHECK(  (3 != tensor_type(2)) );

        BOOST_CHECK( !( tensor_type(2)+3 <  5) );
        BOOST_CHECK( !( tensor_type(2)+3 >  5) );
        BOOST_CHECK(  ( tensor_type(2)+3 >= 5) );
        BOOST_CHECK(  ( tensor_type(2)+3 <= 5) );
        BOOST_CHECK(  ( tensor_type(2)+3 == 5) );
        BOOST_CHECK(  ( tensor_type(2)+3 != 6) );


        BOOST_CHECK( !( 5 >  tensor_type(2)+3) );
        BOOST_CHECK( !( 5 <  tensor_type(2)+3) );
        BOOST_CHECK(  ( 5 >= tensor_type(2)+3) );
        BOOST_CHECK(  ( 5 <= tensor_type(2)+3) );
        BOOST_CHECK(  ( 5 == tensor_type(2)+3) );
        BOOST_CHECK(  ( 6 != tensor_type(2)+3) );


        BOOST_CHECK( !( tensor_type(2)+tensor_type(3) <  5) );
        BOOST_CHECK( !( tensor_type(2)+tensor_type(3) >  5) );
        BOOST_CHECK(  ( tensor_type(2)+tensor_type(3) >= 5) );
        BOOST_CHECK(  ( tensor_type(2)+tensor_type(3) <= 5) );
        BOOST_CHECK(  ( tensor_type(2)+tensor_type(3) == 5) );
        BOOST_CHECK(  ( tensor_type(2)+tensor_type(3) != 6) );


        BOOST_CHECK( !( 5 >  tensor_type(2)+tensor_type(3)) );
        BOOST_CHECK( !( 5 <  tensor_type(2)+tensor_type(3)) );
        BOOST_CHECK(  ( 5 >= tensor_type(2)+tensor_type(3)) );
        BOOST_CHECK(  ( 5 <= tensor_type(2)+tensor_type(3)) );
        BOOST_CHECK(  ( 5 == tensor_type(2)+tensor_type(3)) );
        BOOST_CHECK(  ( 6 != tensor_type(2)+tensor_type(3)) );

    });

}


BOOST_AUTO_TEST_SUITE_END()
