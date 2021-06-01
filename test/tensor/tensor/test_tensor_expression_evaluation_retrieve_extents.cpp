//
// 	Copyright (c) 2021  Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2021, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#include <boost/test/unit_test.hpp>
#include "../fixture_utility.hpp"

BOOST_AUTO_TEST_SUITE(test_tensor_expression_evaluation_retrieve_extents, 
    * boost::unit_test::description("Validate Expression Evaluation's Retrive Extents")
)

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("expr_eval_retrive_extents")
    *boost::unit_test::description("Testing the expression evaluation's retrive extents for dynamic tensor")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_dynamic,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_dynamic<TestTupleType>
)
{
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using fixture_t = ublas::tuple_fixture_tensor_dynamic<TestTupleType>;

    constexpr auto uplus1 = [](auto const& a){return a + value_type(1); };
    constexpr auto uplus2 = [](auto const& a){return value_type(2) + a; };
    constexpr auto bplus  = std::plus <value_type>{};
    constexpr auto bminus = std::minus<value_type>{};

    constexpr auto equal_range = [](auto const& l, auto const& r){
        BOOST_CHECK_EQUAL_COLLECTIONS(ublas::begin(l), ublas::end(l), ublas::begin(r), ublas::end(r));
    };

    constexpr auto check_same_tensor = [uplus1, uplus2, bplus, bminus, equal_range]<typename tensor_t>(auto /*id*/, tensor_t t){
        auto v = value_type{};

        BOOST_TEST_CONTEXT("[Retrive Extents(With Same Dynamic Tensor)] rank("<< t.rank() <<") dynamic tensor"){
            
            ublas::iota(t,v);
            auto const& e = t.extents();

            equal_range(ublas::detail::retrieve_extents( t ), e);
            
            // uexpr1 = t+1
            // uexpr2 = 2+t
            auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_t>( t, uplus1 );
            auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_t>( t, uplus2 );

            equal_range( ublas::detail::retrieve_extents( uexpr1 ), e );
            equal_range( ublas::detail::retrieve_extents( uexpr2 ), e );

            // bexpr_uexpr = (t+1) + (2+t)
            auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( uexpr1, uexpr2, bplus );

            equal_range( ublas::detail::retrieve_extents( bexpr_uexpr ), e );


            // bexpr_bexpr_uexpr = ((t+1) + (2+t)) - t
            auto bexpr_bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( bexpr_uexpr, t, bminus );

            equal_range( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr ), e );
        }
    };

    auto check_different_tensor = [uplus1, uplus2, bplus, bminus, equal_range, this]<typename tensor_t>(auto i, tensor_t t1){
        auto const size = this->collection.size();

        if(i >= size - 1ul)
            return;

        auto v = value_type{};
        auto t2 = this->collection[i + 1ul];

        BOOST_TEST_CONTEXT("[Retrive Extents(With Different Dynamic Tensor)] rank("<< t1.rank() << ", "<< t2.rank() <<") dynamic tensor"){
            
            ublas::iota(t1,v);
            ublas::iota(t2,v + value_type{1});

            BOOST_CHECK( ublas::detail::retrieve_extents( t1 ) != ublas::detail::retrieve_extents( t2 ) );

            // uexpr1 = t1+1
            // uexpr2 = 2+t2
            auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_t>( t1, uplus1 );
            auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_t>( t2, uplus2 );

            equal_range( ublas::detail::retrieve_extents( t1 ), ublas::detail::retrieve_extents( uexpr1 ) );
            equal_range( ublas::detail::retrieve_extents( t2 ), ublas::detail::retrieve_extents( uexpr2 ) );
            BOOST_CHECK( ublas::detail::retrieve_extents( uexpr1 ) != ublas::detail::retrieve_extents( uexpr2 ) );

            // bexpr_uexpr = (t1+1) + (2+t2)
            auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( uexpr1, uexpr2, bplus );

            equal_range( ublas::detail::retrieve_extents( bexpr_uexpr ), ublas::detail::retrieve_extents(t1) );


            // bexpr_bexpr_uexpr = ((t1+1) + (2+t2)) - t2
            auto bexpr_bexpr_uexpr1 = ublas::detail::make_binary_tensor_expression<tensor_t>( bexpr_uexpr, t2, bminus );

            equal_range( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr1 ), ublas::detail::retrieve_extents(t2) );


            // bexpr_bexpr_uexpr = t2 - ((t1+1) + (2+t2))
            auto bexpr_bexpr_uexpr2 = ublas::detail::make_binary_tensor_expression<tensor_t>( t2, bexpr_uexpr, bminus );

            equal_range( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr2 ), ublas::detail::retrieve_extents(t2) );
        }
    };


    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, [check_same_tensor, &check_different_tensor](auto id, auto const& t){
        std::invoke(check_same_tensor, id, t);
        std::invoke(check_different_tensor, id, t);
    });
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("expr_eval_retrive_extents")
    *boost::unit_test::description("Testing the expression evaluation's retrive extents for static rank tensor")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static_rank,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_static_rank<TestTupleType>
)
{
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using fixture_t = ublas::tuple_fixture_tensor_static_rank<TestTupleType>;
    static constexpr auto fixture_size = fixture_t::size;


    constexpr auto uplus1 = [](auto const& a){return a + value_type(1); };
    constexpr auto uplus2 = [](auto const& a){return value_type(2) + a; };
    constexpr auto bplus  = std::plus <value_type>{};
    constexpr auto bminus = std::minus<value_type>{};

    constexpr auto equal_range = [](auto const& l, auto const& r){
        BOOST_CHECK_EQUAL_COLLECTIONS(ublas::begin(l), ublas::end(l), ublas::begin(r), ublas::end(r));
    };

    constexpr auto check_same_tensor = [uplus1, uplus2, bplus, bminus, equal_range]<typename tensor_t>(auto /*id*/, tensor_t t){
        auto v = value_type{};

        BOOST_TEST_CONTEXT("[Retrive Extents(With Same Static Rank Tensor)] static rank("<< t.rank() <<") tensor"){
            
            ublas::iota(t,v);
            auto const& e = t.extents();

            equal_range(ublas::detail::retrieve_extents( t ), e);
            
            // uexpr1 = t+1
            // uexpr2 = 2+t
            auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_t>( t, uplus1 );
            auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_t>( t, uplus2 );

            equal_range( ublas::detail::retrieve_extents( uexpr1 ), e );
            equal_range( ublas::detail::retrieve_extents( uexpr2 ), e );

            // bexpr_uexpr = (t+1) + (2+t)
            auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( uexpr1, uexpr2, bplus );

            equal_range( ublas::detail::retrieve_extents( bexpr_uexpr ), e );


            // bexpr_bexpr_uexpr = ((t+1) + (2+t)) - t
            auto bexpr_bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( bexpr_uexpr, t, bminus );

            equal_range( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr ), e );
        }
    };

    auto check_different_tensor = [uplus1, uplus2, bplus, bminus, equal_range, this]<typename IType, typename tensor_t>(IType /*id*/, tensor_t t1){
        
        constexpr std::size_t i = IType::value;

        if constexpr(i < fixture_size - 1ul){

            auto v = value_type{};
            auto t2 = std::get<i + 1ul>( this->collection );
            
            if constexpr(t1.rank() == t2.rank()){

                BOOST_TEST_CONTEXT("[Retrive Extents(With Different Static Rank Tensor)] static rank("<< t1.rank() << ", "<< t2.rank() <<") tensor"){
                    
                    ublas::iota(t1,v);
                    ublas::iota(t2,v + value_type{1});

                    BOOST_CHECK( ublas::detail::retrieve_extents( t1 ) != ublas::detail::retrieve_extents( t2 ) );

                    // uexpr1 = t1+1
                    // uexpr2 = 2+t2
                    auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_t>( t1, uplus1 );
                    auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_t>( t2, uplus2 );

                    equal_range( ublas::detail::retrieve_extents( t1 ), ublas::detail::retrieve_extents( uexpr1 ) );
                    equal_range( ublas::detail::retrieve_extents( t2 ), ublas::detail::retrieve_extents( uexpr2 ) );
                    BOOST_CHECK( ublas::detail::retrieve_extents( uexpr1 ) != ublas::detail::retrieve_extents( uexpr2 ) );

                    // bexpr_uexpr = (t1+1) + (2+t2)
                    auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( uexpr1, uexpr2, bplus );

                    equal_range( ublas::detail::retrieve_extents( bexpr_uexpr ), ublas::detail::retrieve_extents(t1) );


                    // bexpr_bexpr_uexpr = ((t1+1) + (2+t2)) - t2
                    auto bexpr_bexpr_uexpr1 = ublas::detail::make_binary_tensor_expression<tensor_t>( bexpr_uexpr, t2, bminus );

                    equal_range( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr1 ), ublas::detail::retrieve_extents(t2) );


                    // bexpr_bexpr_uexpr = t2 - ((t1+1) + (2+t2))
                    auto bexpr_bexpr_uexpr2 = ublas::detail::make_binary_tensor_expression<tensor_t>( t2, bexpr_uexpr, bminus );

                    equal_range( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr2 ), ublas::detail::retrieve_extents(t2) );
                }
            }

        }

    };


    auto const& self = static_cast<fixture_t const&>(*this);
    ublas::for_each_fixture(self, [check_same_tensor, &check_different_tensor](auto id, auto const& t){
        std::invoke(check_same_tensor, id, t);
        std::invoke(check_different_tensor, id, t);
    });
}

// FIXME: Enable after the strides computation is fixed [ issue #119 ]
// BOOST_TEST_DECORATOR(
//     *boost::unit_test::label("expr_eval_retrive_extents")
//     *boost::unit_test::description("Testing the expression evaluation's retrive extents for static tensor")
// )
// BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static,
//     TestTupleType,
//     boost::numeric::ublas::test_types,
//     boost::numeric::ublas::tuple_fixture_tensor_static<TestTupleType>
// )
// {
//     namespace ublas = boost::numeric::ublas;
//     using value_type = typename TestTupleType::first_type;
//     using fixture_t = ublas::tuple_fixture_tensor_static<TestTupleType>;

//     constexpr auto uplus1 = [](auto const& a){return a + value_type(1); };
//     constexpr auto uplus2 = [](auto const& a){return value_type(2) + a; };
//     constexpr auto bplus  = std::plus <value_type>{};
//     constexpr auto bminus = std::minus<value_type>{};

//     constexpr auto equal_range = [](auto const& l, auto const& r){
//         BOOST_CHECK_EQUAL_COLLECTIONS(ublas::begin(l), ublas::end(l), ublas::begin(r), ublas::end(r));
//     };

//     constexpr auto check_same_tensor = [uplus1, uplus2, bplus, bminus, equal_range]<typename tensor_t>(auto /*id*/, tensor_t t){
//         auto v = value_type{};

//         BOOST_TEST_CONTEXT("[Retrive Extents(With Same Static Tensor)] rank("<< t.rank() <<") static tensor"){
            
//             ublas::iota(t,v);
//             auto const& e = t.extents();

//             equal_range(ublas::detail::retrieve_extents( t ), e);
            
//             // uexpr1 = t+1
//             // uexpr2 = 2+t
//             auto uexpr1 = ublas::detail::make_unary_tensor_expression<tensor_t>( t, uplus1 );
//             auto uexpr2 = ublas::detail::make_unary_tensor_expression<tensor_t>( t, uplus2 );

//             equal_range( ublas::detail::retrieve_extents( uexpr1 ), e );
//             equal_range( ublas::detail::retrieve_extents( uexpr2 ), e );

//             // bexpr_uexpr = (t+1) + (2+t)
//             auto bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( uexpr1, uexpr2, bplus );

//             equal_range( ublas::detail::retrieve_extents( bexpr_uexpr ), e );


//             // bexpr_bexpr_uexpr = ((t+1) + (2+t)) - t
//             auto bexpr_bexpr_uexpr = ublas::detail::make_binary_tensor_expression<tensor_t>( bexpr_uexpr, t, bminus );

//             equal_range( ublas::detail::retrieve_extents( bexpr_bexpr_uexpr ), e );
//         }
//     };

//     auto const& self = static_cast<fixture_t const&>(*this);
//     ublas::for_each_fixture(self, check_same_tensor);
// }


BOOST_AUTO_TEST_SUITE_END()
