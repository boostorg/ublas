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
#include "fixture_utility.hpp"

BOOST_AUTO_TEST_SUITE(test_einstein_notation, 
    * boost::unit_test::description("Validate Einstien Notation")
)

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("einstien_notation")
    *boost::unit_test::description("Testing the einstien notation for dynamic tensor")
)
BOOST_AUTO_TEST_CASE_TEMPLATE(test_tensor_dynamic,
    TestTupleType,
    boost::numeric::ublas::test_types
)
{
    namespace ublas = boost::numeric::ublas;
    namespace index = ublas::index;
    using value_type = typename TestTupleType::first_type;
    using layout_type = typename TestTupleType::second_type;
    using inner_t = inner_type_t<value_type>;
    using tensor_t = ublas::tensor_dynamic<value_type, layout_type>;

    BOOST_TEST_CONTEXT("[Dynamic Tensor Einstien Notation] tensor with LHS[5, 3] : RHS[3, 4]"){
        
        auto A = tensor_t(5,3);
        auto B = tensor_t{3,4};

        for(auto j = 0u; j < A.size(1); ++j){
            for(auto i = 0u; i < A.size(0); ++i){
                A.at( i,j ) = value_type{ static_cast< inner_t >(i+1) };
            }
        }

        for(auto j = 0u; j < B.size(1); ++j){
            for(auto i = 0u; i < B.size(0); ++i){
                B.at( i,j ) = value_type{ static_cast< inner_t >(i+1) };
            }
        }

        auto AB = A(index::_,index::_e) * B(index::_e,index::_);

        for(auto j = 0u; j < AB.size(1); ++j){
            for(auto i = 0u; i < AB.size(0); ++i){
                auto e0   = B.size(0);
                auto sum  = std::div(e0 * ( e0 + 1 ), 2);
                auto quot = value_type{ static_cast<inner_t>(sum.quot) };
                BOOST_CHECK_EQUAL( AB.at(i,j) , A.at(i,0)*quot );
            }
        }

    }

    BOOST_TEST_CONTEXT("[Dynamic Tensor Einstien Notation] tensor with LHS[4, 5, 3] : RHS[3, 4, 2]"){
        
        auto A = tensor_t{4,5,3};
        auto B = tensor_t{3,4,2};

        for(auto k = 0u; k < A.size(2); ++k){
            for(auto j = 0u; j < A.size(1); ++j){
                for(auto i = 0u; i < A.size(0); ++i){
                A.at( i,j,k ) = value_type{ static_cast< inner_t >(i+1) };
                }
            }
        }

        for(auto k = 0u; k < B.size(2); ++k){
            for(auto j = 0u; j < B.size(1); ++j){
                for(auto i = 0u; i < B.size(0); ++i){
                B.at( i,j,k ) = value_type{ static_cast< inner_t >(i+1) };
                }
            }
        }

        auto AB = A(index::_d,index::_,index::_f) * B(index::_f,index::_d,index::_);

        auto const nf = ( B.size(0) * (B.size(0)+1) / 2 );
        auto const nd = ( A.size(0) * (A.size(0)+1) / 2 );

        for(auto j = 0u; j < AB.size(1); ++j){
            for(auto i = 0u; i < AB.size(0); ++i){
                auto const rhs = value_type{ static_cast< inner_t >(nf * nd) };
                BOOST_CHECK_EQUAL( AB.at( i,j ) , rhs );
            }
        }

    }

    BOOST_TEST_CONTEXT("[Dynamic Tensor Einstien Notation] tensor with LHS[4, 3] : RHS[3, 4, 2]"){
        
        auto A = tensor_t{4,3};
        auto B = tensor_t{3,4,2};

        for(auto j = 0u; j < A.size(1); ++j){
            for(auto i = 0u; i < A.size(0); ++i){
                A.at( i,j ) = value_type{ static_cast< inner_t >(i+1) };
            }
        }


        for(auto k = 0u; k < B.size(2); ++k){
            for(auto j = 0u; j < B.size(1); ++j){
                for(auto i = 0u; i < B.size(0); ++i){
                    B.at( i,j,k ) = value_type{ static_cast< inner_t >(i+1) };
                }
            }
        }

        auto AB = A(index::_d,index::_f) * B(index::_f,index::_d,index::_);

        // n*(n+1)/2;
        auto const nf = ( B.size(0) * (B.size(0)+1) / 2 );
        auto const nd = ( A.size(0) * (A.size(0)+1) / 2 );

        for(auto i = 0u; i < AB.size(0); ++i){
            auto const rhs = value_type{ static_cast< inner_t >(nf * nd) };
            BOOST_CHECK_EQUAL ( AB.at( i  ) , rhs );
        }

    }
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("einstien_notation")
    *boost::unit_test::description("Testing the einstien notation for static rank tensor")
)
BOOST_AUTO_TEST_CASE_TEMPLATE(test_tensor_static_rank,
    TestTupleType,
    boost::numeric::ublas::test_types
)
{
    namespace ublas = boost::numeric::ublas;
    namespace index = ublas::index;
    using value_type = typename TestTupleType::first_type;
    using layout_type = typename TestTupleType::second_type;
    using inner_t = inner_type_t<value_type>;
    using tensor2_t = ublas::tensor_static_rank<value_type, 2ul, layout_type>;
    using tensor3_t = ublas::tensor_static_rank<value_type, 3ul, layout_type>;

    BOOST_TEST_CONTEXT("[Static Rank Tensor Einstien Notation] tensor with LHS[5, 3] : RHS[3, 4]"){
        
        auto A = tensor2_t(5,3);
        auto B = tensor2_t(3,4);

        for(auto j = 0u; j < A.size(1); ++j){
            for(auto i = 0u; i < A.size(0); ++i){
                A.at( i,j ) = value_type{ static_cast< inner_t >(i+1) };
            }
        }

        for(auto j = 0u; j < B.size(1); ++j){
            for(auto i = 0u; i < B.size(0); ++i){
                B.at( i,j ) = value_type{ static_cast< inner_t >(i+1) };
            }
        }

        auto AB = A(index::_,index::_e) * B(index::_e,index::_);

        for(auto j = 0u; j < AB.size(1); ++j){
            for(auto i = 0u; i < AB.size(0); ++i){
                auto e0   = B.size(0);
                auto sum  = std::div(e0 * ( e0 + 1 ), 2);
                auto quot = value_type{ static_cast<inner_t>(sum.quot) };
                BOOST_CHECK_EQUAL( AB.at(i,j) , A.at(i,0)*quot );
            }
        }

    }

    BOOST_TEST_CONTEXT("[Static Rank Tensor Einstien Notation] tensor with LHS[4, 5, 3] : RHS[3, 4, 2]"){
        
        auto A = tensor3_t(4,5,3);
        auto B = tensor3_t(3,4,2);

        for(auto k = 0u; k < A.size(2); ++k){
            for(auto j = 0u; j < A.size(1); ++j){
                for(auto i = 0u; i < A.size(0); ++i){
                A.at( i,j,k ) = value_type{ static_cast< inner_t >(i+1) };
                }
            }
        }

        for(auto k = 0u; k < B.size(2); ++k){
            for(auto j = 0u; j < B.size(1); ++j){
                for(auto i = 0u; i < B.size(0); ++i){
                B.at( i,j,k ) = value_type{ static_cast< inner_t >(i+1) };
                }
            }
        }

        auto AB = A(index::_d,index::_,index::_f) * B(index::_f,index::_d,index::_);

        auto const nf = ( B.size(0) * (B.size(0)+1) / 2 );
        auto const nd = ( A.size(0) * (A.size(0)+1) / 2 );

        for(auto j = 0u; j < AB.size(1); ++j){
            for(auto i = 0u; i < AB.size(0); ++i){
                auto const rhs = value_type{ static_cast< inner_t >(nf * nd) };
                BOOST_CHECK_EQUAL( AB.at( i,j ) , rhs );
            }
        }

    }

    // FIXME: Enable this test after the issue #121 has been fixed.
    // BOOST_TEST_CONTEXT("[Static Rank Tensor Einstien Notation] tensor with LHS[4, 3] : RHS[3, 4, 2]"){
        
    //     auto A = tensor2_t(4,3);
    //     auto B = tensor3_t(3,4,2);

    //     for(auto j = 0u; j < A.size(1); ++j){
    //         for(auto i = 0u; i < A.size(0); ++i){
    //             A.at( i,j ) = value_type{ static_cast< inner_t >(i+1) };
    //         }
    //     }


    //     for(auto k = 0u; k < B.size(2); ++k){
    //         for(auto j = 0u; j < B.size(1); ++j){
    //             for(auto i = 0u; i < B.size(0); ++i){
    //                 B.at( i,j,k ) = value_type{ static_cast< inner_t >(i+1) };
    //             }
    //         }
    //     }

    //     auto AB = A(index::_d,index::_f) * B(index::_f,index::_d,index::_);

    //     // n*(n+1)/2;
    //     auto const nf = ( B.size(0) * (B.size(0)+1) / 2 );
    //     auto const nd = ( A.size(0) * (A.size(0)+1) / 2 );

    //     for(auto i = 0u; i < AB.size(0); ++i){
    //         auto const rhs = value_type{ static_cast< inner_t >(nf * nd) };
    //         BOOST_CHECK_EQUAL ( AB.at( i  ) , rhs );
    //     }

    // }
}

// TODO: Add support of static tensor product then enable it.
// BOOST_TEST_DECORATOR(
//     *boost::unit_test::label("einstien_notation")
//     *boost::unit_test::description("Testing the einstien notation for static tensor")
// )
// BOOST_AUTO_TEST_CASE_TEMPLATE(test_tensor_static,
//     TestTupleType,
//     boost::numeric::ublas::test_types
// )
// {
//     namespace ublas = boost::numeric::ublas;
//     namespace index = ublas::index;
//     using value_type = typename TestTupleType::first_type;
//     using layout_type = typename TestTupleType::second_type;
//     using inner_t = inner_type_t<value_type>;

//     BOOST_TEST_CONTEXT("[Static Tensor Einstien Notation] tensor with LHS[5, 3] : RHS[3, 4]"){
        
//         auto A = ublas::tensor_static<value_type, ublas::extents<5, 3>, layout_type>();
//         auto B = ublas::tensor_static<value_type, ublas::extents<3, 4>, layout_type>();

//         for(auto j = 0u; j < A.size(1); ++j){
//             for(auto i = 0u; i < A.size(0); ++i){
//                 A.at( i,j ) = value_type{ static_cast< inner_t >(i+1) };
//             }
//         }

//         for(auto j = 0u; j < B.size(1); ++j){
//             for(auto i = 0u; i < B.size(0); ++i){
//                 B.at( i,j ) = value_type{ static_cast< inner_t >(i+1) };
//             }
//         }

//         auto AB = A(index::_,index::_e) * B(index::_e,index::_);

//         for(auto j = 0u; j < AB.size(1); ++j){
//             for(auto i = 0u; i < AB.size(0); ++i){
//                 auto e0   = B.size(0);
//                 auto sum  = std::div(e0 * ( e0 + 1 ), 2);
//                 auto quot = value_type{ static_cast<inner_t>(sum.quot) };
//                 BOOST_CHECK_EQUAL( AB.at(i,j) , A.at(i,0)*quot );
//             }
//         }

//     }

//     BOOST_TEST_CONTEXT("[Static Einstien Notation] tensor with LHS[4, 5, 3] : RHS[3, 4, 2]"){
        
//         auto A = ublas::tensor_static<value_type, ublas::extents<4, 5, 3>, layout_type>();
//         auto B = ublas::tensor_static<value_type, ublas::extents<3, 4, 2>, layout_type>();

//         for(auto k = 0u; k < A.size(2); ++k){
//             for(auto j = 0u; j < A.size(1); ++j){
//                 for(auto i = 0u; i < A.size(0); ++i){
//                 A.at( i,j,k ) = value_type{ static_cast< inner_t >(i+1) };
//                 }
//             }
//         }

//         for(auto k = 0u; k < B.size(2); ++k){
//             for(auto j = 0u; j < B.size(1); ++j){
//                 for(auto i = 0u; i < B.size(0); ++i){
//                 B.at( i,j,k ) = value_type{ static_cast< inner_t >(i+1) };
//                 }
//             }
//         }

//         auto AB = A(index::_d,index::_,index::_f) * B(index::_f,index::_d,index::_);

//         auto const nf = ( B.size(0) * (B.size(0)+1) / 2 );
//         auto const nd = ( A.size(0) * (A.size(0)+1) / 2 );

//         for(auto j = 0u; j < AB.size(1); ++j){
//             for(auto i = 0u; i < AB.size(0); ++i){
//                 auto const rhs = value_type{ static_cast< inner_t >(nf * nd) };
//                 BOOST_CHECK_EQUAL( AB.at( i,j ) , rhs );
//             }
//         }

//     }

//     // BOOST_TEST_CONTEXT("[Static Einstien Notation] tensor with LHS[4, 3] : RHS[3, 4, 2]"){
        
//     //     auto A = ublas::tensor_static<value_type, ublas::extents<4, 3>, layout_type>();
//     //     auto B = ublas::tensor_static<value_type, ublas::extents<3, 4, 2>, layout_type>();

//     //     for(auto j = 0u; j < A.size(1); ++j){
//     //         for(auto i = 0u; i < A.size(0); ++i){
//     //             A.at( i,j ) = value_type{ static_cast< inner_t >(i+1) };
//     //         }
//     //     }


//     //     for(auto k = 0u; k < B.size(2); ++k){
//     //         for(auto j = 0u; j < B.size(1); ++j){
//     //             for(auto i = 0u; i < B.size(0); ++i){
//     //                 B.at( i,j,k ) = value_type{ static_cast< inner_t >(i+1) };
//     //             }
//     //         }
//     //     }

//     //     auto AB = A(index::_d,index::_f) * B(index::_f,index::_d,index::_);

//     //     // n*(n+1)/2;
//     //     auto const nf = ( B.size(0) * (B.size(0)+1) / 2 );
//     //     auto const nd = ( A.size(0) * (A.size(0)+1) / 2 );

//     //     for(auto i = 0u; i < AB.size(0); ++i){
//     //         auto const rhs = value_type{ static_cast< inner_t >(nf * nd) };
//     //         BOOST_CHECK_EQUAL ( AB.at( i  ) , rhs );
//     //     }

//     // }
// }


BOOST_AUTO_TEST_SUITE_END()
