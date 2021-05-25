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

BOOST_AUTO_TEST_SUITE(test_functions_real_imag_conj,
    * boost::unit_test::description("Validate Real, Imaginary, and Conjugate Product")
)

template<typename TensorType, typename T>
struct rebind_tensor;

template<typename TensorType, typename T>
using rebind_tensor_t = typename rebind_tensor<TensorType,T>::type;

template<typename T, typename ValueType, typename LayoutType>
struct rebind_tensor<boost::numeric::ublas::tensor_dynamic<ValueType, LayoutType>, T>
{
    using type = boost::numeric::ublas::tensor_dynamic<T, LayoutType>;
};

template<typename T, typename ValueType, std::size_t N, typename LayoutType>
struct rebind_tensor<boost::numeric::ublas::tensor_static_rank<ValueType, N, LayoutType>, T>
{
    using type = boost::numeric::ublas::tensor_static_rank<T, N, LayoutType>;
};

template<typename T, typename ValueType, typename E, typename LayoutType>
struct rebind_tensor<boost::numeric::ublas::tensor_static<ValueType, E, LayoutType>, T>
{
    using type = boost::numeric::ublas::tensor_static<T, E, LayoutType>;
};

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("real_imag_conj_fn")
    *boost::unit_test::description("Testing the real, imag, and conj for dynamic tensor")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_dynamic,
    TestTupleType,
    boost::numeric::ublas::cpp_basic_std_types,
    boost::numeric::ublas::tuple_fixture_tensor_dynamic<TestTupleType>
)
{
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using fixture_t = ublas::tuple_fixture_tensor_dynamic<TestTupleType>;
    using complex_t = std::complex<value_type>;

    auto const& self = static_cast<fixture_t const&>(*this);

    ublas::for_each_fixture(self, []<typename tensor_type>(auto /*id*/, tensor_type const& t){
        using complex_tensor_t = rebind_tensor_t<tensor_type, complex_t>;
        auto const rank = t.rank();
        
        if(rank < 1ul) return;
        auto const& n = t.extents();

        BOOST_TEST_CONTEXT("[Dynamic RIC(1)] tensor with rank("<< rank <<")"){
            auto a = t;
            ublas::iota(a, value_type{0});
            auto r0  = a;
            auto r00 = complex_tensor_t(n);

            tensor_type b = (a+a) / value_type{2};
            tensor_type r1 = ublas::real( (a+a) / value_type{2}  );
            std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::real( l );  }   );
            BOOST_CHECK_EQUAL_COLLECTIONS(r0.begin(), r0.end(), r1.begin(), r1.end());

            tensor_type r2 = ublas::imag( (a+a) / value_type{2}  );
            std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::imag( l );  }   );
            BOOST_CHECK_EQUAL_COLLECTIONS(r0.begin(), r0.end(), r2.begin(), r2.end());
            
            complex_tensor_t r3 = ublas::conj( (a+a) / value_type{2}  );
            std::transform(  b.begin(), b.end(), r00.begin(), [](auto const& l){ return std::conj( l );  }   );
            BOOST_CHECK_EQUAL_COLLECTIONS(r00.begin(), r00.end(), r3.begin(), r3.end());

        }

        BOOST_TEST_CONTEXT("[Dynamic RIC(2)] tensor with rank("<< rank <<")"){
            auto a = complex_tensor_t(n);
            ublas::iota(a, value_type{0});

            auto r0 = t;
            auto r00  = complex_tensor_t(n);

            complex_tensor_t b = (a+a) / complex_t( 2,2 );
            tensor_type r1 = ublas::real( (a+a) / complex_t( 2,2 )  );
            std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::real( l );  }   );
            BOOST_CHECK_EQUAL_COLLECTIONS(r0.begin(), r0.end(), r1.begin(), r1.end());

            tensor_type r2 = ublas::imag( (a+a) / complex_t( 2,2 )  );
            std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::imag( l );  }   );
            BOOST_CHECK_EQUAL_COLLECTIONS(r0.begin(), r0.end(), r2.begin(), r2.end());
            
            complex_tensor_t r3 = ublas::conj( (a+a) / complex_t( 2,2 )  );
            std::transform(  b.begin(), b.end(), r00.begin(), [](auto const& l){ return std::conj( l );  }   );
            BOOST_CHECK_EQUAL_COLLECTIONS(r00.begin(), r00.end(), r3.begin(), r3.end());

        }
    });


}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("real_imag_conj_fn")
    *boost::unit_test::description("Testing the real, imag, and conj for static rank tensor")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static_rank,
    TestTupleType,
    boost::numeric::ublas::cpp_basic_std_types,
    boost::numeric::ublas::tuple_fixture_tensor_static_rank<TestTupleType>
)
{
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using fixture_t = ublas::tuple_fixture_tensor_static_rank<TestTupleType>;
    using complex_t = std::complex<value_type>;

    auto const& self = static_cast<fixture_t const&>(*this);

    ublas::for_each_fixture(self, []<typename tensor_type>(auto /*id*/, tensor_type const& t){
        using complex_tensor_t = rebind_tensor_t<tensor_type, complex_t>;
        using extents_type = typename tensor_type::extents_type;

        auto const& n = t.extents();
        constexpr auto rank = std::tuple_size_v<extents_type>;

        if constexpr (rank > 0ul) {

            BOOST_TEST_CONTEXT("[Static Rank RIC(1)] tensor with rank("<< rank <<")"){
                auto a = t;
                ublas::iota(a, value_type{0});
                auto r0  = a;
                auto r00 = complex_tensor_t(n);

                tensor_type b = (a+a) / value_type{2};
                tensor_type r1 = ublas::real( (a+a) / value_type{2}  );
                std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::real( l );  }   );
                BOOST_CHECK_EQUAL_COLLECTIONS(r0.begin(), r0.end(), r1.begin(), r1.end());

                tensor_type r2 = ublas::imag( (a+a) / value_type{2}  );
                std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::imag( l );  }   );
                BOOST_CHECK_EQUAL_COLLECTIONS(r0.begin(), r0.end(), r2.begin(), r2.end());
                
                complex_tensor_t r3 = ublas::conj( (a+a) / value_type{2}  );
                std::transform(  b.begin(), b.end(), r00.begin(), [](auto const& l){ return std::conj( l );  }   );
                BOOST_CHECK_EQUAL_COLLECTIONS(r00.begin(), r00.end(), r3.begin(), r3.end());

            }

            BOOST_TEST_CONTEXT("[Static Rank RIC(2)] tensor with rank("<< rank <<")"){
                auto a = complex_tensor_t(n);
                ublas::iota(a, value_type{0});

                auto r0 = t;
                auto r00  = complex_tensor_t(n);

                complex_tensor_t b = (a+a) / complex_t( 2,2 );
                tensor_type r1 = ublas::real( (a+a) / complex_t( 2,2 )  );
                std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::real( l );  }   );
                BOOST_CHECK_EQUAL_COLLECTIONS(r0.begin(), r0.end(), r1.begin(), r1.end());

                tensor_type r2 = ublas::imag( (a+a) / complex_t( 2,2 )  );
                std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::imag( l );  }   );
                BOOST_CHECK_EQUAL_COLLECTIONS(r0.begin(), r0.end(), r2.begin(), r2.end());
                
                complex_tensor_t r3 = ublas::conj( (a+a) / complex_t( 2,2 )  );
                std::transform(  b.begin(), b.end(), r00.begin(), [](auto const& l){ return std::conj( l );  }   );
                BOOST_CHECK_EQUAL_COLLECTIONS(r00.begin(), r00.end(), r3.begin(), r3.end());

            }
        }

    });


}

// TODO: Add support for static tensor and then enable this test
// BOOST_TEST_DECORATOR(
//     *boost::unit_test::label("real_imag_conj_fn")
//     *boost::unit_test::description("Testing the real, imag, and conj for static tensor")
// )
// BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static,
//     TestTupleType,
//     boost::numeric::ublas::cpp_basic_std_types,
//     boost::numeric::ublas::tuple_fixture_tensor_static<TestTupleType>
// )
// {
//     namespace ublas = boost::numeric::ublas;
//     using value_type = typename TestTupleType::first_type;
//     using fixture_t = ublas::tuple_fixture_tensor_static<TestTupleType>;
//     using complex_t = std::complex<value_type>;

//     auto const& self = static_cast<fixture_t const&>(*this);

//     ublas::for_each_fixture(self, []<typename tensor_type>(auto /*id*/, tensor_type const& t){
//         using complex_tensor_t = rebind_tensor_t<tensor_type, complex_t>;
//         using extents_type = typename tensor_type::extents_type;

//         constexpr auto rank = ublas::size_v<extents_type>;

//         if constexpr (rank > 1ul) {

//             BOOST_TEST_CONTEXT("[Static RIC(1)] tensor with rank("<< rank <<")"){
//                 auto a = t;
//                 ublas::iota(a, value_type{0});
//                 auto r0  = a;
//                 auto r00 = complex_tensor_t();

//                 tensor_type b = (a+a) / value_type{2};
//                 tensor_type r1 = ublas::real( (a+a) / value_type{2}  );
//                 std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::real( l );  }   );
//                 BOOST_CHECK_EQUAL_COLLECTIONS(r0.begin(), r0.end(), r1.begin(), r1.end());

//                 tensor_type r2 = ublas::imag( (a+a) / value_type{2}  );
//                 std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::imag( l );  }   );
//                 BOOST_CHECK_EQUAL_COLLECTIONS(r0.begin(), r0.end(), r2.begin(), r2.end());
                
//                 complex_tensor_t r3 = ublas::conj( (a+a) / value_type{2}  );
//                 std::transform(  b.begin(), b.end(), r00.begin(), [](auto const& l){ return std::conj( l );  }   );
//                 BOOST_CHECK_EQUAL_COLLECTIONS(r00.begin(), r00.end(), r3.begin(), r3.end());

//             }

//             BOOST_TEST_CONTEXT("[Static RIC(2)] tensor with rank("<< rank <<")"){
//                 auto a = complex_tensor_t();
//                 ublas::iota(a, value_type{0});

//                 auto r0 = t;
//                 auto r00  = complex_tensor_t();

//                 complex_tensor_t b = (a+a) / complex_t( 2,2 );
//                 tensor_type r1 = ublas::real( (a+a) / complex_t( 2,2 )  );
//                 std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::real( l );  }   );
//                 BOOST_CHECK_EQUAL_COLLECTIONS(r0.begin(), r0.end(), r1.begin(), r1.end());

//                 tensor_type r2 = ublas::imag( (a+a) / complex_t( 2,2 )  );
//                 std::transform(  b.begin(), b.end(), r0.begin(), [](auto const& l){ return std::imag( l );  }   );
//                 BOOST_CHECK_EQUAL_COLLECTIONS(r0.begin(), r0.end(), r2.begin(), r2.end());
                
//                 complex_tensor_t r3 = ublas::conj( (a+a) / complex_t( 2,2 )  );
//                 std::transform(  b.begin(), b.end(), r00.begin(), [](auto const& l){ return std::conj( l );  }   );
//                 BOOST_CHECK_EQUAL_COLLECTIONS(r00.begin(), r00.end(), r3.begin(), r3.end());

//             }
//         }

//     });


// }

BOOST_AUTO_TEST_SUITE_END()
