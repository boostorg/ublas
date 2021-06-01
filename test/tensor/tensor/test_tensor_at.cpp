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

BOOST_AUTO_TEST_SUITE(test_tensor_at, 
    *boost::unit_test::description("Validate At Method")
    *boost::unit_test::depends_on("test_extents_size")
)

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::tensor_dynamic::at(i)")
    *boost::unit_test::description("Testing the dynamic tensor's read/write [at] method for single-dimensional")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_dynamic_single_dim,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_dynamic<TestTupleType>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using fixture_type = ublas::tuple_fixture_tensor_dynamic<TestTupleType>;

    constexpr auto check = [](auto t){
        auto const sz = t.size();
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < sz; ++i, v += value_type(1)){
            BOOST_TEST_CHECKPOINT("[At Method] rank("<< rank <<") dynamic tensor at(" << i << ") = " << v);
            t.at(i) = v;
            BOOST_REQUIRE_EQUAL(t.at(i), v);
        }
    };

    auto const self = static_cast<fixture_type const&>(*this);
    ublas::for_each_fixture(self, [&check](auto /*id*/, auto const& t){
        check(t);
    });
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::tensor_dynamic::at(i)")
    *boost::unit_test::description("Testing the dynamic tensor's read/write [at] method for multidimensional")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_dynamic_multi_dim,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_dynamic<TestTupleType>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using fixture_type = ublas::tuple_fixture_tensor_dynamic<TestTupleType>;

    constexpr auto check1 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            BOOST_TEST_CHECKPOINT("[At Method] rank("<< rank <<") dynamic tensor at(" << i << ") = " << v);
            t.at(i) = v;
            BOOST_REQUIRE_EQUAL(t.at(i), v);
            v += value_type(1);
        }
    };

    constexpr auto check2 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            for(auto j = 0ul; j < t.size(1); ++j){
                BOOST_TEST_CHECKPOINT("[At Method] rank("<< rank <<") dynamic tensor at(" << i <<", "<< j << ") = " << v);
                t.at(i,j) = v;
                BOOST_REQUIRE_EQUAL(t.at(i,j), v);
                v += value_type(1);
            }
        }

        BOOST_CHECK_THROW((void)t.at(0,0,0), std::invalid_argument);
        BOOST_CHECK_THROW((void)t.at(0,0,0,0), std::invalid_argument);
    };

    constexpr auto check3 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            for(auto j = 0ul; j < t.size(1); ++j){
                for(auto k = 0ul; k < t.size(2); ++k){
                    BOOST_TEST_CHECKPOINT("[At Method] rank("<< rank <<") dynamic tensor at(" << i <<", "<< j <<", "<< k << ") = " << v);
                    t.at(i,j,k) = v;
                    BOOST_REQUIRE_EQUAL(t.at(i,j,k), v);
                    v += value_type(1);
                }
            }
        }
        BOOST_CHECK_THROW((void)t.at(0,0), std::invalid_argument);
        BOOST_CHECK_THROW((void)t.at(0,0,0,0), std::invalid_argument);
    };

    constexpr auto check4 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            for(auto j = 0ul; j < t.size(1); ++j){
                for(auto k = 0ul; k < t.size(2); ++k){
                    for(auto l = 0ul; l < t.size(3); ++l){
                        BOOST_TEST_CHECKPOINT("[At Method] rank("<< rank <<") dynamic tensor at(" << i <<", "<< j <<", "<< k <<", "<< l << ") = " << v);
                        t.at(i,j,k,l) = v;
                        BOOST_REQUIRE_EQUAL(t.at(i,j,k,l), v);
                        v += value_type(1);
                    }
                }
            }
        }
        BOOST_CHECK_THROW((void)t.at(0,0), std::invalid_argument);
        BOOST_CHECK_THROW((void)t.at(0,0,0), std::invalid_argument);
        BOOST_CHECK_THROW((void)t.at(0,0,0,0,0), std::invalid_argument);
    };

    auto const self = static_cast<fixture_type const&>(*this);
    ublas::for_each_fixture(self, [&check1, &check2, &check3, &check4](auto /*id*/, auto const& t){
        if(t.rank() == 1) check1(t);
        else if(t.rank() == 2) check2(t);
        else if(t.rank() == 3) check3(t);
        else if(t.rank() == 4) check4(t);
        else assert(false && "Not Implemented");

    });
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::tensor_static_rank::at(i)")
    *boost::unit_test::description("Testing the static rank tensor's read/write [at] method for single-dimensional")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static_rank_single_dim,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_static_rank<TestTupleType>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using fixture_type = ublas::tuple_fixture_tensor_static_rank<TestTupleType>;

    constexpr auto check = [](auto t){
        auto const sz = t.size();
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < sz; ++i, v += value_type(1)){
            BOOST_TEST_CHECKPOINT("[At Method] rank("<< rank <<") static rank tensor at(" << i << ") = " << v);
            t.at(i) = v;
            BOOST_REQUIRE_EQUAL(t.at(i), v);
        }
    };

    auto const self = static_cast<fixture_type const&>(*this);
    ublas::for_each_fixture(self, [&check](auto /*id*/, auto const& t){
        check(t);
    });
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::tensor_static_rank::at(i)")
    *boost::unit_test::description("Testing the static rank tensor's read/write [at] method for multidimensional")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static_rank_multi_dim,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_static_rank<TestTupleType>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using fixture_type = ublas::tuple_fixture_tensor_static_rank<TestTupleType>;

    constexpr auto check1 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            BOOST_TEST_CHECKPOINT("[At Method] static rank("<< rank <<") rank tensor at(" << i << ") = " << v);
            t.at(i) = v;
            BOOST_REQUIRE_EQUAL(t.at(i), v);
            v += value_type(1);
        }
    };

    constexpr auto check2 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            for(auto j = 0ul; j < t.size(1); ++j){
                BOOST_TEST_CHECKPOINT("[At Method] static rank("<< rank <<") rank tensor at(" << i <<", "<< j << ") = " << v);
                t.at(i,j) = v;
                BOOST_REQUIRE_EQUAL(t.at(i,j), v);
                v += value_type(1);
            }
        }
    };

    constexpr auto check3 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            for(auto j = 0ul; j < t.size(1); ++j){
                for(auto k = 0ul; k < t.size(2); ++k){
                    BOOST_TEST_CHECKPOINT("[At Method] static rank("<< rank <<") rank tensor at(" << i <<", "<< j <<", "<< k << ") = " << v);
                    t.at(i,j,k) = v;
                    BOOST_REQUIRE_EQUAL(t.at(i,j,k), v);
                    v += value_type(1);
                }
            }
        }
    };

    constexpr auto check4 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            for(auto j = 0ul; j < t.size(1); ++j){
                for(auto k = 0ul; k < t.size(2); ++k){
                    for(auto l = 0ul; l < t.size(3); ++l){
                        BOOST_TEST_CHECKPOINT("[At Method] static rank("<< rank <<") rank tensor at(" << i <<", "<< j <<", "<< k <<", "<< l << ") = " << v);
                        t.at(i,j,k,l) = v;
                        BOOST_REQUIRE_EQUAL(t.at(i,j,k,l), v);
                        v += value_type(1);
                    }
                }
            }
        }
    };

    auto const self = static_cast<fixture_type const&>(*this);
    ublas::for_each_fixture(self, [&check1, &check2, &check3, &check4]<typename TensorType>(auto /*id*/, TensorType const& t){
        constexpr auto e = typename TensorType::extents_type{};
        if constexpr(ublas::size(e) == 1) check1(t);
        else if constexpr(ublas::size(e) == 2) check2(t);
        else if constexpr(ublas::size(e) == 3) check3(t);
        else if constexpr(ublas::size(e) == 4) check4(t);
        else assert(false && "Not Implemented");

    });
}


BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::tensor_static::at(i)")
    *boost::unit_test::description("Testing the static tensor's read/write [at] method for single-dimensional")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static_single_dim,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_static<TestTupleType>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using fixture_type = ublas::tuple_fixture_tensor_static<TestTupleType>;

    constexpr auto check = [](auto t){
        auto const sz = t.size();
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < sz; ++i, v += value_type(1)){
            BOOST_TEST_CHECKPOINT("[At Method] rank("<< rank <<") static tensor at(" << i << ") = " << v);
            t.at(i) = v;
            BOOST_REQUIRE_EQUAL(t.at(i), v);
        }
    };
    
    auto const self = static_cast<fixture_type const&>(*this);
    ublas::for_each_fixture(self, [&check](auto /*id*/, auto const& t){
        check(t);
    });
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::tensor_static::at(i)")
    *boost::unit_test::description("Testing the static tensor's read/write [at] method for multidimensional")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static_multi_dim,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_static<TestTupleType>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using fixture_type = ublas::tuple_fixture_tensor_static<TestTupleType>;

    constexpr auto check1 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            BOOST_TEST_CHECKPOINT("[At Method] rank("<< rank <<") static tensor at(" << i << ") = " << v);
            t.at(i) = v;
            BOOST_REQUIRE_EQUAL(t.at(i), v);
            v += value_type(1);
        }
    };

    constexpr auto check2 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            for(auto j = 0ul; j < t.size(1); ++j){
                BOOST_TEST_CHECKPOINT("[At Method] rank("<< rank <<") static tensor at(" << i <<", "<< j << ") = " << v);
                t.at(i,j) = v;
                BOOST_REQUIRE_EQUAL(t.at(i,j), v);
                v += value_type(1);
            }
        }
    };

    constexpr auto check3 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            for(auto j = 0ul; j < t.size(1); ++j){
                for(auto k = 0ul; k < t.size(2); ++k){
                    BOOST_TEST_CHECKPOINT("[At Method] rank("<< rank <<") static tensor at(" << i <<", "<< j <<", "<< k << ") = " << v);
                    t.at(i,j,k) = v;
                    BOOST_REQUIRE_EQUAL(t.at(i,j,k), v);
                    v += value_type(1);
                }
            }
        }
    };

    constexpr auto check4 = [](auto t){
        auto const rank = t.rank();
        value_type v{}; 
        for(auto i = 0ul; i < t.size(0); ++i){
            for(auto j = 0ul; j < t.size(1); ++j){
                for(auto k = 0ul; k < t.size(2); ++k){
                    for(auto l = 0ul; l < t.size(3); ++l){
                        BOOST_TEST_CHECKPOINT("[At Method] rank("<< rank <<") static tensor at(" << i <<", "<< j <<", "<< k <<", "<< l << ") = " << v);
                        t.at(i,j,k,l) = v;
                        BOOST_REQUIRE_EQUAL(t.at(i,j,k,l), v);
                        v += value_type(1);
                    }
                }
            }
        }
    };

    auto const self = static_cast<fixture_type const&>(*this);
    ublas::for_each_fixture(self, [&check1, &check2, &check3, &check4]<typename TensorType>(auto /*id*/, TensorType const& t){
        using extents_type = typename TensorType::extents_type;
        if constexpr(ublas::size_v<extents_type> == 1) check1(t);
        else if constexpr(ublas::size_v<extents_type> == 2) check2(t);
        else if constexpr(ublas::size_v<extents_type> == 3) check3(t);
        else if constexpr(ublas::size_v<extents_type> == 4) check4(t);
        else assert(false && "Not Implemented");
    });
}


BOOST_AUTO_TEST_SUITE_END()
