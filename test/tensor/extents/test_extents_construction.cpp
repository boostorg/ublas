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

BOOST_AUTO_TEST_SUITE(test_extents_constructor, 
    *boost::unit_test::description("Validate constructor")
    *boost::unit_test::depends_on("test_extents_empty")
    *boost::unit_test::depends_on("test_extents_size")
    *boost::unit_test::depends_on("test_extents_at")
    *boost::unit_test::depends_on("test_extents_access_operator")
)


BOOST_FIXTURE_TEST_CASE(test_extents_dynamic, 
    boost::numeric::ublas::fixture_extents_dynamic<std::size_t>,
    *boost::unit_test::label("boost::numeric::ublas::extents<>(...)")
    *boost::unit_test::description("Testing dynamic extents construction"))
{
    namespace ublas = boost::numeric::ublas;
    using extents_type = ublas::extents<>;
    
    BOOST_TEST_CONTEXT("[Dynamic Extents Constructor] default constructor"){
        auto e = extents_type();
        BOOST_REQUIRE_EQUAL (ublas::size(e), 0ul);
        BOOST_REQUIRE       (ublas::empty(e));
        BOOST_CHECK_THROW   ((void)e.at(0), std::out_of_range);
    }
    
    BOOST_TEST_CONTEXT("[Dynamic Extents Constructor] construction using base type"){
        using extents_base_type = typename extents_type::base_type;
        extents_base_type vec {1,2,3,4,5};
        auto e = extents_type(std::move(vec));
        BOOST_REQUIRE_EQUAL ( ublas::size(e), 5ul);
        BOOST_REQUIRE       (!ublas::empty(e));
        BOOST_CHECK_THROW   ((void)e.at(5), std::out_of_range);
        BOOST_REQUIRE_EQUAL (e[0], 1ul);
        BOOST_REQUIRE_EQUAL (e[1], 2ul);
        BOOST_REQUIRE_EQUAL (e[2], 3ul);
        BOOST_REQUIRE_EQUAL (e[3], 4ul);
        BOOST_REQUIRE_EQUAL (e[4], 5ul);
    }

    BOOST_TEST_CONTEXT("[Dynamic Extents Constructor] construction using initilizer list"){
        auto e = extents_type{1,2,3,4,5};
        BOOST_REQUIRE_EQUAL ( ublas::size(e), 5ul);
        BOOST_REQUIRE       (!ublas::empty(e));
        BOOST_CHECK_THROW   ((void)e.at(5), std::out_of_range);
        BOOST_REQUIRE_EQUAL (e[0], 1ul);
        BOOST_REQUIRE_EQUAL (e[1], 2ul);
        BOOST_REQUIRE_EQUAL (e[2], 3ul);
        BOOST_REQUIRE_EQUAL (e[3], 4ul);
        BOOST_REQUIRE_EQUAL (e[4], 5ul);
    }
    
    BOOST_TEST_CONTEXT("[Dynamic Extents Constructor] construction using iterator"){
        std::array<std::size_t, 5ul> arr {1,2,3,4,5};
        auto e = extents_type(arr.begin(), arr.end());
        BOOST_REQUIRE_EQUAL ( ublas::size(e), 5ul);
        BOOST_REQUIRE       (!ublas::empty(e));
        BOOST_CHECK_THROW   ((void)e.at(5), std::out_of_range);
        BOOST_REQUIRE_EQUAL (e[0], 1ul);
        BOOST_REQUIRE_EQUAL (e[1], 2ul);
        BOOST_REQUIRE_EQUAL (e[2], 3ul);
        BOOST_REQUIRE_EQUAL (e[3], 4ul);
        BOOST_REQUIRE_EQUAL (e[4], 5ul);
    }

    BOOST_TEST_CONTEXT("[Dynamic Extents Constructor] copy constructor"){
        auto e = extents_type(n4231);
        BOOST_REQUIRE_EQUAL ( ublas::size(e), 4ul);
        BOOST_REQUIRE       (!ublas::empty(e));
        BOOST_CHECK_THROW   ((void)e.at(5), std::out_of_range);
        BOOST_REQUIRE_EQUAL (e[0], 4ul);
        BOOST_REQUIRE_EQUAL (e[1], 2ul);
        BOOST_REQUIRE_EQUAL (e[2], 3ul);
        BOOST_REQUIRE_EQUAL (e[3], 1ul);
    }

    BOOST_TEST_CONTEXT("[Dynamic Extents Constructor] move constructor"){
        auto temp = extents_type(n4231);
        auto e = extents_type(std::move(temp));
        BOOST_REQUIRE_EQUAL ( ublas::size(e), 4ul);
        BOOST_REQUIRE       (!ublas::empty(e));
        BOOST_CHECK_THROW   ((void)e.at(5), std::out_of_range);
        BOOST_REQUIRE_EQUAL (e[0], 4ul);
        BOOST_REQUIRE_EQUAL (e[1], 2ul);
        BOOST_REQUIRE_EQUAL (e[2], 3ul);
        BOOST_REQUIRE_EQUAL (e[3], 1ul);
        BOOST_REQUIRE_EQUAL ( ublas::size(temp), 0ul);
        BOOST_REQUIRE       ( ublas::empty(temp));
    }

    BOOST_TEST_CONTEXT("[Dynamic Extents Constructor] invalid construction"){
        BOOST_CHECK_THROW( (extents_type{1, 1, 0}), std::invalid_argument);
        BOOST_CHECK_THROW( (extents_type{1, 0   }), std::invalid_argument);
        BOOST_CHECK_THROW( (extents_type{0      }), std::invalid_argument);
        BOOST_CHECK_THROW( (extents_type{0, 1   }), std::invalid_argument);
    }

}

BOOST_FIXTURE_TEST_CASE(test_extents_static_rank, 
    boost::numeric::ublas::fixture_extents_static_rank<std::size_t>,
    *boost::unit_test::label("boost::numeric::ublas::extents<>(...)")
    *boost::unit_test::description("Testing static_rank extents construction"))
{
    namespace ublas = boost::numeric::ublas;
    
    using extents_type = ublas::extents<4>;

    BOOST_TEST_CONTEXT("[Static Rank Extents Constructor] default constructor"){
        auto e = extents_type();
        BOOST_REQUIRE_EQUAL ( ublas::size(e), 4ul);
        BOOST_REQUIRE       (!ublas::empty(e));
        BOOST_CHECK_NO_THROW((void)e.at(0));
        BOOST_CHECK_NO_THROW((void)e.at(1));
        BOOST_CHECK_NO_THROW((void)e.at(2));
        BOOST_CHECK_NO_THROW((void)e.at(3));
        BOOST_CHECK_THROW   ((void)e.at(4), std::out_of_range);
    }
    
    BOOST_TEST_CONTEXT("[Static Rank Extents Constructor] construction using base type"){
        using extents_base_type = typename extents_type::base_type;
        extents_base_type arr {1,2,3,4};
        auto e = extents_type(std::move(arr));
        BOOST_REQUIRE_EQUAL ( ublas::size(e), 4ul);
        BOOST_REQUIRE       (!ublas::empty(e));
        BOOST_CHECK_THROW   ((void)e.at(5), std::out_of_range);
        BOOST_REQUIRE_EQUAL (e[0], 1ul);
        BOOST_REQUIRE_EQUAL (e[1], 2ul);
        BOOST_REQUIRE_EQUAL (e[2], 3ul);
        BOOST_REQUIRE_EQUAL (e[3], 4ul);
    }

    BOOST_TEST_CONTEXT("[Static Rank Extents Constructor] construction using initilizer list"){
        auto e = extents_type{1,2,3,4};
        BOOST_REQUIRE_EQUAL ( ublas::size(e), 4ul);
        BOOST_REQUIRE       (!ublas::empty(e));
        BOOST_CHECK_THROW   ((void)e.at(5), std::out_of_range);
        BOOST_REQUIRE_EQUAL (e[0], 1ul);
        BOOST_REQUIRE_EQUAL (e[1], 2ul);
        BOOST_REQUIRE_EQUAL (e[2], 3ul);
        BOOST_REQUIRE_EQUAL (e[3], 4ul);
    }
    
    BOOST_TEST_CONTEXT("[Static Rank Extents Constructor] construction using iterator"){
        std::array<std::size_t, 4ul> arr {1,2,3,4};
        auto e = extents_type(arr.begin(), arr.end());
        BOOST_REQUIRE_EQUAL ( ublas::size(e), 4ul);
        BOOST_REQUIRE       (!ublas::empty(e));
        BOOST_CHECK_THROW   ((void)e.at(5), std::out_of_range);
        BOOST_REQUIRE_EQUAL (e[0], 1ul);
        BOOST_REQUIRE_EQUAL (e[1], 2ul);
        BOOST_REQUIRE_EQUAL (e[2], 3ul);
        BOOST_REQUIRE_EQUAL (e[3], 4ul);
    }

    BOOST_TEST_CONTEXT("[Static Rank Extents Constructor] copy constructor"){
        auto e = extents_type(n4231);
        BOOST_REQUIRE_EQUAL ( ublas::size(e), 4ul);
        BOOST_REQUIRE       (!ublas::empty(e));
        BOOST_CHECK_THROW   ((void)e.at(5), std::out_of_range);
        BOOST_REQUIRE_EQUAL (e[0], 4ul);
        BOOST_REQUIRE_EQUAL (e[1], 2ul);
        BOOST_REQUIRE_EQUAL (e[2], 3ul);
        BOOST_REQUIRE_EQUAL (e[3], 1ul);
    }

    BOOST_TEST_CONTEXT("[Static Rank Extents Constructor] move constructor"){
        auto temp = extents_type(n4231);
        auto e = extents_type(std::move(temp));
        BOOST_REQUIRE_EQUAL ( ublas::size(e), 4ul);
        BOOST_REQUIRE       (!ublas::empty(e));
        BOOST_CHECK_THROW   ((void)e.at(5), std::out_of_range);
        BOOST_REQUIRE_EQUAL (e[0], 4ul);
        BOOST_REQUIRE_EQUAL (e[1], 2ul);
        BOOST_REQUIRE_EQUAL (e[2], 3ul);
        BOOST_REQUIRE_EQUAL (e[3], 1ul);
        BOOST_REQUIRE_EQUAL ( ublas::size(temp), 4ul);
        BOOST_REQUIRE       (!ublas::empty(temp));
    }

    BOOST_TEST_CONTEXT("[Static Rank Extents Constructor] invalid construction"){
        BOOST_CHECK_THROW( (extents_type{1, 1, 0      }), std::length_error);
        BOOST_CHECK_THROW( (extents_type{1, 1, 1, 1, 1}), std::length_error);
        BOOST_CHECK_THROW( (extents_type{1, 1, 0, 1   }), std::invalid_argument);
        BOOST_CHECK_THROW( (extents_type{0, 1, 0, 1   }), std::invalid_argument);
    }

}

BOOST_FIXTURE_TEST_CASE(test_extents_static, 
    boost::numeric::ublas::fixture_extents_static<std::size_t>,
    *boost::unit_test::label("boost::numeric::ublas::extents<>(...)")
    *boost::unit_test::description("Testing static extents construction"))
{
    namespace ublas = boost::numeric::ublas;
    
    using extents_type = ublas::extents<4,2,3,1>;

    BOOST_TEST_CONTEXT("[Static Rank Extents Constructor] default constructor"){
        auto e = extents_type();
        BOOST_REQUIRE_EQUAL ( ublas::size_v<extents_type>, 4ul);
        BOOST_REQUIRE       (!ublas::empty_v<extents_type>);
        BOOST_REQUIRE_EQUAL (e[0], 4ul);
        BOOST_REQUIRE_EQUAL (e[1], 2ul);
        BOOST_REQUIRE_EQUAL (e[2], 3ul);
        BOOST_REQUIRE_EQUAL (e[3], 1ul);
        BOOST_CHECK_THROW   ((void)e.at(4), std::out_of_range);
    }

    BOOST_TEST_CONTEXT("[Static Rank Extents Constructor] copy constructor"){
        auto e = extents_type(n4231);
        BOOST_REQUIRE_EQUAL ( ublas::size_v<extents_type>, 4ul);
        BOOST_REQUIRE       (!ublas::empty_v<extents_type>);
        BOOST_CHECK_THROW   ((void)e.at(5), std::out_of_range);
        BOOST_REQUIRE_EQUAL (e[0], 4ul);
        BOOST_REQUIRE_EQUAL (e[1], 2ul);
        BOOST_REQUIRE_EQUAL (e[2], 3ul);
        BOOST_REQUIRE_EQUAL (e[3], 1ul);
    }

    BOOST_TEST_CONTEXT("[Static Rank Extents Constructor] move constructor"){
        auto temp = extents_type(n4231);
        auto e = extents_type(std::move(temp));
        BOOST_REQUIRE_EQUAL ( ublas::size_v<extents_type>, 4ul);
        BOOST_REQUIRE       (!ublas::empty_v<extents_type>);
        BOOST_CHECK_THROW   ((void)e.at(5), std::out_of_range);
        BOOST_REQUIRE_EQUAL (e[0], 4ul);
        BOOST_REQUIRE_EQUAL (e[1], 2ul);
        BOOST_REQUIRE_EQUAL (e[2], 3ul);
        BOOST_REQUIRE_EQUAL (e[3], 1ul);
        
        BOOST_REQUIRE_EQUAL ( ublas::size_v<decltype(temp)>, 4ul);
        BOOST_REQUIRE       (!ublas::empty_v<decltype(temp)>);
        BOOST_REQUIRE_EQUAL (temp[0], 4ul);
        BOOST_REQUIRE_EQUAL (temp[1], 2ul);
        BOOST_REQUIRE_EQUAL (temp[2], 3ul);
        BOOST_REQUIRE_EQUAL (temp[3], 1ul);
    }

}

BOOST_AUTO_TEST_SUITE_END()
