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
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

BOOST_AUTO_TEST_SUITE(test_tensor_assignment_operator, 
    *boost::unit_test::description("Validate tensor operator=(...)")
    *boost::unit_test::depends_on("test_tensor_constructor")
)


BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::tensor_dynamic::operator=(...)")
    *boost::unit_test::description("Testing dynamic tensor operator=(...)")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_dynamic,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_dynamic<TestTupleType>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using layout_type = typename TestTupleType::second_type;
    using tensor_t = ublas::tensor_dynamic<value_type,layout_type>;
    using strides_t = typename tensor_t::strides_type;
    using fixture_type = boost::numeric::ublas::tuple_fixture_tensor_dynamic<TestTupleType>;
    
    // FIXME: Enable this test after the expressions are fixed
    // BOOST_TEST_CONTEXT("[Dynamic Tensor Assignment Operator] assigning tensor expression"){
        
    //     auto const e = fixture_type::t32.extents();

    //     auto const a = tensor_t(e, value_type(1));
    //     BOOST_REQUIRE_EQUAL ( a.size(), 6ul);
    //     BOOST_REQUIRE_EQUAL ( a.rank(), 2ul);
    //     BOOST_REQUIRE       (!a.empty());

    //     auto const three = value_type(3);
    //     auto const expr = a + three * a;
        
    //     auto t = tensor_t(e);
    //     t = expr;

    //     BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
    //     BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
    //     BOOST_REQUIRE       (!t.empty());
    //     BOOST_CHECK_NO_THROW( (void)t.at(3ul));
    //     BOOST_CHECK_THROW   ( (void)t.at(0,1,1),   std::invalid_argument);
    //     BOOST_CHECK_THROW   ( (void)t.at(0,1,2),   std::invalid_argument);
    //     BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
    //     BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
    //     BOOST_REQUIRE ( t.extents() == e );
    //     auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
    //     BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
    //     auto const four = value_type(4);
    //     for(auto i = 0ul; i < t.size(); ++i){
    //         BOOST_TEST_CHECKPOINT("[Tensor Expression  Assignment] testing tensor element(" << i << ") = " <<t[i]);
    //         BOOST_REQUIRE_EQUAL(t[i], four);
    //     }
    // }

    BOOST_TEST_CONTEXT("[Dynamic Tensor Copy Assignment Operator] copying tensor"){
        auto o = fixture_type::t32;
        o = value_type(1);
        BOOST_REQUIRE_EQUAL ( o.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( o.rank(), 2ul);
        BOOST_REQUIRE       (!o.empty());

        auto t = tensor_t();
        t = o;

        BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
        BOOST_REQUIRE       (!t.empty());

        BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        BOOST_CHECK_THROW   ( (void)t.at(0,1,1),   std::invalid_argument);
        BOOST_CHECK_THROW   ( (void)t.at(0,1,2),   std::invalid_argument);
        BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == o.extents() );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t), std::end(t), std::begin(o), std::end(o));
    }

    BOOST_TEST_CONTEXT("[Dynamic Tensor Move Assignment Operator] moving tensor"){
        auto const& e = fixture_type::t32.extents();
        auto o = tensor_t(e, value_type(1));
        BOOST_REQUIRE_EQUAL ( o.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( o.rank(), 2ul);
        BOOST_REQUIRE       (!o.empty());

        auto t = tensor_t();
        t = std::move(o);

        BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
        BOOST_REQUIRE       (!t.empty());
        BOOST_REQUIRE_EQUAL ( o.size(), 0ul);
        BOOST_REQUIRE_EQUAL ( o.rank(), 0ul);
        BOOST_REQUIRE       ( o.empty());

        BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        BOOST_CHECK_THROW   ( (void)t.at(0,1,1),   std::invalid_argument);
        BOOST_CHECK_THROW   ( (void)t.at(0,1,2),   std::invalid_argument);
        BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == e );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
        auto const one = value_type(1);
        for(auto i = 0ul; i < t.size(); ++i){
            BOOST_TEST_CHECKPOINT("[Tensor Move constructor] testing tensor element(" << i << ") = " <<t[i]);
            BOOST_REQUIRE_EQUAL(t[i], one);
        }
    }

    BOOST_TEST_CONTEXT("[Dynamic Tensor Assingment Operator] assigning default value"){
        auto const e = fixture_type::t4231.extents();
        auto t = tensor_t(e);
        t = value_type(1);

        BOOST_REQUIRE_EQUAL ( t.size(), 24ul);
        BOOST_REQUIRE_EQUAL ( t.rank(),  4ul);
        BOOST_REQUIRE       (!t.empty());
        BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        BOOST_CHECK_THROW   ( (void)t.at(0,1,1),   std::invalid_argument);
        BOOST_CHECK_THROW   ( (void)t.at(0,1,2),   std::invalid_argument);
        BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        BOOST_CHECK_THROW   ( (void)t.at(5,2,3,4), std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == e );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 4, 8, 24ul} : strides_t{6ul, 3ul, 1ul, 1ul});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
        for(auto const& v : t){
            BOOST_REQUIRE_EQUAL(v, value_type(1));
        }
    }

}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::tensor_static_rank::operator=(...)")
    *boost::unit_test::description("Testing static rank tensor operator=(...)")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static_rank,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_static_rank<TestTupleType>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using layout_type = typename TestTupleType::second_type;
    using tensor_t = ublas::tensor_static_rank<value_type,2ul,layout_type>;
    using strides_t = typename tensor_t::strides_type;
    using fixture_type = boost::numeric::ublas::tuple_fixture_tensor_static_rank<TestTupleType>;
    
    BOOST_TEST_CONTEXT("[Static Rank Tensor Assingment Operator] assigning default value"){
        auto const e = fixture_type::t32.extents();
        auto t = tensor_t(e);
        t = value_type(1);

        BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
        BOOST_REQUIRE       (!t.empty());
        BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == e );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
        for(auto const& v : t){
            BOOST_REQUIRE_EQUAL(v, value_type(1));
        }
    }

    // FIXME: Enable this test after the expressions are fixed
    // BOOST_TEST_CONTEXT("[Static Rank Tensor Assignment Operator] assigning tensor expression"){
        
    //     auto const e = fixture_type::t32.extents();

    //     auto a = tensor_t(e);
    //     a = value_type(1);
    //     BOOST_REQUIRE_EQUAL ( a.size(), 6ul);
    //     BOOST_REQUIRE_EQUAL ( a.rank(), 2ul);
    //     BOOST_REQUIRE       (!a.empty());

    //     auto const three = value_type(3);
    //     auto const expr = a + three * a;
        
    //     auto t = tensor_t(e);
    //     t = expr;

    //     BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
    //     BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
    //     BOOST_REQUIRE       (!t.empty());
    //     BOOST_CHECK_NO_THROW( (void)t.at(3ul));
    //     BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
    //     BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
    //     BOOST_REQUIRE ( t.extents() == e );
    //     auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
    //     BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
    //     auto const four = value_type(4);
    //     for(auto i = 0ul; i < t.size(); ++i){
    //         BOOST_TEST_CHECKPOINT("[Tensor Expression  Assignment] testing tensor element(" << i << ") = " <<t[i]);
    //         BOOST_REQUIRE_EQUAL(t[i], four);
    //     }
    // }

    BOOST_TEST_CONTEXT("[Static Rank Tensor Copy Assignment Operator] copying tensor"){
        auto o = fixture_type::t32;
        o = value_type(1);
        BOOST_REQUIRE_EQUAL ( o.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( o.rank(), 2ul);
        BOOST_REQUIRE       (!o.empty());

        auto t = tensor_t();
        t = o;

        BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
        BOOST_REQUIRE       (!t.empty());

        BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == o.extents() );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t), std::end(t), std::begin(o), std::end(o));
    }

    BOOST_TEST_CONTEXT("[Static Rank Tensor Move Assignment Operator] moving tensor"){
        auto const& e = fixture_type::t32.extents();
        auto o = tensor_t(e);
        o = value_type(1);

        BOOST_REQUIRE_EQUAL ( o.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( o.rank(), 2ul);
        BOOST_REQUIRE       (!o.empty());

        auto t = tensor_t();
        t = std::move(o);

        BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
        BOOST_REQUIRE       (!t.empty());
        BOOST_REQUIRE_EQUAL ( o.size(), 0ul);
        BOOST_REQUIRE_EQUAL ( o.rank(), 2ul);
        BOOST_REQUIRE       ( o.empty());

        BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == e );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
        auto const one = value_type(1);
        for(auto i = 0ul; i < t.size(); ++i){
            BOOST_TEST_CHECKPOINT("[Tensor Move constructor] testing tensor element(" << i << ") = " <<t[i]);
            BOOST_REQUIRE_EQUAL(t[i], one);
        }
    }

}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::tensor_static::operator=(...)")
    *boost::unit_test::description("Testing static tensor operator=(...)")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_static<TestTupleType>
){
    namespace ublas = boost::numeric::ublas;
    using fixture_type  = boost::numeric::ublas::tuple_fixture_tensor_static<TestTupleType>;
    using value_type    = typename TestTupleType::first_type;
    using layout_type   = typename TestTupleType::second_type;
    using tensor_t      = typename fixture_type::t32_type;
    using shape_t       = typename tensor_t::extents_type;
    using strides_t     = typename tensor_t::strides_type;
    
    BOOST_TEST_CONTEXT("[Static Tensor Assingment Operator] assigning default value"){
        auto t = tensor_t();
        t = value_type(1);

        BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
        BOOST_REQUIRE       (!t.empty());
        // BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        // BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        // BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == shape_t{} );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
        for(auto const& v : t){
            BOOST_REQUIRE_EQUAL(v, value_type(1));
        }
    }

    // FIXME: Enable this test after the expressions are fixed
    // BOOST_TEST_CONTEXT("[Static Tensor Assignment Operator] assigning tensor expression"){
        
    //     auto const a = tensor_t(value_type(1));
    //     BOOST_REQUIRE_EQUAL ( a.size(), 6ul);
    //     BOOST_REQUIRE_EQUAL ( a.rank(), 2ul);
    //     BOOST_REQUIRE       (!a.empty());

    //     auto const three = value_type(3);
    //     auto const expr = a + three * a;
        
    //     auto t = tensor_t();
    //     t = expr;

    //     BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
    //     BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
    //     BOOST_REQUIRE       (!t.empty());
    //     // TODO: Enable after adding bounds checking to static tensor
    //     // BOOST_CHECK_NO_THROW( (void)t.at(3ul));
    //     // BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
    //     // BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
    //     BOOST_REQUIRE ( t.extents() == shape_t{} );
    //     auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
    //     BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
    //     auto const four = value_type(4);
    //     for(auto i = 0ul; i < t.size(); ++i){
    //         BOOST_TEST_CHECKPOINT("[Tensor Expression  Assignment] testing tensor element(" << i << ") = " <<t[i]);
    //         BOOST_REQUIRE_EQUAL(t[i], four);
    //     }
    // }

    BOOST_TEST_CONTEXT("[Static Tensor Copy Assignment Operator] copying tensor"){
        auto const o = tensor_t(value_type(1));
        BOOST_REQUIRE_EQUAL ( o.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( o.rank(), 2ul);
        BOOST_REQUIRE       (!o.empty());

        auto t = tensor_t();
        t = o;

        BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
        BOOST_REQUIRE       (!t.empty());

        // TODO: Enable after adding bounds checking to static tensor
        // BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        // BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        // BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == o.extents() );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t), std::end(t), std::begin(o), std::end(o));
    }

    BOOST_TEST_CONTEXT("[Static Tensor Move Assignment Operator] moving tensor"){
        
        auto const o = tensor_t(value_type(1));
        BOOST_REQUIRE_EQUAL ( o.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( o.rank(), 2ul);
        BOOST_REQUIRE       (!o.empty());

        auto t = tensor_t();
        t = std::move(o);

        BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
        BOOST_REQUIRE       (!t.empty());
        BOOST_REQUIRE_EQUAL ( o.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( o.rank(), 2ul);
        BOOST_REQUIRE       (!o.empty());

        // TODO: Enable after adding bounds checking to static tensor
        // BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        // BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        // BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == shape_t{} );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
        auto const one = value_type(1);
        for(auto i = 0ul; i < t.size(); ++i){
            BOOST_TEST_CHECKPOINT("[Tensor Move constructor] testing tensor element(" << i << ") = " <<t[i]);
            BOOST_REQUIRE_EQUAL(t[i], one);
        }
    }

}

BOOST_AUTO_TEST_SUITE_END()
