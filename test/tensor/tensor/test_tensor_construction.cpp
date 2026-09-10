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

BOOST_AUTO_TEST_SUITE(test_tensor_constructor, 
    *boost::unit_test::description("Validate tensor constructor")
    *boost::unit_test::depends_on("test_tensor_empty")
    *boost::unit_test::depends_on("test_tensor_size")
    *boost::unit_test::depends_on("test_tensor_at")
    *boost::unit_test::depends_on("test_tensor_access_operator")
)


BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::tensor_dynamic(...)")
    *boost::unit_test::description("Testing dynamic tensor construction")
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
    using shape_t = typename tensor_t::extents_type;
    using strides_t = typename tensor_t::strides_type;
    using fixture_type = boost::numeric::ublas::tuple_fixture_tensor_dynamic<TestTupleType>;
    
    BOOST_TEST_CONTEXT("[Dynamic Tensor Construction] default constructor"){
        auto const t = tensor_t{};
        BOOST_REQUIRE_EQUAL (t.size(), 0ul);
        BOOST_REQUIRE_EQUAL (t.rank(), 0ul);
        BOOST_REQUIRE       (t.empty());
        BOOST_CHECK_THROW   ((void)t.at(0), std::out_of_range);
    }
    
    BOOST_TEST_CONTEXT("[Dynamic Tensor Construction] construction using variadic integral arguments"){
        auto const t = tensor_t(1,2,3);
        BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( t.rank(), 3ul);
        BOOST_REQUIRE       (!t.empty());
        BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        BOOST_CHECK_NO_THROW( (void)t.at(0,1,1));
        BOOST_CHECK_NO_THROW( (void)t.at(0,1,2));
        BOOST_CHECK_THROW   ( (void)t.at(7ul), std::out_of_range);
        BOOST_CHECK_THROW   ( (void)t.at(1,2,3), std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == (shape_t{1ul,2ul,3ul}) );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1,1,2} : strides_t{6, 3, 1});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
    }
    
    BOOST_TEST_CONTEXT("[Dynamic Tensor Construction] construction using extents"){
        auto const e = fixture_type::t4231.extents();
        auto const t = tensor_t(e);
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
    }
    
    BOOST_TEST_CONTEXT("[Dynamic Tensor Construction] construction using extents and default value"){
        auto const e = fixture_type::t4231.extents();
        auto const t = tensor_t(e, value_type(1));
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
    
    BOOST_TEST_CONTEXT("[Dynamic Tensor Construction] construction using extents and container"){
        using container_type = typename tensor_t::container_type;
        auto const c = container_type{
            value_type(1), value_type(2),
            value_type(3), value_type(4),
            value_type(5), value_type(6)
        };
        auto const e = fixture_type::t32.extents();
        auto const t = tensor_t(e, c);
        BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
        BOOST_REQUIRE       (!t.empty());
        BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        BOOST_CHECK_THROW   ( (void)t.at(0,1,1),   std::invalid_argument);
        BOOST_CHECK_THROW   ( (void)t.at(0,1,2),   std::invalid_argument);
        BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == e );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t), std::end(t), std::begin(c), std::end(c));
    }

    // TODO: Enable after fixing the issue #116   
    // BOOST_TEST_CONTEXT("[Dynamic Tensor Construction] construction using other layout"){
    //     using other_layout = std::conditional_t<
    //         std::is_same_v<ublas::layout::first_order, layout_type>,
    //         ublas::layout::last_order,
    //         ublas::layout::first_order
    //     >;

    //     auto const e = fixture_type::t32.extents();
    //     auto const o = ublas::tensor_dynamic<value_type,other_layout>(e, value_type(1));
    //     BOOST_REQUIRE_EQUAL ( o.size(), 6ul);
    //     BOOST_REQUIRE_EQUAL ( o.rank(), 2ul);
    //     BOOST_REQUIRE       (!o.empty());
        
    //     auto const t = tensor_t(o);
    //     BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
    //     BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
    //     BOOST_REQUIRE       (!t.empty());

    // }

    // FIXME: Enable this tests after the tensor expression is fixed
    // BOOST_TEST_CONTEXT("[Dynamic Tensor Construction] construction using tensor expression"){
        
    //     auto const e = fixture_type::t32.extents();

    //     auto const a = tensor_t(e, value_type(1));
    //     BOOST_REQUIRE_EQUAL ( a.size(), 6ul);
    //     BOOST_REQUIRE_EQUAL ( a.rank(), 2ul);
    //     BOOST_REQUIRE       (!a.empty());

    //     auto const three = value_type(3);
    //     auto const expr = a + three * a;
    //     auto const t = tensor_t(expr);

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
    //         BOOST_TEST_CHECKPOINT("[Tensor Expression Constructor] Testing tensor element(" << i << ") = " <<t[i]);
    //         BOOST_REQUIRE_EQUAL(t[i], four);
    //     }
    // }

    BOOST_TEST_CONTEXT("[Dynamic Tensor Construction] construction using legacy ublas matrix"){
        
        using matrix_type = typename tensor_t::matrix_type;
        auto const e = fixture_type::t32.extents();
        auto const a = matrix_type(e[0], e[1], value_type(1));
        BOOST_REQUIRE_EQUAL ( a.size1(), e[0]);
        BOOST_REQUIRE_EQUAL ( a.size2(), e[1]);

        auto const t = tensor_t(a);

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
            BOOST_TEST_CHECKPOINT("[Tensor Legacy uBLAS Matrix Constructor] testing tensor element(" << i << ") = " <<t[i]);
            BOOST_REQUIRE_EQUAL(t[i], one);
        }
    }

    BOOST_TEST_CONTEXT("[Dynamic Tensor Construction] construction using legacy ublas vector"){
        using vector_type = typename tensor_t::vector_type;
        auto const e = fixture_type::t2.extents();
        auto const a = vector_type(e[0], value_type(1));
        BOOST_REQUIRE_EQUAL ( a.size(), e[0]);

        auto const t = tensor_t(a);

        BOOST_CHECK_NO_THROW( (void)t.at(1ul));
        BOOST_CHECK_THROW   ( (void)t.at(0,1,1),   std::invalid_argument);
        BOOST_CHECK_THROW   ( (void)t.at(0,1,2),   std::invalid_argument);
        BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == fixture_type::t21.extents() );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 1} : strides_t{1ul, 1ul});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
        auto const one = value_type(1);
        for(auto i = 0ul; i < t.size(); ++i){
            BOOST_TEST_CHECKPOINT("[Tensor Legacy uBLAS Vector Constructor] testing tensor element(" << i << ") = " <<t[i]);
            BOOST_REQUIRE_EQUAL(t[i], one);
        }
    }

    // FIXME: Enable this tests after the tensor expression is fixed
    // BOOST_TEST_CONTEXT("[Dynamic Tensor Construction] construction using legacy ublas matrix expression"){
        
    //     using matrix_type = typename tensor_t::matrix_type;
    //     auto const e = fixture_type::t32.extents();
    //     auto const a = matrix_type(e[0], e[1], value_type(1));
    //     BOOST_REQUIRE_EQUAL ( a.size1(), e[0]);
    //     BOOST_REQUIRE_EQUAL ( a.size2(), e[1]);

    //     auto const expr = a + value_type(3) * a;

    //     auto const t = tensor_t(expr);

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
    //         BOOST_TEST_CHECKPOINT("[Tensor Legacy uBLAS Matrix Expression Constructor] testing tensor element(" << i << ") = " <<t[i]);
    //         BOOST_REQUIRE_EQUAL(t[i], four);
    //     }
    // }

    // FIXME: Enable this tests after the tensor expression is fixed
    // BOOST_TEST_CONTEXT("[Dynamic Tensor Construction] construction using legacy ublas vector expression"){
    //     using vector_type = typename tensor_t::vector_type;
    //     auto const e = fixture_type::t2.extents();
    //     auto const a = vector_type(e[0], value_type(1));
    //     BOOST_REQUIRE_EQUAL ( a.size(), e[0]);

    //     auto const three = value_type(3);
    //     auto const expr = a + three * a;
    //     auto const t = tensor_t(expr);

    //     BOOST_CHECK_NO_THROW( (void)t.at(1ul));
    //     BOOST_CHECK_THROW   ( (void)t.at(0,1,1),   std::invalid_argument);
    //     BOOST_CHECK_THROW   ( (void)t.at(0,1,2),   std::invalid_argument);
    //     BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        
    //     BOOST_REQUIRE ( t.extents() == fixture_type::t21.extents() );
    //     auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 1} : strides_t{1ul, 1ul});
    //     BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
    //     auto const four = value_type(4);
    //     for(auto i = 0ul; i < t.size(); ++i){
    //         BOOST_TEST_CHECKPOINT("[Tensor Legacy uBLAS Vector Expression Constructor] testing tensor element(" << i << ") = " <<t[i]);
    //         BOOST_REQUIRE_EQUAL(t[i], four);
    //     }
    // }

    BOOST_TEST_CONTEXT("[Dynamic Tensor Copy Construction] copy constructing tensor"){
        auto o = fixture_type::t32;
        o = value_type(1);
        BOOST_REQUIRE_EQUAL ( o.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( o.rank(), 2ul);
        BOOST_REQUIRE       (!o.empty());

        auto const t = tensor_t(o);
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

    BOOST_TEST_CONTEXT("[Dynamic Tensor Move Construction] move constructing tensor"){
        auto const& e = fixture_type::t32.extents();
        auto o = tensor_t(e, value_type(1));
        BOOST_REQUIRE_EQUAL ( o.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( o.rank(), 2ul);
        BOOST_REQUIRE       (!o.empty());

        auto const t = tensor_t(std::move(o));
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

}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("boost::numeric::ublas::tensor_static_rank(...)")
    *boost::unit_test::description("Testing dynamic tensor construction")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static_rank,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_static_rank<TestTupleType>
){
    namespace ublas = boost::numeric::ublas;

    using value_type    = typename TestTupleType::first_type;
    using layout_type   = typename TestTupleType::second_type;
    using fixture_type  = boost::numeric::ublas::tuple_fixture_tensor_static_rank<TestTupleType>;
    
    
    BOOST_TEST_CONTEXT("[Static Rank Tensor Construction] default constructor"){
        using tensor_t      = ublas::tensor_static_rank<value_type,3ul,layout_type>;
        auto const t = tensor_t();
        BOOST_REQUIRE_EQUAL (t.size(), 0ul);
        BOOST_REQUIRE_EQUAL (t.rank(), 3ul);
        BOOST_REQUIRE       (t.empty());
        BOOST_CHECK_THROW   ((void)t.at(0), std::out_of_range);
    }
    
    BOOST_TEST_CONTEXT("[Static Rank Tensor Construction] construction using variadic integral arguments"){
        using tensor_t      = ublas::tensor_static_rank<value_type,3ul,layout_type>;
        using shape_t       = typename tensor_t::extents_type;
        using strides_t     = typename tensor_t::strides_type;

        auto const t = tensor_t(1,2,3);
        BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( t.rank(), 3ul);
        BOOST_REQUIRE       (!t.empty());
        BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        BOOST_CHECK_NO_THROW( (void)t.at(0,1,1));
        BOOST_CHECK_NO_THROW( (void)t.at(0,1,2));
        BOOST_CHECK_THROW   ( (void)t.at(7ul), std::out_of_range);
        BOOST_CHECK_THROW   ( (void)t.at(1,2,3), std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == (shape_t{1ul,2ul,3ul}) );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1,1,2} : strides_t{6, 3, 1});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
    }
    
    BOOST_TEST_CONTEXT("[Static Rank Tensor Construction] construction using extents"){
        using tensor_t      = ublas::tensor_static_rank<value_type,4ul,layout_type>;
        using strides_t     = typename tensor_t::strides_type;

        auto const e = fixture_type::t4231.extents();
        auto const t = tensor_t(e);
        BOOST_REQUIRE_EQUAL ( t.size(), 24ul);
        BOOST_REQUIRE_EQUAL ( t.rank(),  4ul);
        BOOST_REQUIRE       (!t.empty());
        BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        BOOST_CHECK_THROW   ( (void)t.at(5,2,3,4), std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == e );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 4, 8, 24ul} : strides_t{6ul, 3ul, 1ul, 1ul});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
    }
    
    BOOST_TEST_CONTEXT("[Static Rank Tensor Construction] construction using extents and container"){
        using tensor_t          = ublas::tensor_static_rank<value_type,2ul,layout_type>;
        using strides_t         = typename tensor_t::strides_type;
        using container_type    = typename tensor_t::container_type;

        auto const c = container_type{
            value_type(1), value_type(2),
            value_type(3), value_type(4),
            value_type(5), value_type(6)
        };
        auto const e = fixture_type::t32.extents();
        auto const t = tensor_t(e, c);
        BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
        BOOST_REQUIRE       (!t.empty());
        BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == e );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t), std::end(t), std::begin(c), std::end(c));
    }

    // TODO: Enable after fixing the issue #116   
    // BOOST_TEST_CONTEXT("[Static Rank Tensor Construction] construction using other layout"){
    //     using other_layout = std::conditional_t<
    //         std::is_same_v<ublas::layout::first_order, layout_type>,
    //         ublas::layout::last_order,
    //         ublas::layout::first_order
    //     >;

    //     auto const e = fixture_type::t32.extents();
    //     auto const o = ublas::tensor_static_rank<value_type,other_layout>(e, value_type(1));
    //     BOOST_REQUIRE_EQUAL ( o.size(), 6ul);
    //     BOOST_REQUIRE_EQUAL ( o.rank(), 2ul);
    //     BOOST_REQUIRE       (!o.empty());
        
    //     auto const t = tensor_t(o);
    //     BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
    //     BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
    //     BOOST_REQUIRE       (!t.empty());

    // }

    // FIXME: Enable this tests after the tensor expression is fixed
    // BOOST_TEST_CONTEXT("[Static Rank Tensor Construction] construction using tensor expression"){

    //     using tensor_t      = ublas::tensor_static_rank<value_type,2ul,layout_type>;
    //     using strides_t     = typename tensor_t::strides_type;
        
    //     auto const e = fixture_type::t32.extents();

    //     auto a = tensor_t(e);
    //     a = value_type(1);

    //     BOOST_REQUIRE_EQUAL ( a.size(), 6ul);
    //     BOOST_REQUIRE_EQUAL ( a.rank(), 2ul);
    //     BOOST_REQUIRE       (!a.empty());

    //     auto const three = value_type(3);
    //     auto const expr = a + three * a;
    //     auto const t = tensor_t(expr);

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
    //         BOOST_TEST_CHECKPOINT("[Tensor Expression Constructor] Testing tensor element(" << i << ") = " <<t[i]);
    //         BOOST_REQUIRE_EQUAL(t[i], four);
    //     }
    // }

    BOOST_TEST_CONTEXT("[Static Rank Tensor Construction] construction using legacy ublas matrix"){
        
        using tensor_t      = ublas::tensor_static_rank<value_type,2ul,layout_type>;
        using strides_t     = typename tensor_t::strides_type;

        using matrix_type = typename tensor_t::matrix_type;
        auto const e = fixture_type::t32.extents();
        auto const a = matrix_type(e[0], e[1], value_type(1));
        BOOST_REQUIRE_EQUAL ( a.size1(), e[0]);
        BOOST_REQUIRE_EQUAL ( a.size2(), e[1]);

        auto const t = tensor_t(a);

        BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == e );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
        auto const one = value_type(1);
        for(auto i = 0ul; i < t.size(); ++i){
            BOOST_TEST_CHECKPOINT("[Tensor Legacy uBLAS Matrix Constructor] testing tensor element(" << i << ") = " <<t[i]);
            BOOST_REQUIRE_EQUAL(t[i], one);
        }
    }

    BOOST_TEST_CONTEXT("[Static Rank Tensor Construction] construction using legacy ublas vector"){
        using tensor_t      = ublas::tensor_static_rank<value_type,2ul,layout_type>;
        using strides_t     = typename tensor_t::strides_type;

        using vector_type = typename tensor_t::vector_type;
        auto const e = fixture_type::t2.extents();
        auto const a = vector_type(e[0], value_type(1));
        BOOST_REQUIRE_EQUAL ( a.size(), e[0]);

        auto const t = tensor_t(a);

        BOOST_CHECK_NO_THROW( (void)t.at(1ul));
        BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == fixture_type::t21.extents() );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 1} : strides_t{1ul, 1ul});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
        auto const one = value_type(1);
        for(auto i = 0ul; i < t.size(); ++i){
            BOOST_TEST_CHECKPOINT("[Tensor Legacy uBLAS Vector Constructor] testing tensor element(" << i << ") = " <<t[i]);
            BOOST_REQUIRE_EQUAL(t[i], one);
        }
    }
    
    // FIXME: Enable this tests after the tensor expression is fixed
    // BOOST_TEST_CONTEXT("[Static Rank Tensor Construction] construction using legacy ublas matrix expression"){
    //     using tensor_t      = ublas::tensor_static_rank<value_type,2ul,layout_type>;
    //     using strides_t     = typename tensor_t::strides_type;

    //     using matrix_type = typename tensor_t::matrix_type;
    //     auto const e = fixture_type::t32.extents();
    //     auto const a = matrix_type(e[0], e[1], value_type(1));
    //     BOOST_REQUIRE_EQUAL ( a.size1(), e[0]);
    //     BOOST_REQUIRE_EQUAL ( a.size2(), e[1]);

    //     auto const three = value_type(3);
    //     auto const expr = a + three * a;

    //     auto const t = tensor_t(expr);

    //     BOOST_CHECK_NO_THROW( (void)t.at(3ul));
    //     BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
    //     BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
    //     BOOST_REQUIRE ( t.extents() == e );
    //     auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
    //     BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
    //     auto const four = value_type(4);
    //     for(auto i = 0ul; i < t.size(); ++i){
    //         BOOST_TEST_CHECKPOINT("[Tensor Legacy uBLAS Matrix Expression Constructor] testing tensor element(" << i << ") = " <<t[i]);
    //         BOOST_REQUIRE_EQUAL(t[i], four);
    //     }
    // }

    // FIXME: Enable this tests after the tensor expression is fixed
    // BOOST_TEST_CONTEXT("[Static Rank Tensor Construction] construction using legacy ublas vector expression"){
    //     using tensor_t      = ublas::tensor_static_rank<value_type,2ul,layout_type>;
    //     using strides_t     = typename tensor_t::strides_type;

    //     using vector_type = typename tensor_t::vector_type;
    //     auto const e = fixture_type::t2.extents();
    //     auto const a = vector_type(e[0], value_type(1));
    //     BOOST_REQUIRE_EQUAL ( a.size(), e[0]);

    //     auto const three = value_type(3);
    //     auto const expr = a + three * a;
    //     auto const t = tensor_t(expr);

    //     BOOST_CHECK_NO_THROW( (void)t.at(1ul));
    //     BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        
    //     BOOST_REQUIRE ( t.extents() == fixture_type::t21.extents() );
    //     auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 1} : strides_t{1ul, 1ul});
    //     BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
    //     auto const four = value_type(4);
    //     for(auto i = 0ul; i < t.size(); ++i){
    //         BOOST_TEST_CHECKPOINT("[Tensor Legacy uBLAS Vector Expression Constructor] testing tensor element(" << i << ") = " <<t[i]);
    //         BOOST_REQUIRE_EQUAL(t[i], four);
    //     }
    // }

    BOOST_TEST_CONTEXT("[Static Rank Tensor Copy Construction] copy constructing tensor"){
        using tensor_t      = ublas::tensor_static_rank<value_type,2ul,layout_type>;
        using strides_t     = typename tensor_t::strides_type;

        auto o = fixture_type::t32;
        o = value_type(1);
        BOOST_REQUIRE_EQUAL ( o.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( o.rank(), 2ul);
        BOOST_REQUIRE       (!o.empty());

        auto const t = tensor_t(o);
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

    BOOST_TEST_CONTEXT("[Static Rank Tensor Move Construction] move constructing tensor"){
        using tensor_t      = ublas::tensor_static_rank<value_type,2ul,layout_type>;
        using strides_t     = typename tensor_t::strides_type;

        auto const& e = fixture_type::t32.extents();
        auto o = tensor_t(e);
        o = value_type(1);

        BOOST_REQUIRE_EQUAL ( o.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( o.rank(), 2ul);
        BOOST_REQUIRE       (!o.empty());

        auto const t = tensor_t(std::move(o));
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
    *boost::unit_test::label("boost::numeric::ublas::tensor_static(...)")
    *boost::unit_test::description("Testing dynamic tensor construction")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_tensor_static,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::tuple_fixture_tensor_static<TestTupleType>
){
    namespace ublas = boost::numeric::ublas;

    using value_type    = typename TestTupleType::first_type;
    using layout_type   = typename TestTupleType::second_type;
    using fixture_type  = boost::numeric::ublas::tuple_fixture_tensor_static<TestTupleType>;
    
    
    BOOST_TEST_CONTEXT("[Static Tensor Construction] default constructor"){
        using tensor_t      = ublas::tensor_static<value_type,ublas::extents<1,1,1>,layout_type>;
        auto const t = tensor_t{};
        BOOST_REQUIRE_EQUAL ( t.size(), 1ul);
        BOOST_REQUIRE_EQUAL ( t.rank(), 3ul);
        BOOST_REQUIRE       (!t.empty());
        
        // TODO: Enable afte adding bounds checking
        // BOOST_CHECK_NO_THROW( (void)t.at(0));
        // BOOST_CHECK_THROW   ( (void)t.at(1), std::out_of_range);
    }

    BOOST_TEST_CONTEXT("[Static Tensor Construction] construction using container"){
        using tensor_t          = typename fixture_type::t32_type;
        using shape_t           = typename tensor_t::extents_type;
        using strides_t         = typename tensor_t::strides_type;
        using container_type    = typename tensor_t::container_type;

        auto const c = container_type{
            value_type(1), value_type(2),
            value_type(3), value_type(4),
            value_type(5), value_type(6)
        };

        auto const t = tensor_t(c);
        BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
        BOOST_REQUIRE       (!t.empty());
        
        // TODO: Enable afte adding bounds checking
        // BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        // BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        // BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == shape_t{} );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t), std::end(t), std::begin(c), std::end(c));
    }

    BOOST_TEST_CONTEXT("[Static Tensor Construction] construction using default value"){
        using tensor_t          = typename fixture_type::t32_type;
        using shape_t           = typename tensor_t::extents_type;
        using strides_t         = typename tensor_t::strides_type;

        auto const t = tensor_t(value_type(1));
        BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
        BOOST_REQUIRE       (!t.empty());

        // TODO: Enable afte adding bounds checking
        // BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        // BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        // BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == shape_t{} );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
        auto const one = value_type(1);
        for(auto i = 0ul; i < t.size(); ++i){
            BOOST_TEST_CHECKPOINT("[Default Value Constructed] testing tensor element(" << i << ") = " <<t[i]);
            BOOST_REQUIRE_EQUAL(t[i], one);
        }
    }

    // TODO: Enable after fixing the issue #116   
    // BOOST_TEST_CONTEXT("[Static Tensor Construction] construction using other layout"){
    //     using other_layout = std::conditional_t<
    //         std::is_same_v<ublas::layout::first_order, layout_type>,
    //         ublas::layout::last_order,
    //         ublas::layout::first_order
    //     >;

    //     auto const e = fixture_type::t32.extents();
    //     auto const o = ublas::tensor_static<value_type,other_layout>(e, value_type(1));
    //     BOOST_REQUIRE_EQUAL ( o.size(), 6ul);
    //     BOOST_REQUIRE_EQUAL ( o.rank(), 2ul);
    //     BOOST_REQUIRE       (!o.empty());
        
    //     auto const t = tensor_t(o);
    //     BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
    //     BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
    //     BOOST_REQUIRE       (!t.empty());

    // }

    // FIXME: Enable this tests after the tensor expression is fixed
    // BOOST_TEST_CONTEXT("[Static Tensor Construction] construction using tensor expression"){

    //     using tensor_t          = typename fixture_type::t32_type;
    //     using shape_t           = typename tensor_t::extents_type;
    //     using strides_t         = typename tensor_t::strides_type;

    //     auto a = tensor_t(value_type(1));

    //     BOOST_REQUIRE_EQUAL ( a.size(), 6ul);
    //     BOOST_REQUIRE_EQUAL ( a.rank(), 2ul);
    //     BOOST_REQUIRE       (!a.empty());
        
    //     auto const three = value_type(3);
    //     auto const expr = a + three * a;
    //     auto const t = tensor_t(expr);

    //     BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
    //     BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
    //     BOOST_REQUIRE       (!t.empty());
    //     // TODO: Enable it after adding out of bound check to static tensor
    //     // BOOST_CHECK_NO_THROW( (void)t.at(3ul));
    //     // BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
    //     // BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
    //     BOOST_REQUIRE ( t.extents() == shape_t{} );
    //     auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
    //     BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
    //     auto const four = value_type(4);
    //     for(auto i = 0ul; i < t.size(); ++i){
    //         BOOST_TEST_CHECKPOINT("[Tensor Expression Constructor] Testing tensor element(" << i << ") = " <<t[i]);
    //         BOOST_REQUIRE_EQUAL(t[i], four);
    //     }
    // }

    BOOST_TEST_CONTEXT("[Static Tensor Construction] construction using legacy ublas matrix"){
        
        using tensor_t          = typename fixture_type::t32_type;
        using shape_t           = typename tensor_t::extents_type;
        using strides_t         = typename tensor_t::strides_type;

        using matrix_type = typename tensor_t::matrix_type;
        auto const e = shape_t{};
        auto const a = matrix_type(e[0], e[1], value_type(1));
        BOOST_REQUIRE_EQUAL ( a.size1(), e[0]);
        BOOST_REQUIRE_EQUAL ( a.size2(), e[1]);

        auto const t = tensor_t(a);
        // TODO: Enable afte adding bounds checking
        // BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        // BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        // BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == shape_t{} );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
        auto const one = value_type(1);
        for(auto i = 0ul; i < t.size(); ++i){
            BOOST_TEST_CHECKPOINT("[Tensor Legacy uBLAS Matrix Constructor] testing tensor element(" << i << ") = " <<t[i]);
            BOOST_REQUIRE_EQUAL(t[i], one);
        }
    }

    // TODO: Enable after fixing the issue #116
    // BOOST_TEST_CONTEXT("[Static Tensor Construction] construction using legacy ublas vector"){
    //     using tensor_t          = typename fixture_type::t2_type;
    //     using shape_t           = typename tensor_t::extents_type;
    //     using strides_t         = typename tensor_t::strides_type;

    //     using vector_type = typename tensor_t::vector_type;
    //     auto const e = shape_t{};
    //     auto const a = vector_type(e[0], value_type(1));
    //     BOOST_REQUIRE_EQUAL ( a.size(), e[0]);

    //     auto const t = tensor_t(a);
    //     // TODO: Enable afte adding bounds checking
    //     // BOOST_CHECK_NO_THROW( (void)t.at(1ul));
    //     // BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        
    //     BOOST_REQUIRE ( t.extents() == fixture_type::t21.extents() );
    //     auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 1} : strides_t{1ul, 1ul});
    //     BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
    //     auto const one = value_type(1);
    //     for(auto i = 0ul; i < t.size(); ++i){
    //         BOOST_TEST_CHECKPOINT("[Tensor Legacy uBLAS Vector Constructor] testing tensor element(" << i << ") = " <<t[i]);
    //         BOOST_REQUIRE_EQUAL(t[i], one);
    //     }
    // }
    
    // FIXME: Enable this tests after the tensor expression is fixed
    // BOOST_TEST_CONTEXT("[Static Tensor Construction] construction using legacy ublas matrix expression"){
        
    //     using tensor_t          = typename fixture_type::t32_type;
    //     using shape_t           = typename tensor_t::extents_type;
    //     using strides_t         = typename tensor_t::strides_type;

    //     using matrix_type = typename tensor_t::matrix_type;
    //     auto const e = shape_t{};
    //     auto const a = matrix_type(e[0], e[1], value_type(1));
    //     BOOST_REQUIRE_EQUAL ( a.size1(), e[0]);
    //     BOOST_REQUIRE_EQUAL ( a.size2(), e[1]);

    //     auto const three = value_type(3);
    //     auto const expr = a + three * a;

    //     auto const t = tensor_t(expr);
    //     // TODO: Enable afte adding bounds checking
    //     // BOOST_CHECK_NO_THROW( (void)t.at(3ul));
    //     // BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
    //     // BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
    //     BOOST_REQUIRE ( t.extents() == shape_t{} );
    //     auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
    //     BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
    //     auto const four = value_type(4);
    //     for(auto i = 0ul; i < t.size(); ++i){
    //         BOOST_TEST_CHECKPOINT("[Tensor Legacy uBLAS Matrix Expression Constructor] testing tensor element(" << i << ") = " <<t[i]);
    //         BOOST_REQUIRE_EQUAL(t[i], four);
    //     }
    // }

    // TODO: Enable after fixing the issue #116
    // BOOST_TEST_CONTEXT("[Static Tensor Construction] construction using legacy ublas vector expression"){
        
    //     using tensor_t          = typename fixture_type::t2_type;
    //     using shape_t           = typename tensor_t::extents_type;
    //     using strides_t         = typename tensor_t::strides_type;

    //     using vector_type = typename tensor_t::vector_type;
    //     auto const e = shape_t{};
    //     auto const a = vector_type(e[0], value_type(1));
    //     BOOST_REQUIRE_EQUAL ( a.size(), e[0]);

    //     auto expr = a + value_type(3) * a;
    //     auto const t = tensor_t(expr);
    //     // TODO: Enable afte adding bounds checking
    //     // BOOST_CHECK_NO_THROW( (void)t.at(1ul));
    //     // BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        
    //     BOOST_REQUIRE ( t.extents() == shape_t{} );
    //     auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 1} : strides_t{1ul, 1ul});
    //     BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        
    //     auto const four = value_type(4);
    //     for(auto i = 0ul; i < t.size(); ++i){
    //         BOOST_TEST_CHECKPOINT("[Tensor Legacy uBLAS Vector Expression Constructor] testing tensor element(" << i << ") = " <<t[i]);
    //         BOOST_REQUIRE_EQUAL(t[i], four);
    //     }
    // }

    BOOST_TEST_CONTEXT("[Static Tensor Copy Construction] copy constructing tensor"){
        using tensor_t          = typename fixture_type::t32_type;
        using strides_t         = typename tensor_t::strides_type;

        auto o = fixture_type::t32;
        o = value_type(1);

        BOOST_REQUIRE_EQUAL ( o.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( o.rank(), 2ul);
        BOOST_REQUIRE       (!o.empty());

        auto const t = tensor_t(o);
        BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
        BOOST_REQUIRE       (!t.empty());
        // TODO: Enable afte adding bounds checking
        // BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        // BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        // BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == o.extents() );
        auto const strides = (std::is_same_v<layout_type,ublas::layout::first_order> ? strides_t{1, 3} : strides_t{2ul, 1ul});
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t.strides()), std::end(t.strides()), std::begin(strides), std::end(strides));
        BOOST_REQUIRE_EQUAL_COLLECTIONS(std::begin(t), std::end(t), std::begin(o), std::end(o));
    }

    BOOST_TEST_CONTEXT("[Static Tensor Move Construction] move constructing tensor"){
        using tensor_t          = typename fixture_type::t32_type;
        using strides_t         = typename tensor_t::strides_type;

        auto o = tensor_t(value_type(1));

        BOOST_REQUIRE_EQUAL ( o.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( o.rank(), 2ul);
        BOOST_REQUIRE       (!o.empty());

        auto const t = tensor_t(std::move(o));
        BOOST_REQUIRE_EQUAL ( t.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( t.rank(), 2ul);
        BOOST_REQUIRE       (!t.empty());
        BOOST_REQUIRE_EQUAL ( o.size(), 6ul);
        BOOST_REQUIRE_EQUAL ( o.rank(), 2ul);
        BOOST_REQUIRE       (!o.empty());
        // TODO: Enable afte adding bounds checking
        // BOOST_CHECK_NO_THROW( (void)t.at(3ul));
        // BOOST_CHECK_THROW   ( (void)t.at(25ul),    std::out_of_range);
        // BOOST_CHECK_THROW   ( (void)t.at(5,2),     std::out_of_range);
        
        BOOST_REQUIRE ( t.extents() == o.extents() );
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
