//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//


#include <iostream>
#include <random>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/test/unit_test.hpp>

#include "utility.hpp"



BOOST_AUTO_TEST_SUITE ( test_tensor_static_matrix_interoperability )

using test_types = zip<int,float>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;


BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_matrix_copy_ctor, pair,  test_types)
{
    namespace ublas = boost::numeric::ublas;
    using value  = typename pair::first_type;
    using layout = typename pair::second_type;
    using tensor  = ublas::tensor_dynamic<value,layout>;
    using matrix = typename tensor::matrix_type;

    ublas::tensor_static<value, ublas::extents<1,1>,layout> a2 = matrix(1,1);
    BOOST_CHECK_EQUAL(  a2.size() , 1 );
    BOOST_CHECK( !a2.empty() );

    ublas::tensor_static<value, ublas::extents<2,1>,layout> a3 = matrix(2,1);
    BOOST_CHECK_EQUAL(  a3.size() , 2 );
    BOOST_CHECK( !a3.empty() );

    ublas::tensor_static<value, ublas::extents<1,2>,layout> a4 = matrix(1,2);
    BOOST_CHECK_EQUAL(  a4.size() , 2 );
    BOOST_CHECK( !a4.empty() );

    ublas::tensor_static<value, ublas::extents<2,3>,layout> a5 = matrix(2,3);
    BOOST_CHECK_EQUAL(  a5.size() , 6 );
    BOOST_CHECK( !a5.empty() );
}


BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_vector_copy_ctor, pair,  test_types)
{
    namespace ublas = boost::numeric::ublas;
    using value  = typename pair::first_type;
    using layout = typename pair::second_type;
    using tensor  = ublas::tensor_dynamic<value,layout>;
    using vector = typename tensor::vector_type;

    ublas::tensor_static<value, ublas::extents<1,1>,layout> a2 = vector(1);
    BOOST_CHECK_EQUAL(  a2.size() , 1 );
    BOOST_CHECK( !a2.empty() );

    ublas::tensor_static<value, ublas::extents<2,1>,layout> a3 = vector(2);
    BOOST_CHECK_EQUAL(  a3.size() , 2 );
    BOOST_CHECK( !a3.empty() );

    ublas::tensor_static<value, ublas::extents<2,1>,layout> a4 = vector(2);
    BOOST_CHECK_EQUAL(  a4.size() , 2 );
    BOOST_CHECK( !a4.empty() );

    ublas::tensor_static<value, ublas::extents<3,1>,layout> a5 = vector(3);
    BOOST_CHECK_EQUAL(  a5.size() , 3 );
    BOOST_CHECK( !a5.empty() );
}


struct fixture
{
    template<size_t... N>
    using shape = boost::numeric::ublas::extents<N...>;

    fixture()=default;

    std::tuple<
        shape<1,1>, // 0
        shape<2,3>, // 1
        shape<5,8>, // 2
        shape<9,7>  // 3
    > extents;
};




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_matrix_copy_ctor_extents, pair,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value  = typename pair::first_type;
    using layout = typename pair::second_type;
    using tensor  = ublas::tensor_dynamic<value,layout>;
    using matrix = typename tensor::matrix_type;

    auto check = [](auto const& /*unused*/, auto& e) {
        using shape = std::decay_t<decltype(e)>;
        using etensor = ublas::tensor_static<value, shape,layout>;

        assert(ublas::size(e)==2);
        etensor t = matrix{e[0],e[1]};
        BOOST_CHECK_EQUAL (  t.size() , ublas::product(e) );
        BOOST_CHECK_EQUAL (  t.rank() , ublas::size   (e) );
        BOOST_CHECK       ( !t.empty()    );
    };

    for_each_in_tuple(extents,check);
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_vector_copy_ctor_extents, pair,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value  = typename pair::first_type;
    using layout = typename pair::second_type;
    using tensor  = ublas::tensor_dynamic<value,layout>;
    using vector = typename tensor::vector_type;

    auto check = [](auto const& /*unused*/, auto& e) {
        using shape = std::decay_t<decltype(e)>;
        using etensor = ublas::tensor_static<value, shape,layout>;

        if constexpr( ublas::get_v<shape,1> == 1 ){
            assert(ublas::size(e)==2);
            if(ublas::empty(e))
                return;

            etensor t = vector(ublas::product(e));
            BOOST_CHECK_EQUAL (  t.size() , ublas::product(e) );
            BOOST_CHECK_EQUAL (  t.rank() , ublas::size   (e) );
            BOOST_CHECK       ( !t.empty()    );
        }

    };

    for_each_in_tuple(extents,check);
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_matrix_copy_assignment, pair,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value  = typename pair::first_type;
    using layout = typename pair::second_type;
    using tensor  = ublas::tensor_dynamic<value,layout>;
    using matrix = typename tensor::matrix_type;

    auto check = [](auto const& /*unused*/, auto& e) {
        using shape = std::decay_t<decltype(e)>;
        using etensor = ublas::tensor_static<value, shape,layout>;

        assert(ublas::size(e) == 2);
        auto t = etensor{};
        auto r = matrix(e[0],e[1]);
        std::iota(r.data().begin(),r.data().end(), 1);
        t = r;

        BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0) );
        BOOST_CHECK_EQUAL (  t.extents().at(1) , e.at(1) );
        BOOST_CHECK_EQUAL (  t.size() , ublas::product(e) );
        BOOST_CHECK_EQUAL (  t.rank() , ublas::size   (e) );
        BOOST_CHECK       ( !t.empty()    );

        for(auto j = 0ul; j < t.size(1); ++j){
            for(auto i = 0ul; i < t.size(0); ++i){
                BOOST_CHECK_EQUAL( t.at(i,j), r(i,j)  );
            }
        }
    };

    for_each_in_tuple(extents,check);
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_vector_copy_assignment, pair,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value  = typename pair::first_type;
    using layout = typename pair::second_type;
    using tensor  = ublas::tensor_dynamic<value,layout>;
    using vector = typename tensor::vector_type;

    auto check = [](auto const& /*unused*/, auto& e) {
        using shape = std::decay_t<decltype(e)>;
        using etensor = ublas::tensor_static<value, shape,layout>;

        assert(ublas::size(e) == 2);

        if constexpr( ublas::get_v<shape,1> == 1 ){
            auto t = etensor{};
            auto r = vector(e[0]*e[1]);
            std::iota(r.data().begin(),r.data().end(), 1);
            t = r;

            BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0)*e.at(1) );
            BOOST_CHECK_EQUAL (  t.extents().at(1) , 1);
            BOOST_CHECK_EQUAL (  t.size() , ublas::product (e) );
            BOOST_CHECK_EQUAL (  t.rank() , ublas::size    (e) );
            BOOST_CHECK       ( !t.empty()    );

            for(auto i = 0ul; i < t.size(); ++i){
                BOOST_CHECK_EQUAL( t[i], r(i)  );
            }
        }
    };

    for_each_in_tuple(extents,check);
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_matrix_move_assignment, pair,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value  = typename pair::first_type;
    using layout = typename pair::second_type;
    using tensor  = ublas::tensor_dynamic<value,layout>;
    using matrix = typename tensor::matrix_type;

    auto check = [](auto const& /*unused*/, auto& e) {
        using shape = std::decay_t<decltype(e)>;
        using etensor = ublas::tensor_static<value, shape,layout>;

        assert(ublas::size(e) == 2);
        auto t = etensor{};
        auto r = matrix(e[0],e[1]);
        std::iota(r.data().begin(),r.data().end(), 1);
        auto q = r;
        t = std::move(r);

        BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0) );
        BOOST_CHECK_EQUAL (  t.extents().at(1) , e.at(1) );
        BOOST_CHECK_EQUAL (  t.size() , ublas::product(e) );
        BOOST_CHECK_EQUAL (  t.rank() , ublas::size   (e) );
        BOOST_CHECK       ( !t.empty()    );

        for(auto j = 0ul; j < t.size(1); ++j){
            for(auto i = 0ul; i < t.size(0); ++i){
                BOOST_CHECK_EQUAL( t.at(i,j), q(i,j)  );
            }
        }
    };

    for_each_in_tuple(extents,check);
}




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_vector_move_assignment, pair,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value  = typename pair::first_type;
    using layout = typename pair::second_type;
    using tensor  = ublas::tensor_dynamic<value,layout>;
    using vector = typename tensor::vector_type;

    auto check = [](auto const& /*unused*/, auto& e) {
        using shape = std::decay_t<decltype(e)>;
        using etensor = ublas::tensor_static<value, shape,layout>;

        assert(ublas::size(e) == 2);
        if constexpr( ublas::get_v<shape,1> == 1 ){
            auto t = etensor{};
            auto r = vector(e[0]*e[1]);
            std::iota(r.data().begin(),r.data().end(), 1);
            auto q = r;
            t = std::move(r);

            BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0) * e.at(1));
            BOOST_CHECK_EQUAL (  t.extents().at(1) , 1);
            BOOST_CHECK_EQUAL (  t.size() , ublas::product(e) );
            BOOST_CHECK_EQUAL (  t.rank() , ublas::size   (e) );
            BOOST_CHECK       ( !t.empty()    );

            for(auto i = 0ul; i < t.size(); ++i){
                BOOST_CHECK_EQUAL( t[i], q(i)  );
            }
        }
    };

    for_each_in_tuple(extents,check);
}





BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_matrix_expressions, pair,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value  = typename pair::first_type;
    using layout = typename pair::second_type;
    using tensor  = ublas::tensor_dynamic<value,layout>;
    using matrix = typename tensor::matrix_type;

    for_each_in_tuple(extents,[](auto const& /*unused*/, auto& e) {
        using shape = std::decay_t<decltype(e)>;
        using etensor = ublas::tensor_static<value, shape,layout>;

        assert(ublas::size(e) == 2);
        auto t = etensor{};
        auto r = matrix(e[0],e[1]);
        std::iota(r.data().begin(),r.data().end(), 1);
        t = r + 3*r;
        etensor s = r + 3*r;
        etensor q = s + r + 3*r + s; // + 3*r


        BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0) );
        BOOST_CHECK_EQUAL (  t.extents().at(1) , e.at(1) );
        BOOST_CHECK_EQUAL (  t.size() , ublas::product(e) );
        BOOST_CHECK_EQUAL (  t.rank() , ublas::size   (e) );
        BOOST_CHECK       ( !t.empty()    );

        BOOST_CHECK_EQUAL (  s.extents().at(0) , e.at(0) );
        BOOST_CHECK_EQUAL (  s.extents().at(1) , e.at(1) );
        BOOST_CHECK_EQUAL (  s.size() , ublas::product(e) );
        BOOST_CHECK_EQUAL (  s.rank() , ublas::size   (e) );
        BOOST_CHECK       ( !s.empty()    );

        BOOST_CHECK_EQUAL (  q.extents().at(0) , e.at(0) );
        BOOST_CHECK_EQUAL (  q.extents().at(1) , e.at(1) );
        BOOST_CHECK_EQUAL (  q.size() , ublas::product(e) );
        BOOST_CHECK_EQUAL (  q.rank() , ublas::size   (e) );
        BOOST_CHECK       ( !q.empty()    );


        for(auto j = 0ul; j < t.size(1); ++j){
            for(auto i = 0ul; i < t.size(0); ++i){
                BOOST_CHECK_EQUAL( t.at(i,j), 4*r(i,j)  );
                BOOST_CHECK_EQUAL( s.at(i,j), t.at(i,j)  );
                BOOST_CHECK_EQUAL( q.at(i,j), 3*s.at(i,j)  );
            }
        }
    });
}






BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_vector_expressions, pair,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value  = typename pair::first_type;
    using layout = typename pair::second_type;
    using tensor  = ublas::tensor_dynamic<value,layout>;
    using vector = typename tensor::vector_type;

    for_each_in_tuple(extents,[](auto const& /*unused*/, auto& e) {
        using shape = std::decay_t<decltype(e)>;
        using etensor = ublas::tensor_static<value, shape,layout>;

        assert(ublas::size(e) == 2);
        if constexpr( ublas::get_v<shape,1> == 1 ){
            auto t = etensor{};
            auto r = vector(e[0]*e[1]);
            std::iota(r.data().begin(),r.data().end(), value{1});
            t = r + 3*r;
            etensor s = r + 3*r;
            etensor q = s + r + 3*r + s; // + 3*r


            BOOST_CHECK_EQUAL (  t.extents().at(0) , e.at(0)*e.at(1) );
            BOOST_CHECK_EQUAL (  t.extents().at(1) , 1);
            BOOST_CHECK_EQUAL (  t.size() , ublas::product(e) );
            BOOST_CHECK_EQUAL (  t.rank() , ublas::size   (e) );
            BOOST_CHECK       ( !t.empty()    );

            BOOST_CHECK_EQUAL (  s.extents().at(0) , e.at(0)*e.at(1) );
            BOOST_CHECK_EQUAL (  s.extents().at(1) , 1);
            BOOST_CHECK_EQUAL (  s.size() , ublas::product(e) );
            BOOST_CHECK_EQUAL (  s.rank() , ublas::size   (e) );
            BOOST_CHECK       ( !s.empty()    );

            BOOST_CHECK_EQUAL (  q.extents().at(0) , e.at(0)*e.at(1) );
            BOOST_CHECK_EQUAL (  q.extents().at(1) , 1);
            BOOST_CHECK_EQUAL (  q.size() , ublas::product(e) );
            BOOST_CHECK_EQUAL (  q.rank() , ublas::size   (e) );
            BOOST_CHECK       ( !q.empty()    );



            for(auto i = 0ul; i < t.size(); ++i){
                BOOST_CHECK_EQUAL( t.at(i), 4*r(i)  );
                BOOST_CHECK_EQUAL( s.at(i), t.at(i)  );
                BOOST_CHECK_EQUAL( q.at(i), 3*s.at(i)  );
            }
        }
    });
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_matrix_vector_expressions, pair,  test_types, fixture )
{
    namespace ublas = boost::numeric::ublas;
    using value  = typename pair::first_type;
    using layout = typename pair::second_type;
    using tensor  = ublas::tensor_dynamic<value,layout>;
    using matrix = typename tensor::matrix_type;
    using vector = typename tensor::vector_type;

    for_each_in_tuple(extents,[](auto const& /*unused*/, auto& e) {
        using shape = std::decay_t<decltype(e)>;

        if constexpr(ublas::product_v<shape> <= 2)
            return;

        assert(ublas::size_v<shape> == 2);

        auto Q = ublas::tensor_static<value, ublas::extents<ublas::get_v<shape,0>,1>,layout>{} ;
        auto A = matrix(e[0],e[1]);
        auto b = vector(e[1]);
        auto c = vector(e[0]);
        std::iota(b.data().begin(),b.data().end(), value{1});
        std::fill(A.data().begin(),A.data().end(), value{1});
        std::fill(c.data().begin(),c.data().end(), value{2});
        std::fill(Q.begin(),Q.end(), 2);

        decltype(Q) T = Q + (ublas::prod(A , b) + 2*c) + 3*Q;

        BOOST_CHECK_EQUAL (  T.extents().at(0) , Q.extents().at(0) );
        BOOST_CHECK_EQUAL (  T.extents().at(1) , Q.extents().at(1));
        BOOST_CHECK_EQUAL (  T.size() , Q.size() );
        BOOST_CHECK_EQUAL (  T.size() , c.size() );
        BOOST_CHECK_EQUAL (  T.rank() , Q.rank() );
        BOOST_CHECK       ( !T.empty()    );

        for(auto i = size_t{0}; i < T.size(); ++i){
            auto n = e[1];
            auto ab = value(n*(n+1)/2);
            BOOST_CHECK_EQUAL( T(i), ab+4*Q(0)+2*c(0)  );
        }

    });
}


BOOST_AUTO_TEST_SUITE_END()
