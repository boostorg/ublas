//  Copyright (c) 2018-2019 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer and Google in producing this work
//  which started as a Google Summer of Code project.
//

#include <iostream>
#include <algorithm>
#include <complex>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/numeric/ublas/tensor/multi_index.hpp>


#include <boost/test/unit_test.hpp>

#include "utility.hpp"


BOOST_AUTO_TEST_SUITE ( test_multi_index )


using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;


BOOST_AUTO_TEST_CASE ( test_index_classes )
{
    namespace i = boost::numeric::ublas::index;


    BOOST_CHECK_EQUAL (  i::_a.value  ,  1  ) ;
    BOOST_CHECK_EQUAL (  i::_b.value  ,  2  ) ;
    BOOST_CHECK_EQUAL (  i::_c.value  ,  3  ) ;
    BOOST_CHECK_EQUAL (  i::_d.value  ,  4  ) ;
    BOOST_CHECK_EQUAL (  i::_e.value  ,  5  ) ;
    BOOST_CHECK_EQUAL (  i::_f.value  ,  6  ) ;
    BOOST_CHECK_EQUAL (  i::_g.value  ,  7  ) ;
    BOOST_CHECK_EQUAL (  i::_h.value  ,  8  ) ;
    BOOST_CHECK_EQUAL (  i::_i.value  ,  9  ) ;
    BOOST_CHECK_EQUAL (  i::_j.value  , 10  ) ;
    BOOST_CHECK_EQUAL (  i::_k.value  , 11  ) ;
    BOOST_CHECK_EQUAL (  i::_l.value  , 12  ) ;
    BOOST_CHECK_EQUAL (  i::_m.value  , 13  ) ;
    BOOST_CHECK_EQUAL (  i::_n.value  , 14  ) ;
    BOOST_CHECK_EQUAL (  i::_o.value  , 15  ) ;
    BOOST_CHECK_EQUAL (  i::_p.value  , 16  ) ;
    BOOST_CHECK_EQUAL (  i::_q.value  , 17  ) ;
    BOOST_CHECK_EQUAL (  i::_r.value  , 18  ) ;
    BOOST_CHECK_EQUAL (  i::_s.value  , 19  ) ;
    BOOST_CHECK_EQUAL (  i::_t.value  , 20  ) ;
    BOOST_CHECK_EQUAL (  i::_u.value  , 21  ) ;
    BOOST_CHECK_EQUAL (  i::_v.value  , 22  ) ;
    BOOST_CHECK_EQUAL (  i::_w.value  , 23  ) ;
    BOOST_CHECK_EQUAL (  i::_x.value  , 24  ) ;
    BOOST_CHECK_EQUAL (  i::_y.value  , 25  ) ;
    BOOST_CHECK_EQUAL (  i::_z.value  , 26  ) ;

}

BOOST_AUTO_TEST_CASE ( test_multi_index_class_construction )
{
    namespace ublas = boost::numeric::ublas;
    namespace i = ublas::index;


    {
      ublas::multi_index<2> ind(i::_a, i::_b);

        BOOST_CHECK_EQUAL ( get<0>( ind ), 1 ) ;
        BOOST_CHECK_EQUAL ( get<1>( ind ), 2 ) ;
    }


    {
        ublas::multi_index<2> ind(i::_d,i::_c);

        BOOST_CHECK_EQUAL ( ind[0] , 4 ) ;
        BOOST_CHECK_EQUAL ( ind[1] , 3 ) ;
    }
}


BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_multi_index_class_generation, value,  test_types )
{
    namespace ublas = boost::numeric::ublas;
    namespace i = ublas::index;
    using value_type   = typename value::first_type;
    using layout_type  = typename value::second_type;
    using tensor_type  = ublas::tensor_dynamic<value_type,layout_type>;

    auto t = std::make_tuple (
          i::_a, // 0
          i::_b, // 1
          i::_c, // 2
          i::_d, // 3
          i::_e  // 4
          );

    {
        auto a = tensor_type(ublas::extents<>{2,3}, value_type{2});
        auto a_ind = a( std::get<0>(t), std::get<2>(t)  );

        BOOST_CHECK_EQUAL ( std::addressof( a_ind.first ), std::addressof( a ) ) ;

        BOOST_CHECK_EQUAL (std::get<0>(a_ind.second)(), i::_a() ) ;
        BOOST_CHECK_EQUAL (std::get<1>(a_ind.second)(), i::_c() ) ;
    }

    {
        auto a = tensor_type(ublas::extents<>{2,3}, value_type{2});
        auto a_ind = a( std::get<2>(t), std::get<0>(t)  );

        BOOST_CHECK_EQUAL ( std::addressof( a_ind.first ), std::addressof( a ) ) ;

        BOOST_CHECK_EQUAL (std::get<0>(a_ind.second)(), i::_c() ) ;
        BOOST_CHECK_EQUAL (std::get<1>(a_ind.second)(), i::_a() ) ;
    }

    {
        auto a = tensor_type(ublas::extents<>{2,3}, value_type{2});
        auto a_ind = a( std::get<2>(t), std::get<3>(t)  );

        BOOST_CHECK_EQUAL (std::addressof(  a_ind.first ), std::addressof( a ) ) ;

        BOOST_CHECK_EQUAL (std::get<0>(a_ind.second)(), i::_c() ) ;
        BOOST_CHECK_EQUAL (std::get<1>(a_ind.second)(), i::_d() ) ;
    }

    {
        auto a = tensor_type(ublas::extents<>{2,3,4}, value_type{2});
        auto a_ind = a( std::get<2>(t), std::get<3>(t), std::get<0>(t)  );

        BOOST_CHECK_EQUAL (std::addressof(  a_ind.first ), std::addressof( a ) ) ;

        BOOST_CHECK_EQUAL (std::get<0>(a_ind.second)(), i::_c() ) ;
        BOOST_CHECK_EQUAL (std::get<1>(a_ind.second)(), i::_d() ) ;
        BOOST_CHECK_EQUAL (std::get<2>(a_ind.second)(), i::_a() ) ;
    }

}

BOOST_AUTO_TEST_SUITE_END()
