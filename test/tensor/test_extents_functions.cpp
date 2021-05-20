//
// 	Copyright (c) 2020, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//


#include <vector>
#include <array>

#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>



BOOST_AUTO_TEST_SUITE(test_shape_functions)

struct fixture_extents_dynamic_rank
{
  using shape_t = boost::numeric::ublas::extents<>;

  static inline auto n     = shape_t{};
  static inline auto n1    = shape_t{1};
  static inline auto n2    = shape_t{2};
  static inline auto n11   = shape_t{1,1};
  static inline auto n12   = shape_t{1,2};
  static inline auto n21   = shape_t{2,1};
  static inline auto n22   = shape_t{2,2};
  static inline auto n32   = shape_t{3,2};
  static inline auto n111  = shape_t{1,1,1};
  static inline auto n211  = shape_t{2,1,1};
  static inline auto n121  = shape_t{1,2,1};
  static inline auto n112  = shape_t{1,1,2};
  static inline auto n123  = shape_t{1,2,3};
  static inline auto n321  = shape_t{3,2,1};
  static inline auto n213  = shape_t{2,1,3};
  static inline auto n432  = shape_t{4,3,2};
};

struct fixture_extents_static_rank
{
  template<std::size_t N>
  using shape_t = boost::numeric::ublas::extents<N>;

  static inline auto n     = shape_t<0>{};
  static inline auto n1    = shape_t<1>{1};
  static inline auto n2    = shape_t<1>{2};
  static inline auto n11   = shape_t<2>{{1,1}};
  static inline auto n12   = shape_t<2>{{1,2}};
  static inline auto n21   = shape_t<2>{{2,1}};
  static inline auto n22   = shape_t<2>{{2,2}};
  static inline auto n32   = shape_t<2>{{3,2}};
  static inline auto n111  = shape_t<3>{{1,1,1}};
  static inline auto n211  = shape_t<3>{{2,1,1}};
  static inline auto n121  = shape_t<3>{{1,2,1}};
  static inline auto n112  = shape_t<3>{{1,1,2}};
  static inline auto n123  = shape_t<3>{{1,2,3}};
  static inline auto n321  = shape_t<3>{{3,2,1}};
  static inline auto n213  = shape_t<3>{{2,1,3}};
  static inline auto n432  = shape_t<3>{{4,3,2}};
};



struct fixture_extents_static
{
  template<std::size_t ... ns>
  using extents_static = boost::numeric::ublas::extents<ns...>;

  static inline auto n     = extents_static<>      {};
  static inline auto n1    = extents_static<1>     {};
  static inline auto n2    = extents_static<2>     {};
  static inline auto n11   = extents_static<1,1>   {};
  static inline auto n12   = extents_static<1,2>   {};
  static inline auto n21   = extents_static<2,1>   {};
  static inline auto n22   = extents_static<2,2>   {};
  static inline auto n32   = extents_static<3,2>   {};
  static inline auto n111  = extents_static<1,1,1> {};
  static inline auto n211  = extents_static<2,1,1> {};
  static inline auto n121  = extents_static<1,2,1> {};
  static inline auto n112  = extents_static<1,1,2> {};
  static inline auto n123  = extents_static<1,2,3> {};
  static inline auto n321  = extents_static<3,2,1> {};
  static inline auto n213  = extents_static<2,1,3> {};
  static inline auto n432  = extents_static<4,3,2> {};
};




BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_rank_is_scalar,
                        fixture_extents_dynamic_rank,
                        *boost::unit_test::label("extents_dynamic_rank")
                          *boost::unit_test::label("is_scalar"))
{

  namespace ub = boost::numeric::ublas;  
  BOOST_CHECK ( !ub::is_scalar( n    ));
  BOOST_CHECK (  ub::is_scalar( n1   ));
  BOOST_CHECK ( !ub::is_scalar( n2   ));
  BOOST_CHECK (  ub::is_scalar( n11  ));
  BOOST_CHECK ( !ub::is_scalar( n12  ));
  BOOST_CHECK ( !ub::is_scalar( n21  ));
  BOOST_CHECK ( !ub::is_scalar( n22  ));
  BOOST_CHECK ( !ub::is_scalar( n32  ));
  BOOST_CHECK (  ub::is_scalar( n111 ));
  BOOST_CHECK ( !ub::is_scalar( n211 ));
  BOOST_CHECK ( !ub::is_scalar( n121 ));
  BOOST_CHECK ( !ub::is_scalar( n112 ));
  BOOST_CHECK ( !ub::is_scalar( n123 ));
  BOOST_CHECK ( !ub::is_scalar( n321 ));
  BOOST_CHECK ( !ub::is_scalar( n213 ));
  BOOST_CHECK ( !ub::is_scalar( n432 ));
}


BOOST_FIXTURE_TEST_CASE(test_extents_static_rank_is_scalar,
                        fixture_extents_static_rank,
                        *boost::unit_test::label("extents_static_rank")
                          *boost::unit_test::label("is_scalar"))
{

  namespace ub = boost::numeric::ublas;
  BOOST_CHECK ( !ub::is_scalar( n    ));
  BOOST_CHECK (  ub::is_scalar( n1   ));
  BOOST_CHECK ( !ub::is_scalar( n2   ));
  BOOST_CHECK (  ub::is_scalar( n11  ));
  BOOST_CHECK ( !ub::is_scalar( n12  ));
  BOOST_CHECK ( !ub::is_scalar( n21  ));
  BOOST_CHECK ( !ub::is_scalar( n22  ));
  BOOST_CHECK ( !ub::is_scalar( n32  ));
  BOOST_CHECK (  ub::is_scalar( n111 ));
  BOOST_CHECK ( !ub::is_scalar( n211 ));
  BOOST_CHECK ( !ub::is_scalar( n121 ));
  BOOST_CHECK ( !ub::is_scalar( n112 ));
  BOOST_CHECK ( !ub::is_scalar( n123 ));
  BOOST_CHECK ( !ub::is_scalar( n321 ));
  BOOST_CHECK ( !ub::is_scalar( n213 ));
  BOOST_CHECK ( !ub::is_scalar( n432 ));

}


BOOST_FIXTURE_TEST_CASE(test_extents_static_is_scalar,
                        fixture_extents_static,
                        *boost::unit_test::label("extents_static")
                          *boost::unit_test::label("is_scalar"))
{

  namespace ub = boost::numeric::ublas;
  BOOST_CHECK ( !ub::is_scalar( n    ));
//FIXME:  BOOST_CHECK (  ub::is_scalar( n1   ));
  BOOST_CHECK ( !ub::is_scalar( n2   ));
  BOOST_CHECK (  ub::is_scalar( n11  ));
  BOOST_CHECK ( !ub::is_scalar( n12  ));
  BOOST_CHECK ( !ub::is_scalar( n21  ));
  BOOST_CHECK ( !ub::is_scalar( n22  ));
  BOOST_CHECK ( !ub::is_scalar( n32  ));
  BOOST_CHECK (  ub::is_scalar( n111 ));
  BOOST_CHECK ( !ub::is_scalar( n211 ));
  BOOST_CHECK ( !ub::is_scalar( n121 ));
  BOOST_CHECK ( !ub::is_scalar( n112 ));
  BOOST_CHECK ( !ub::is_scalar( n123 ));
  BOOST_CHECK ( !ub::is_scalar( n321 ));
  BOOST_CHECK ( !ub::is_scalar( n213 ));
  BOOST_CHECK ( !ub::is_scalar( n432 ));

}


BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_rank_is_vector,
                        fixture_extents_dynamic_rank,
                        *boost::unit_test::label("extents_dynamic_rank")
                          *boost::unit_test::label("is_vector"))
{

  namespace ub = boost::numeric::ublas;
  BOOST_CHECK ( !ub::is_vector( n    ));
  BOOST_CHECK (  ub::is_vector( n1   ));
  BOOST_CHECK (  ub::is_vector( n2   ));
  BOOST_CHECK (  ub::is_vector( n11  ));
  BOOST_CHECK (  ub::is_vector( n12  ));
  BOOST_CHECK (  ub::is_vector( n21  ));
  BOOST_CHECK ( !ub::is_vector( n22  ));
  BOOST_CHECK ( !ub::is_vector( n32  ));
  BOOST_CHECK (  ub::is_vector( n111 ));
  BOOST_CHECK (  ub::is_vector( n211 ));
  BOOST_CHECK (  ub::is_vector( n121 ));
  BOOST_CHECK ( !ub::is_vector( n112 ));
  BOOST_CHECK ( !ub::is_vector( n123 ));
  BOOST_CHECK ( !ub::is_vector( n321 ));
  BOOST_CHECK ( !ub::is_vector( n213 ));
  BOOST_CHECK ( !ub::is_vector( n432 ));
}

BOOST_FIXTURE_TEST_CASE(test_extents_static_rank_is_vector,
                        fixture_extents_static_rank,
                        *boost::unit_test::label("extents_static_rank")
                          *boost::unit_test::label("is_vector"))
{

  namespace ub = boost::numeric::ublas;
  BOOST_CHECK ( !ub::is_vector( n    ));
  BOOST_CHECK (  ub::is_vector( n1   ));
  BOOST_CHECK (  ub::is_vector( n2   ));
  BOOST_CHECK (  ub::is_vector( n11  ));
  BOOST_CHECK (  ub::is_vector( n12  ));
  BOOST_CHECK (  ub::is_vector( n21  ));
  BOOST_CHECK ( !ub::is_vector( n22  ));
  BOOST_CHECK ( !ub::is_vector( n32  ));
  BOOST_CHECK (  ub::is_vector( n111 ));
  BOOST_CHECK (  ub::is_vector( n211 ));
  BOOST_CHECK (  ub::is_vector( n121 ));
  BOOST_CHECK ( !ub::is_vector( n112 ));
  BOOST_CHECK ( !ub::is_vector( n123 ));
  BOOST_CHECK ( !ub::is_vector( n321 ));
  BOOST_CHECK ( !ub::is_vector( n213 ));
  BOOST_CHECK ( !ub::is_vector( n432 ));
}


BOOST_FIXTURE_TEST_CASE(test_extents_static_is_vector,
                        fixture_extents_static,
                        *boost::unit_test::label("extents_static")
                          *boost::unit_test::label("is_vector"))
{

  namespace ub = boost::numeric::ublas;
  BOOST_CHECK ( !ub::is_vector( n1   ));
//FIXME:  BOOST_CHECK (  ub::is_vector( n2   ));
  BOOST_CHECK (  ub::is_vector( n11  ));
  BOOST_CHECK (  ub::is_vector( n12  ));
  BOOST_CHECK (  ub::is_vector( n21  ));
  BOOST_CHECK ( !ub::is_vector( n22  ));
  BOOST_CHECK ( !ub::is_vector( n32  ));
  BOOST_CHECK (  ub::is_vector( n111 ));
  BOOST_CHECK (  ub::is_vector( n211 ));
  BOOST_CHECK (  ub::is_vector( n121 ));
  BOOST_CHECK ( !ub::is_vector( n112 ));
  BOOST_CHECK ( !ub::is_vector( n123 ));
  BOOST_CHECK ( !ub::is_vector( n321 ));
  BOOST_CHECK ( !ub::is_vector( n213 ));
  BOOST_CHECK ( !ub::is_vector( n432 ));
}

BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_rank_is_matrix,
                        fixture_extents_dynamic_rank,
                        *boost::unit_test::label("extents_dynamic_rank")
                          *boost::unit_test::label("is_matrix"))
{

  namespace ub = boost::numeric::ublas;
  BOOST_CHECK ( !ub::is_matrix( n    ));
  BOOST_CHECK (  ub::is_matrix( n1   ));
  BOOST_CHECK (  ub::is_matrix( n2   ));
  BOOST_CHECK (  ub::is_matrix( n11  ));
  BOOST_CHECK (  ub::is_matrix( n12  ));
  BOOST_CHECK (  ub::is_matrix( n21  ));
  BOOST_CHECK (  ub::is_matrix( n22  ));
  BOOST_CHECK (  ub::is_matrix( n32  ));
  BOOST_CHECK (  ub::is_matrix( n111 ));
  BOOST_CHECK (  ub::is_matrix( n211 ));
  BOOST_CHECK (  ub::is_matrix( n121 ));
  BOOST_CHECK ( !ub::is_matrix( n112 ));
  BOOST_CHECK ( !ub::is_matrix( n123 ));
  BOOST_CHECK (  ub::is_matrix( n321 ));
  BOOST_CHECK ( !ub::is_matrix( n213 ));
  BOOST_CHECK ( !ub::is_matrix( n432 ));
}

BOOST_FIXTURE_TEST_CASE(test_extents_static_rank_is_matrix,
                        fixture_extents_static_rank,
                        *boost::unit_test::label("extents_static_rank")
                          *boost::unit_test::label("is_matrix"))
{

  namespace ub = boost::numeric::ublas;
  BOOST_CHECK ( !ub::is_matrix( n    ));
  BOOST_CHECK (  ub::is_matrix( n1   ));
  BOOST_CHECK (  ub::is_matrix( n2   ));
  BOOST_CHECK (  ub::is_matrix( n11  ));
  BOOST_CHECK (  ub::is_matrix( n12  ));
  BOOST_CHECK (  ub::is_matrix( n21  ));
  BOOST_CHECK (  ub::is_matrix( n22  ));
  BOOST_CHECK (  ub::is_matrix( n32  ));
  BOOST_CHECK (  ub::is_matrix( n111 ));
  BOOST_CHECK (  ub::is_matrix( n211 ));
  BOOST_CHECK (  ub::is_matrix( n121 ));
  BOOST_CHECK ( !ub::is_matrix( n112 ));
  BOOST_CHECK ( !ub::is_matrix( n123 ));
  BOOST_CHECK (  ub::is_matrix( n321 ));
  BOOST_CHECK ( !ub::is_matrix( n213 ));
  BOOST_CHECK ( !ub::is_matrix( n432 ));
}


BOOST_FIXTURE_TEST_CASE(test_extents_static_is_matrix,
                        fixture_extents_static,
                        *boost::unit_test::label("extents_static")
                          *boost::unit_test::label("is_matrix"))
{

  namespace ub = boost::numeric::ublas;
  BOOST_CHECK ( !ub::is_matrix( n    ));
//FIXME:  BOOST_CHECK ( !ub::is_matrix( n1   ));
  BOOST_CHECK ( !ub::is_matrix( n2   ));
  BOOST_CHECK (  ub::is_matrix( n11  ));
  BOOST_CHECK (  ub::is_matrix( n12  ));
  BOOST_CHECK (  ub::is_matrix( n21  ));
  BOOST_CHECK (  ub::is_matrix( n22  ));
  BOOST_CHECK (  ub::is_matrix( n32  ));
  BOOST_CHECK (  ub::is_matrix( n111 ));
  BOOST_CHECK (  ub::is_matrix( n211 ));
  BOOST_CHECK (  ub::is_matrix( n121 ));
  BOOST_CHECK ( !ub::is_matrix( n112 ));
  BOOST_CHECK ( !ub::is_matrix( n123 ));
  BOOST_CHECK (  ub::is_matrix( n321 ));
  BOOST_CHECK ( !ub::is_matrix( n213 ));
  BOOST_CHECK ( !ub::is_matrix( n432 ));
}


BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_rank_is_tensor,
                        fixture_extents_dynamic_rank,
                        *boost::unit_test::label("extents_dynamic_rank")
                          *boost::unit_test::label("is_tensor"))
{

  namespace ub = boost::numeric::ublas;
  BOOST_CHECK ( !ub::is_tensor( n    ));
  BOOST_CHECK ( !ub::is_tensor( n1   ));
  BOOST_CHECK ( !ub::is_tensor( n2   ));
  BOOST_CHECK ( !ub::is_tensor( n11  ));
  BOOST_CHECK ( !ub::is_tensor( n12  ));
  BOOST_CHECK ( !ub::is_tensor( n21  ));
  BOOST_CHECK ( !ub::is_tensor( n22  ));
  BOOST_CHECK ( !ub::is_tensor( n32  ));
  BOOST_CHECK ( !ub::is_tensor( n111 ));
  BOOST_CHECK ( !ub::is_tensor( n211 ));
  BOOST_CHECK ( !ub::is_tensor( n121 ));
  BOOST_CHECK (  ub::is_tensor( n112 ));
  BOOST_CHECK (  ub::is_tensor( n123 ));
  BOOST_CHECK ( !ub::is_tensor( n321 ));
  BOOST_CHECK (  ub::is_tensor( n213 ));
  BOOST_CHECK (  ub::is_tensor( n432 ));
}


BOOST_FIXTURE_TEST_CASE(test_extents_static_rank_is_tensor,
                        fixture_extents_static_rank,
                        *boost::unit_test::label("extents_static_rank")
                          *boost::unit_test::label("is_tensor"))
{

  namespace ub = boost::numeric::ublas;
  BOOST_CHECK ( !ub::is_tensor( n    ));
  BOOST_CHECK ( !ub::is_tensor( n1   ));
  BOOST_CHECK ( !ub::is_tensor( n2   ));
  BOOST_CHECK ( !ub::is_tensor( n11  ));
  BOOST_CHECK ( !ub::is_tensor( n12  ));
  BOOST_CHECK ( !ub::is_tensor( n21  ));
  BOOST_CHECK ( !ub::is_tensor( n22  ));
  BOOST_CHECK ( !ub::is_tensor( n32  ));
  BOOST_CHECK ( !ub::is_tensor( n111 ));
  BOOST_CHECK ( !ub::is_tensor( n211 ));
  BOOST_CHECK ( !ub::is_tensor( n121 ));
  BOOST_CHECK (  ub::is_tensor( n112 ));
  BOOST_CHECK (  ub::is_tensor( n123 ));
  BOOST_CHECK ( !ub::is_tensor( n321 ));
  BOOST_CHECK (  ub::is_tensor( n213 ));
  BOOST_CHECK (  ub::is_tensor( n432 ));
}


BOOST_FIXTURE_TEST_CASE(test_extents_static_is_tensor,
                        fixture_extents_static,
                        *boost::unit_test::label("extents_static")
                          *boost::unit_test::label("is_tensor"))
{

  namespace ub = boost::numeric::ublas;
  BOOST_CHECK ( !ub::is_tensor( n    ));
//FIXME:  BOOST_CHECK ( !ub::is_tensor( n1   ));
  BOOST_CHECK ( !ub::is_tensor( n2   ));
  BOOST_CHECK ( !ub::is_tensor( n11  ));
  BOOST_CHECK ( !ub::is_tensor( n12  ));
  BOOST_CHECK ( !ub::is_tensor( n21  ));
  BOOST_CHECK ( !ub::is_tensor( n22  ));
  BOOST_CHECK ( !ub::is_tensor( n32  ));
  BOOST_CHECK ( !ub::is_tensor( n111 ));
  BOOST_CHECK ( !ub::is_tensor( n211 ));
  BOOST_CHECK ( !ub::is_tensor( n121 ));
  BOOST_CHECK (  ub::is_tensor( n112 ));
  BOOST_CHECK (  ub::is_tensor( n123 ));
  BOOST_CHECK ( !ub::is_tensor( n321 ));
  BOOST_CHECK (  ub::is_tensor( n213 ));
  BOOST_CHECK (  ub::is_tensor( n432 ));
}

BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_rank_is_valid,
                        fixture_extents_dynamic_rank,
                        *boost::unit_test::label("extents_dynamic_rank")
                          *boost::unit_test::label("is_valid"))
{

  namespace ub = boost::numeric::ublas;
  BOOST_CHECK (  ub::is_valid( n1   ));
  BOOST_CHECK (  ub::is_valid( n2   ));
  BOOST_CHECK (  ub::is_valid( n11  ));
  BOOST_CHECK (  ub::is_valid( n12  ));
  BOOST_CHECK (  ub::is_valid( n21  ));
  BOOST_CHECK (  ub::is_valid( n22  ));
  BOOST_CHECK (  ub::is_valid( n32  ));
  BOOST_CHECK (  ub::is_valid( n111 ));
  BOOST_CHECK (  ub::is_valid( n211 ));
  BOOST_CHECK (  ub::is_valid( n121 ));
  BOOST_CHECK (  ub::is_valid( n112 ));
  BOOST_CHECK (  ub::is_valid( n123 ));
  BOOST_CHECK (  ub::is_valid( n321 ));
  BOOST_CHECK (  ub::is_valid( n213 ));
  BOOST_CHECK (  ub::is_valid( n432 ));
}


BOOST_FIXTURE_TEST_CASE(test_extents_static_rank_is_valid,
                        fixture_extents_static_rank,
                        *boost::unit_test::label("extents_static_rank")
                          *boost::unit_test::label("is_valid"))
{

  namespace ub = boost::numeric::ublas;
  BOOST_CHECK (  ub::is_valid( n    ));
  BOOST_CHECK (  ub::is_valid( n1   ));
  BOOST_CHECK (  ub::is_valid( n2   ));
  BOOST_CHECK (  ub::is_valid( n11  ));
  BOOST_CHECK (  ub::is_valid( n12  ));
  BOOST_CHECK (  ub::is_valid( n21  ));
  BOOST_CHECK (  ub::is_valid( n22  ));
  BOOST_CHECK (  ub::is_valid( n32  ));
  BOOST_CHECK (  ub::is_valid( n111 ));
  BOOST_CHECK (  ub::is_valid( n211 ));
  BOOST_CHECK (  ub::is_valid( n121 ));
  BOOST_CHECK (  ub::is_valid( n112 ));
  BOOST_CHECK (  ub::is_valid( n123 ));
  BOOST_CHECK (  ub::is_valid( n321 ));
  BOOST_CHECK (  ub::is_valid( n213 ));
  BOOST_CHECK (  ub::is_valid( n432 ));
}


BOOST_FIXTURE_TEST_CASE(test_extents_static_is_valid,
                        fixture_extents_static,
                        *boost::unit_test::label("extents_static")
                          *boost::unit_test::label("is_valid"))
{

  namespace ub = boost::numeric::ublas;
  BOOST_CHECK (  ub::is_valid( n    ));
//FIXME:  BOOST_CHECK (  ub::is_valid( n1   ));
//FIXME:  BOOST_CHECK (  ub::is_valid( n2   ));
  BOOST_CHECK (  ub::is_valid( n11  ));
  BOOST_CHECK (  ub::is_valid( n12  ));
  BOOST_CHECK (  ub::is_valid( n21  ));
  BOOST_CHECK (  ub::is_valid( n22  ));
  BOOST_CHECK (  ub::is_valid( n32  ));
  BOOST_CHECK (  ub::is_valid( n111 ));
  BOOST_CHECK (  ub::is_valid( n211 ));
  BOOST_CHECK (  ub::is_valid( n121 ));
  BOOST_CHECK (  ub::is_valid( n112 ));
  BOOST_CHECK (  ub::is_valid( n123 ));
  BOOST_CHECK (  ub::is_valid( n321 ));
  BOOST_CHECK (  ub::is_valid( n213 ));
  BOOST_CHECK (  ub::is_valid( n432 ));
}


BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_rank_product,
                        fixture_extents_dynamic_rank,
                        *boost::unit_test::label("extents_dynamic_rank")
                          *boost::unit_test::label("product"))
{

  namespace ub = boost::numeric::ublas;
  BOOST_CHECK_EQUAL ( ub::product( n    ), 0U);
  BOOST_CHECK_EQUAL ( ub::product( n1   ), 1U);
  BOOST_CHECK_EQUAL ( ub::product( n2   ), 2U);
  BOOST_CHECK_EQUAL ( ub::product( n11  ), 1U);
  BOOST_CHECK_EQUAL ( ub::product( n12  ), 2U);
  BOOST_CHECK_EQUAL ( ub::product( n21  ), 2U);
  BOOST_CHECK_EQUAL ( ub::product( n22  ), 4U);
  BOOST_CHECK_EQUAL ( ub::product( n32  ), 6U);
  BOOST_CHECK_EQUAL ( ub::product( n111 ), 1U);
  BOOST_CHECK_EQUAL ( ub::product( n211 ), 2U);
  BOOST_CHECK_EQUAL ( ub::product( n121 ), 2U);
  BOOST_CHECK_EQUAL ( ub::product( n112 ), 2U);
  BOOST_CHECK_EQUAL ( ub::product( n123 ), 6U);
  BOOST_CHECK_EQUAL ( ub::product( n321 ), 6U);
  BOOST_CHECK_EQUAL ( ub::product( n213 ), 6U);
  BOOST_CHECK_EQUAL ( ub::product( n432 ),24U);
}


BOOST_FIXTURE_TEST_CASE(test_extents_static_rank_product,
                        fixture_extents_static_rank,
                        *boost::unit_test::label("extents_static_rank")
                          *boost::unit_test::label("product"))
{

  namespace ub = boost::numeric::ublas;
  BOOST_CHECK_EQUAL ( ub::product( n    ), 0U);
  BOOST_CHECK_EQUAL ( ub::product( n1   ), 1U);
  BOOST_CHECK_EQUAL ( ub::product( n2   ), 2U);
  BOOST_CHECK_EQUAL ( ub::product( n11  ), 1U);
  BOOST_CHECK_EQUAL ( ub::product( n12  ), 2U);
  BOOST_CHECK_EQUAL ( ub::product( n21  ), 2U);
  BOOST_CHECK_EQUAL ( ub::product( n22  ), 4U);
  BOOST_CHECK_EQUAL ( ub::product( n32  ), 6U);
  BOOST_CHECK_EQUAL ( ub::product( n111 ), 1U);
  BOOST_CHECK_EQUAL ( ub::product( n211 ), 2U);
  BOOST_CHECK_EQUAL ( ub::product( n121 ), 2U);
  BOOST_CHECK_EQUAL ( ub::product( n112 ), 2U);
  BOOST_CHECK_EQUAL ( ub::product( n123 ), 6U);
  BOOST_CHECK_EQUAL ( ub::product( n321 ), 6U);
  BOOST_CHECK_EQUAL ( ub::product( n213 ), 6U);
  BOOST_CHECK_EQUAL ( ub::product( n432 ),24U);
}


BOOST_FIXTURE_TEST_CASE(test_extents_static_product,
                        fixture_extents_static,
                        *boost::unit_test::label("extents_static")
                          *boost::unit_test::label("product"))
{

  namespace ub = boost::numeric::ublas;
  BOOST_CHECK_EQUAL ( ub::product( n    ), 0U);
//FIXME:  BOOST_CHECK_EQUAL ( ub::product( n1   ), 1U);
//FIXME:  BOOST_CHECK_EQUAL ( ub::product( n2   ), 2U);
  BOOST_CHECK_EQUAL ( ub::product( n11  ), 1U);
  BOOST_CHECK_EQUAL ( ub::product( n12  ), 2U);
  BOOST_CHECK_EQUAL ( ub::product( n21  ), 2U);
  BOOST_CHECK_EQUAL ( ub::product( n22  ), 4U);
  BOOST_CHECK_EQUAL ( ub::product( n32  ), 6U);
  BOOST_CHECK_EQUAL ( ub::product( n111 ), 1U);
  BOOST_CHECK_EQUAL ( ub::product( n211 ), 2U);
  BOOST_CHECK_EQUAL ( ub::product( n121 ), 2U);
  BOOST_CHECK_EQUAL ( ub::product( n112 ), 2U);
  BOOST_CHECK_EQUAL ( ub::product( n123 ), 6U);
  BOOST_CHECK_EQUAL ( ub::product( n321 ), 6U);
  BOOST_CHECK_EQUAL ( ub::product( n213 ), 6U);
  BOOST_CHECK_EQUAL ( ub::product( n432 ),24U);
}


BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_rank_equal,
                        fixture_extents_dynamic_rank,
                        *boost::unit_test::label("extents_dynamic_rank")
                          *boost::unit_test::label("equal"))
{
  BOOST_CHECK (  n   == n   );
  BOOST_CHECK (  n1  == n1  );
  BOOST_CHECK (  n2  == n2  );
  BOOST_CHECK (  n11 == n11 );
  BOOST_CHECK (  n12 == n12 );
  BOOST_CHECK (  n21 == n21 );
  BOOST_CHECK (  n22 == n22 );
  BOOST_CHECK (  n32 == n32 );
}


BOOST_FIXTURE_TEST_CASE(test_extents_static_rank_equal,
                        fixture_extents_static_rank,
                        *boost::unit_test::label("extents_static_rank")
                          *boost::unit_test::label("equal"))
{
  BOOST_CHECK (  n   == n   );
  BOOST_CHECK (  n1  == n1  );
  BOOST_CHECK (  n2  == n2  );
  BOOST_CHECK (  n11 == n11 );
  BOOST_CHECK (  n12 == n12 );
  BOOST_CHECK (  n21 == n21 );
  BOOST_CHECK (  n22 == n22 );
  BOOST_CHECK (  n32 == n32 );
}


BOOST_FIXTURE_TEST_CASE(test_extents_static_equal,
                        fixture_extents_static,
                        *boost::unit_test::label("extents_static")
                          *boost::unit_test::label("equal"))
{
  BOOST_CHECK (  n   == n   );
  BOOST_CHECK (  n1  == n1  );
  BOOST_CHECK (  n2  == n2  );
  BOOST_CHECK (  n11 == n11 );
  BOOST_CHECK (  n12 == n12 );
  BOOST_CHECK (  n21 == n21 );
  BOOST_CHECK (  n22 == n22 );
  BOOST_CHECK (  n32 == n32 );
}


BOOST_FIXTURE_TEST_CASE(test_extents_dynamic_rank_not_equal,
                        fixture_extents_dynamic_rank,
                        *boost::unit_test::label("extents_dynamic_rank")
                          *boost::unit_test::label("not_equal"))
{
  BOOST_CHECK (  ! (n   != n )  );
  BOOST_CHECK (  ! (n1  != n1)  );
  BOOST_CHECK (  ! (n2  != n2)  );
  BOOST_CHECK (  ! (n11 != n11) );
  BOOST_CHECK (  ! (n12 != n12) );
  BOOST_CHECK (  ! (n21 != n21) );
  BOOST_CHECK (  ! (n22 != n22) );
  BOOST_CHECK (  ! (n32 != n32) );
  BOOST_CHECK (   (n2  != n1)  );
  BOOST_CHECK (   (n11 != n12) );
  BOOST_CHECK (   (n12 != n21) );
  BOOST_CHECK (   (n21 != n22) );
  BOOST_CHECK (   (n22 != n32) );
}

BOOST_AUTO_TEST_SUITE_END()
