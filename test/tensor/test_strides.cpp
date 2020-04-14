//
// 	Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//



#include <boost/test/unit_test.hpp>
#include <boost/numeric/ublas/tensor/strides.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>

//BOOST_AUTO_TEST_SUITE(test_strides, * boost::unit_test::depends_on("test_extents"));

BOOST_AUTO_TEST_SUITE(test_strides)

using test_types = std::tuple<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;

BOOST_AUTO_TEST_CASE_TEMPLATE( test_strides_ctor, value, test_types)
{
	using namespace boost::numeric;

	using extents_type  = ublas::basic_extents<unsigned>;
	using strides_type = ublas::strides_t<extents_type,ublas::first_order>;

	strides_type         s0{};
	BOOST_CHECK        ( s0.empty());
	BOOST_CHECK_EQUAL  ( s0.size(), 0);

	strides_type        s1{extents_type{1,1}};
	BOOST_CHECK       (!s1.empty());
	BOOST_CHECK_EQUAL ( s1.size(), 2);

	strides_type        s2{extents_type{1,2}};
	BOOST_CHECK       (!s2.empty());
	BOOST_CHECK_EQUAL ( s2.size(), 2);

	strides_type        s3{extents_type{2,1}};
	BOOST_CHECK       (!s3.empty());
	BOOST_CHECK_EQUAL ( s3.size(), 2);

	strides_type        s4{extents_type{2,3}};
	BOOST_CHECK       (!s4.empty());
	BOOST_CHECK_EQUAL ( s4.size(), 2);

	strides_type        s5{extents_type{2,3,1}};
	BOOST_CHECK       (!s5.empty());
	BOOST_CHECK_EQUAL ( s5.size(), 3);

	strides_type        s6{extents_type{1,2,3}};
	BOOST_CHECK       (!s6.empty());
	BOOST_CHECK_EQUAL ( s6.size(), 3);

	strides_type        s7{extents_type{4,2,3}};
	BOOST_CHECK       (!s7.empty());
	BOOST_CHECK_EQUAL ( s7.size(), 3);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( test_static_rank_strides_ctor, value, test_types)
{
	namespace ub = boost::numeric::ublas;

	ub::basic_fixed_rank_strides<unsigned, 0, ub::first_order> s0{ub::dynamic_extents<0>{}};
	BOOST_CHECK        ( s0.empty());
	BOOST_CHECK_EQUAL  ( s0.size(), 0);

	ub::basic_fixed_rank_strides<unsigned, 2, ub::first_order> s1{ub::dynamic_extents<2>{1,1}};
	BOOST_CHECK       (!s1.empty());
	BOOST_CHECK_EQUAL ( s1.size(), 2);

	ub::basic_fixed_rank_strides<unsigned, 2, ub::first_order> s2{ub::dynamic_extents<2>{1,2}};
	BOOST_CHECK       (!s2.empty());
	BOOST_CHECK_EQUAL ( s2.size(), 2);

	ub::basic_fixed_rank_strides<unsigned, 2, ub::first_order> s3{ub::dynamic_extents<2>{2,1}};
	BOOST_CHECK       (!s3.empty());
	BOOST_CHECK_EQUAL ( s3.size(), 2);

	ub::basic_fixed_rank_strides<unsigned, 2, ub::first_order> s4{ub::dynamic_extents<2>{2,3}};
	BOOST_CHECK       (!s4.empty());
	BOOST_CHECK_EQUAL ( s4.size(), 2);

	ub::basic_fixed_rank_strides<unsigned, 3, ub::first_order> s5{ub::dynamic_extents<3>{2,3,1}};
	BOOST_CHECK       (!s5.empty());
	BOOST_CHECK_EQUAL ( s5.size(), 3);

	ub::basic_fixed_rank_strides<unsigned, 3, ub::first_order> s6{ub::dynamic_extents<3>{1,2,3}};
	BOOST_CHECK       (!s6.empty());
	BOOST_CHECK_EQUAL ( s6.size(), 3);

	ub::basic_fixed_rank_strides<unsigned, 3, ub::first_order> s7{ub::dynamic_extents<3>{4,2,3}};
	BOOST_CHECK       (!s7.empty());
	BOOST_CHECK_EQUAL ( s7.size(), 3);
}



BOOST_AUTO_TEST_CASE( test_strides_ctor_access_first_order)
{
	using namespace boost::numeric;

	using extents_type  = ublas::basic_extents<unsigned>;
	using strides_type = ublas::strides_t<extents_type,ublas::first_order>;

	strides_type         s1{extents_type{1,1}};
	BOOST_REQUIRE_EQUAL( s1.size(),2);
	BOOST_CHECK_EQUAL  ( s1[0], 1);
	BOOST_CHECK_EQUAL  ( s1[1], 1);

	strides_type          s2{extents_type{1,2}};
	BOOST_REQUIRE_EQUAL ( s2.size(),2);
	BOOST_CHECK_EQUAL   ( s2[0], 1);
	BOOST_CHECK_EQUAL   ( s2[1], 1);

	strides_type          s3{extents_type{2,1}};
	BOOST_REQUIRE_EQUAL ( s3.size(),2);
	BOOST_CHECK_EQUAL   ( s3[0], 1);
	BOOST_CHECK_EQUAL   ( s3[1], 1);

	strides_type          s4{extents_type{2,3}};
	BOOST_REQUIRE_EQUAL ( s4.size(),2);
	BOOST_CHECK_EQUAL   ( s4[0], 1);
	BOOST_CHECK_EQUAL   ( s4[1], 2);

	strides_type          s5{extents_type{2,3,1}};
	BOOST_REQUIRE_EQUAL ( s5.size(),3);
	BOOST_CHECK_EQUAL   ( s5[0], 1);
	BOOST_CHECK_EQUAL   ( s5[1], 2);
	BOOST_CHECK_EQUAL   ( s5[2], 6);

	strides_type          s6{extents_type{1,2,3}};
	BOOST_REQUIRE_EQUAL ( s6.size(),3);
	BOOST_CHECK_EQUAL   ( s6[0], 1);
	BOOST_CHECK_EQUAL   ( s6[1], 1);
	BOOST_CHECK_EQUAL   ( s6[2], 2);

	strides_type          s7{extents_type{2,1,3}};
	BOOST_REQUIRE_EQUAL ( s7.size(),3);
	BOOST_CHECK_EQUAL   ( s7[0], 1);
	BOOST_CHECK_EQUAL   ( s7[1], 2);
	BOOST_CHECK_EQUAL   ( s7[2], 2);

	strides_type          s8{extents_type{4,2,3}};
	BOOST_REQUIRE_EQUAL ( s8.size(),3);
	BOOST_CHECK_EQUAL   ( s8[0], 1);
	BOOST_CHECK_EQUAL   ( s8[1], 4);
	BOOST_CHECK_EQUAL   ( s8[2], 8);
}

BOOST_AUTO_TEST_CASE( test_static_rank_strides_ctor_access_first_order)
{
	namespace ub = boost::numeric::ublas;

	ub::basic_fixed_rank_strides<unsigned, 2, ub::first_order> s1{ub::dynamic_extents<2>{1,1}};
	BOOST_REQUIRE_EQUAL( s1.size(),2);
	BOOST_CHECK_EQUAL  ( s1[0], 1);
	BOOST_CHECK_EQUAL  ( s1[1], 1);

	ub::basic_fixed_rank_strides<unsigned, 2, ub::first_order> s2{ub::dynamic_extents<2>{1,2}};
	BOOST_REQUIRE_EQUAL ( s2.size(),2);
	BOOST_CHECK_EQUAL   ( s2[0], 1);
	BOOST_CHECK_EQUAL   ( s2[1], 1);

	ub::basic_fixed_rank_strides<unsigned, 2, ub::first_order> s3{ub::dynamic_extents<2>{2,1}};
	BOOST_REQUIRE_EQUAL ( s3.size(),2);
	BOOST_CHECK_EQUAL   ( s3[0], 1);
	BOOST_CHECK_EQUAL   ( s3[1], 1);

	ub::basic_fixed_rank_strides<unsigned, 2, ub::first_order> s4{ub::dynamic_extents<2>{2,3}};
	BOOST_REQUIRE_EQUAL ( s4.size(),2);
	BOOST_CHECK_EQUAL   ( s4[0], 1);
	BOOST_CHECK_EQUAL   ( s4[1], 2);

	ub::basic_fixed_rank_strides<unsigned, 3, ub::first_order> s5{ub::dynamic_extents<3>{2,3,1}};
	BOOST_REQUIRE_EQUAL ( s5.size(),3);
	BOOST_CHECK_EQUAL   ( s5[0], 1);
	BOOST_CHECK_EQUAL   ( s5[1], 2);
	BOOST_CHECK_EQUAL   ( s5[2], 6);

	ub::basic_fixed_rank_strides<unsigned, 3, ub::first_order> s6{ub::dynamic_extents<3>{1,2,3}};
	BOOST_REQUIRE_EQUAL ( s6.size(),3);
	BOOST_CHECK_EQUAL   ( s6[0], 1);
	BOOST_CHECK_EQUAL   ( s6[1], 1);
	BOOST_CHECK_EQUAL   ( s6[2], 2);

	ub::basic_fixed_rank_strides<unsigned, 3, ub::first_order> s7{ub::dynamic_extents<3>{2,1,3}};
	BOOST_REQUIRE_EQUAL ( s7.size(),3);
	BOOST_CHECK_EQUAL   ( s7[0], 1);
	BOOST_CHECK_EQUAL   ( s7[1], 2);
	BOOST_CHECK_EQUAL   ( s7[2], 2);

	ub::basic_fixed_rank_strides<unsigned, 3, ub::first_order> s8{ub::dynamic_extents<3>{4,2,3}};
	BOOST_REQUIRE_EQUAL ( s8.size(),3);
	BOOST_CHECK_EQUAL   ( s8[0], 1);
	BOOST_CHECK_EQUAL   ( s8[1], 4);
	BOOST_CHECK_EQUAL   ( s8[2], 8);
}

BOOST_AUTO_TEST_CASE( test_strides_ctor_access_last_order)
{
	using namespace boost::numeric;

	using extents_type  = ublas::basic_extents<unsigned>;
	using strides_type = ublas::strides_t<extents_type,ublas::last_order>;

	strides_type         s1{extents_type{1,1}};
	BOOST_REQUIRE_EQUAL( s1.size(),2);
	BOOST_CHECK_EQUAL  ( s1[0], 1);
	BOOST_CHECK_EQUAL  ( s1[1], 1);

	strides_type          s2{extents_type{1,2}};
	BOOST_REQUIRE_EQUAL ( s2.size(),2);
	BOOST_CHECK_EQUAL   ( s2[0], 1);
	BOOST_CHECK_EQUAL   ( s2[1], 1);

	strides_type          s3{extents_type{2,1}};
	BOOST_REQUIRE_EQUAL ( s3.size(),2);
	BOOST_CHECK_EQUAL   ( s3[0], 1);
	BOOST_CHECK_EQUAL   ( s3[1], 1);

	strides_type          s4{extents_type{2,3}};
	BOOST_REQUIRE_EQUAL ( s4.size(),2);
	BOOST_CHECK_EQUAL   ( s4[0], 3);
	BOOST_CHECK_EQUAL   ( s4[1], 1);

	strides_type          s5{extents_type{2,3,1}};
	BOOST_REQUIRE_EQUAL ( s5.size(),3);
	BOOST_CHECK_EQUAL   ( s5[0], 3);
	BOOST_CHECK_EQUAL   ( s5[1], 1);
	BOOST_CHECK_EQUAL   ( s5[2], 1);

	strides_type          s6{extents_type{1,2,3}};
	BOOST_REQUIRE_EQUAL ( s6.size(),3);
	BOOST_CHECK_EQUAL   ( s6[0], 6);
	BOOST_CHECK_EQUAL   ( s6[1], 3);
	BOOST_CHECK_EQUAL   ( s6[2], 1);

	strides_type          s7{extents_type{2,1,3}};
	BOOST_REQUIRE_EQUAL ( s7.size(),3);
	BOOST_CHECK_EQUAL   ( s7[0], 3);
	BOOST_CHECK_EQUAL   ( s7[1], 3);
	BOOST_CHECK_EQUAL   ( s7[2], 1);

	strides_type          s8{extents_type{4,2,3}};
	BOOST_REQUIRE_EQUAL ( s8.size(),3);
	BOOST_CHECK_EQUAL   ( s8[0], 6);
	BOOST_CHECK_EQUAL   ( s8[1], 3);
	BOOST_CHECK_EQUAL   ( s8[2], 1);
}

BOOST_AUTO_TEST_CASE( test_static_rank_strides_ctor_access_last_order)
{
	namespace ub = boost::numeric::ublas;

	ub::basic_fixed_rank_strides<unsigned, 2, ub::last_order> s1{ub::dynamic_extents<2>{1,1}};
	BOOST_REQUIRE_EQUAL( s1.size(),2);
	BOOST_CHECK_EQUAL  ( s1[0], 1);
	BOOST_CHECK_EQUAL  ( s1[1], 1);

	ub::basic_fixed_rank_strides<unsigned, 2, ub::last_order> s2{ub::dynamic_extents<2>{1,2}};
	BOOST_REQUIRE_EQUAL ( s2.size(),2);
	BOOST_CHECK_EQUAL   ( s2[0], 1);
	BOOST_CHECK_EQUAL   ( s2[1], 1);

	ub::basic_fixed_rank_strides<unsigned, 2, ub::last_order> s3{ub::dynamic_extents<2>{2,1}};
	BOOST_REQUIRE_EQUAL ( s3.size(),2);
	BOOST_CHECK_EQUAL   ( s3[0], 1);
	BOOST_CHECK_EQUAL   ( s3[1], 1);

	ub::basic_fixed_rank_strides<unsigned, 2, ub::last_order> s4{ub::dynamic_extents<2>{2,3}};
	BOOST_REQUIRE_EQUAL ( s4.size(),2);
	BOOST_CHECK_EQUAL   ( s4[0], 3);
	BOOST_CHECK_EQUAL   ( s4[1], 1);

	ub::basic_fixed_rank_strides<unsigned, 3, ub::last_order> s5{ub::dynamic_extents<3>{2,3,1}};
	BOOST_REQUIRE_EQUAL ( s5.size(),3);
	BOOST_CHECK_EQUAL   ( s5[0], 3);
	BOOST_CHECK_EQUAL   ( s5[1], 1);
	BOOST_CHECK_EQUAL   ( s5[2], 1);

	ub::basic_fixed_rank_strides<unsigned, 3, ub::last_order> s6{ub::dynamic_extents<3>{1,2,3}};
	BOOST_REQUIRE_EQUAL ( s6.size(),3);
	BOOST_CHECK_EQUAL   ( s6[0], 6);
	BOOST_CHECK_EQUAL   ( s6[1], 3);
	BOOST_CHECK_EQUAL   ( s6[2], 1);

	ub::basic_fixed_rank_strides<unsigned, 3, ub::last_order> s7{ub::dynamic_extents<3>{2,1,3}};
	BOOST_REQUIRE_EQUAL ( s7.size(),3);
	BOOST_CHECK_EQUAL   ( s7[0], 3);
	BOOST_CHECK_EQUAL   ( s7[1], 3);
	BOOST_CHECK_EQUAL   ( s7[2], 1);

	ub::basic_fixed_rank_strides<unsigned, 3, ub::last_order> s8{ub::dynamic_extents<3>{4,2,3}};
	BOOST_REQUIRE_EQUAL ( s8.size(),3);
	BOOST_CHECK_EQUAL   ( s8[0], 6);
	BOOST_CHECK_EQUAL   ( s8[1], 3);
	BOOST_CHECK_EQUAL   ( s8[2], 1);
}

BOOST_AUTO_TEST_SUITE_END()
