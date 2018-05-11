//  Copyright (c) 2018 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB in producing this work.
//
//  And we acknowledge the support from all contributors.



#include <boost/numeric/ublas/tensor/expression.hpp>
#include <boost/numeric/ublas/tensor.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include "utility.hpp"


using double_extended = boost::multiprecision::cpp_bin_float_double_extended;

using test_types = zip<int,long,float,double,double_extended>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;

//using test_types = zip<int>::with_t<boost::numeric::ublas::last_order>;

BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_expression_access, value,  test_types)
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using tensor_type = ublas::tensor<value_type, layout_type>;
	using super_type  = typename tensor_type::super_type;

	auto v = value_type{};

	auto t = tensor_type{5,4,3};
	std::iota(t.begin(), t.end(), v);
	const auto& super_const = static_cast<super_type const&>( t );
	auto& super = static_cast<super_type &>( t );

	for(auto i = 0ul; i < t.size(); ++i)
		BOOST_CHECK_EQUAL( super_const(i), t(i)  );

	for(auto i = 0ul; i < t.size(); ++i, ++v)
		super(i) = v;

	for(auto i = 0ul; i < t.size(); ++i)
		BOOST_CHECK_EQUAL( super(i), t(i)  );
}



//BOOST_AUTO_TEST_CASE_TEMPLATE( test_tensor_expression_make_lambda, value,  test_types)
//{
//	using namespace boost::numeric;
//	using value_type  = typename value::first_type;
//	using layout_type = typename value::second_type;
//	using tensor_type = ublas::tensor<value_type, layout_type>;

//	auto t = tensor_type{5,4,3};

//	auto op = [&t](std::size_t i){ return t(i)+1;};
//	auto lambda = ublas::detail::lambda<tensor_type, decltype(op)>(op);

//	for(auto i = 0ul; i < t.size(); ++i)
//		BOOST_CHECK_EQUAL( lambda(i), t(i)+1  );

//	auto lambda2 = ublas::detail::make_lambda<tensor_type>( [&t](std::size_t i){ return t(i)+1;}  );
//	auto lambda3 = ublas::detail::make_lambda<tensor_type>( [&lambda2](std::size_t i){ return lambda2(i)+1;}  );

//	auto t2 = tensor_type(t.extents());
//	t2 = lambda2;

//	for(auto i = 0ul; i < t.size(); ++i)
//		BOOST_CHECK_EQUAL( t2(i), t(i)+1  );

//	auto t3 = tensor_type(t.extents());
//	t3 = lambda3;

//	for(auto i = 0ul; i < t.size(); ++i)
//		BOOST_CHECK_EQUAL( t3(i), t(i)+2  );

//}
