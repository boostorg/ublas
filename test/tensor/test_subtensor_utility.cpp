//  Copyright (c) 2018 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer and Google in producing this work
//  which started as a Google Summer of Code project.
//


#include <random>
#include <boost/test/unit_test.hpp>

#include "utility.hpp"
#include <boost/numeric/ublas/tensor/subtensor_utility.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>
#include <boost/numeric/ublas/tensor/span.hpp>
#include <boost/numeric/ublas/tensor/tags.hpp>



BOOST_AUTO_TEST_SUITE ( subtensor_utility_testsuite )



struct fixture_sliced_span {
	using span_type = boost::numeric::ublas::sliced_span;

	fixture_sliced_span()
		: spans{
				span_type(),    // 0, a(:)
				span_type(0,0), // 1, a(0:0)
				span_type(0,2), // 2, a(0:2)
				span_type(1,1), // 3, a(1:1)
				span_type(1,3),  // 4, a(1:3)
        span_type(1,boost::numeric::ublas::max), // 5, a(1:end)
        span_type(boost::numeric::ublas::max) // 6, a(end)
				}
	{}
	std::vector<span_type> spans;
};


BOOST_FIXTURE_TEST_CASE( transform_sliced_span_test, fixture_sliced_span )
{

  namespace ublas = boost::numeric::ublas;

//	template<class size_type, class span_tag>
	BOOST_CHECK( ublas::detail::transform_span(spans.at(0), std::size_t(2) ) == ublas::sliced_span(0,1) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(0), std::size_t(3) ) == ublas::sliced_span(0,2) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(0), std::size_t(4) ) == ublas::sliced_span(0,3) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(1), std::size_t(2) ) == ublas::sliced_span(0,0) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(1), std::size_t(3) ) == ublas::sliced_span(0,0) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(1), std::size_t(4) ) == ublas::sliced_span(0,0) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(2), std::size_t(3) ) == ublas::sliced_span(0,2) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(2), std::size_t(4) ) == ublas::sliced_span(0,2) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(2), std::size_t(5) ) == ublas::sliced_span(0,2) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(3), std::size_t(2) ) == ublas::sliced_span(1,1) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(3), std::size_t(3) ) == ublas::sliced_span(1,1) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(3), std::size_t(4) ) == ublas::sliced_span(1,1) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(4), std::size_t(4) ) == ublas::sliced_span(1,3) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(4), std::size_t(5) ) == ublas::sliced_span(1,3) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(4), std::size_t(6) ) == ublas::sliced_span(1,3) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(5), std::size_t(4) ) == ublas::sliced_span(1,3) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(5), std::size_t(5) ) == ublas::sliced_span(1,4) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(5), std::size_t(6) ) == ublas::sliced_span(1,5) );


	BOOST_CHECK( ublas::detail::transform_span(spans.at(6), std::size_t(4) ) == ublas::sliced_span(3,3) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(6), std::size_t(5) ) == ublas::sliced_span(4,4) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(6), std::size_t(6) ) == ublas::sliced_span(5,5) );
}


struct fixture_strided_span {
	using span_type = boost::numeric::ublas::strided_span;

	fixture_strided_span()
		: spans{
				span_type(),       // 0, a(:)
				span_type(0,1,0),  // 1, a(0:1:0)
				span_type(0,2,2),  // 2, a(0:2:2)
				span_type(1,1,1),  // 3, a(1:1:1)
				span_type(1,1,3),  // 4, a(1:1:3)
        span_type(1,2,boost::numeric::ublas::max), // 5, a(1:2:end)
        span_type(boost::numeric::ublas::max) // 6, a(end)
				}
	{}
	std::vector<span_type> spans;
};


BOOST_FIXTURE_TEST_CASE( transform_strided_span_test, fixture_strided_span )
{

	using namespace boost::numeric;

//	template<class size_type, class span_tag>
	BOOST_CHECK( ublas::detail::transform_span(spans.at(0), std::size_t(2) ) == ublas::strided_span(0,1,1) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(0), std::size_t(3) ) == ublas::strided_span(0,1,2) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(0), std::size_t(4) ) == ublas::strided_span(0,1,3) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(1), std::size_t(2) ) == ublas::strided_span(0,1,0) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(1), std::size_t(3) ) == ublas::strided_span(0,1,0) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(1), std::size_t(4) ) == ublas::strided_span(0,1,0) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(2), std::size_t(3) ) == ublas::strided_span(0,2,2) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(2), std::size_t(4) ) == ublas::strided_span(0,2,2) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(2), std::size_t(5) ) == ublas::strided_span(0,2,2) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(3), std::size_t(2) ) == ublas::strided_span(1,1,1) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(3), std::size_t(3) ) == ublas::strided_span(1,1,1) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(3), std::size_t(4) ) == ublas::strided_span(1,1,1) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(4), std::size_t(4) ) == ublas::strided_span(1,1,3) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(4), std::size_t(5) ) == ublas::strided_span(1,1,3) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(4), std::size_t(6) ) == ublas::strided_span(1,1,3) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(5), std::size_t(4) ) == ublas::strided_span(1,2,3) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(5), std::size_t(5) ) == ublas::strided_span(1,2,3) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(5), std::size_t(6) ) == ublas::strided_span(1,2,5) );

	BOOST_CHECK( ublas::detail::transform_span(spans.at(6), std::size_t(4) ) == ublas::strided_span(3,1,3) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(6), std::size_t(5) ) == ublas::strided_span(4,1,4) );
	BOOST_CHECK( ublas::detail::transform_span(spans.at(6), std::size_t(6) ) == ublas::strided_span(5,1,5) );
}






struct fixture_shape {
    using shape = boost::numeric::ublas::extents<>;

  fixture_shape() : extents{
				shape{},    // 0
				shape{1,1}, // 1
				shape{1,2}, // 2
				shape{2,1}, // 3
				shape{2,3}, // 4
				shape{2,3,1}, // 5
				shape{4,1,3}, // 6
				shape{1,2,3}, // 7
				shape{4,2,3}, // 8
				shape{4,2,3,5} // 9
        }
	{}
  std::vector<shape> extents;
};

BOOST_FIXTURE_TEST_CASE( generate_span_array_test, fixture_shape )
{
  namespace ublas = boost::numeric::ublas;
  using span = ublas::sliced_span;

	// shape{}
	{
  auto v = ublas::detail::generate_span_array<span>(extents[0]);
	auto r = std::vector<span>{};
	BOOST_CHECK ( std::equal( v.begin(), v.end(), r.begin(), [](span const& l, span const& r){ return l == r; } )  );
	}


	// shape{1,1}
	{
  auto v = ublas::detail::generate_span_array<span>(extents[1],span(),span());
  auto r = std::vector<span>{span(0,0),span(0,0)};
  BOOST_CHECK ( std::equal( v.begin(), v.end(), r.begin(), [](span const& l, span const& r){ return l == r; } )  );
	}

  // shape{1,1}
  {
    auto v = ublas::detail::generate_span_array<span>(extents[1],ublas::max,span(ublas::max));
	auto r = std::vector<span>{span(0,0),span(0,0)};
  BOOST_CHECK ( std::equal( v.begin(), v.end(), r.begin(), [](span const& l, span const& r){ return l == r; } )  );
	}

  // shape{1,1}
  {
    auto v = ublas::detail::generate_span_array<span>(extents[1],0,ublas::max);
	auto r = std::vector<span>{span(0,0),span(0,0)};
	BOOST_CHECK ( std::equal( v.begin(), v.end(), r.begin(), [](span const& l, span const& r){ return l == r; } )  );
	}

	// shape{1,2}
  {
    auto v = ublas::detail::generate_span_array<span>(extents[2],0,ublas::max);
	auto r = std::vector<span>{span(0,0),span(1,1)};
	BOOST_CHECK ( std::equal( v.begin(), v.end(), r.begin(), [](span const& l, span const& r){ return l == r; } )  );
	}

  // shape{1,2}
	{
  auto v = ublas::detail::generate_span_array<span>(extents[2],0,1);
	auto r = std::vector<span>{span(0,0),span(1,1)};
	BOOST_CHECK ( std::equal( v.begin(), v.end(), r.begin(), [](span const& l, span const& r){ return l == r; } )  );
	}

	{
  auto v = ublas::detail::generate_span_array<span>(extents[2],span(),span());
	auto r = std::vector<span>{span(0,0),span(0,1)};
	BOOST_CHECK ( std::equal( v.begin(), v.end(), r.begin(), [](span const& l, span const& r){ return l == r; } )  );
	}

	// shape{2,3}
	{
  auto v = ublas::detail::generate_span_array<span>(extents[4],span(),span());
	auto r = std::vector<span>{span(0,1),span(0,2)};
	BOOST_CHECK ( std::equal( v.begin(), v.end(), r.begin(), [](span const& l, span const& r){ return l == r; } )  );
	}

	{
  auto v = ublas::detail::generate_span_array<span>(extents[4],1,span(1,ublas::max));
	auto r = std::vector<span>{span(1,1),span(1,2)};
	BOOST_CHECK ( std::equal( v.begin(), v.end(), r.begin(), [](span const& l, span const& r){ return l == r; } )  );
	}

	// shape{2,3,1}
	{
  auto v = ublas::detail::generate_span_array<span>(extents[5],span(),span(),0);
	auto r = std::vector<span>{span(0,1),span(0,2),span(0,0)};
	BOOST_CHECK ( std::equal( v.begin(), v.end(), r.begin(), [](span const& l, span const& r){ return l == r; } )  );
	}

	{
  auto v = ublas::detail::generate_span_array<span>(extents[5],1,span(),ublas::max);
	auto r = std::vector<span>{span(1,1),span(0,2),span(0,0)};
	BOOST_CHECK ( std::equal( v.begin(), v.end(), r.begin(), [](span const& l, span const& r){ return l == r; } )  );
	}
}



struct fixture_span_vector_shape {
  using shape = boost::numeric::ublas::extents<>;
	using span  = boost::numeric::ublas::sliced_span;


	fixture_span_vector_shape()
		: extents_{
				shape{},    // 0
				shape{1,1}, // 1
				shape{1,2}, // 2
				shape{2,3}, // 3
				shape{4,2,3}, // 4
				shape{4,2,3,5} // 5
		}
		, span_vectors_{
        /*A(:)*/             boost::numeric::ublas::detail::generate_span_array<span>(extents_[0]),
        /*A(0,0)*/           boost::numeric::ublas::detail::generate_span_array<span>(extents_[1],0,0),
        /*A(0,:)*/           boost::numeric::ublas::detail::generate_span_array<span>(extents_[2],0,span()),
        /*A(1,1:2)*/         boost::numeric::ublas::detail::generate_span_array<span>(extents_[3],1,span(1,2)),
        /*A(1:3,1,1:2)*/     boost::numeric::ublas::detail::generate_span_array<span>(extents_[4],span(1,3),1,span(0,1)),
        /*A(1:3,1,0:1,2:4)*/ boost::numeric::ublas::detail::generate_span_array<span>(extents_[5],span(1,3),1,span(0,1),span(2,4)),
		}
		, reference_ {
				shape{},
				shape{1,1},
				shape{1,2},
				shape{1,2},
				shape{3,1,2},
				shape{3,1,2,3}
		}
	{
		assert(extents_.size() == reference_.size());
    assert(extents_.size() == std::tuple_size_v<decltype(span_vectors_)>);
  }
  std::array<shape,6> extents_;
  std::tuple<
    std::array<span,0>,
    std::array<span,2>,
    std::array<span,2>,
    std::array<span,2>,
    std::array<span,3>,
    std::array<span,4>
  > span_vectors_;

  std::array<shape,6> reference_;
};



BOOST_FIXTURE_TEST_CASE( extents_test, fixture_span_vector_shape )
{
  namespace ublas = boost::numeric::ublas;

  BOOST_CHECK ( std::equal( ublas::begin(std::get<0>(reference_)), ublas::begin(std::get<0>(reference_)),  ublas::begin(ublas::detail::to_extents( std::get<0>(span_vectors_) ) ) ) );
  BOOST_CHECK ( std::equal( ublas::begin(std::get<1>(reference_)), ublas::begin(std::get<1>(reference_)),  ublas::begin(ublas::detail::to_extents( std::get<1>(span_vectors_) ) ) ) );
  BOOST_CHECK ( std::equal( ublas::begin(std::get<2>(reference_)), ublas::begin(std::get<2>(reference_)),  ublas::begin(ublas::detail::to_extents( std::get<2>(span_vectors_) ) ) ) );
  BOOST_CHECK ( std::equal( ublas::begin(std::get<3>(reference_)), ublas::begin(std::get<3>(reference_)),  ublas::begin(ublas::detail::to_extents( std::get<3>(span_vectors_) ) ) ) );
  BOOST_CHECK ( std::equal( ublas::begin(std::get<4>(reference_)), ublas::begin(std::get<4>(reference_)),  ublas::begin(ublas::detail::to_extents( std::get<4>(span_vectors_) ) ) ) );
  BOOST_CHECK ( std::equal( ublas::begin(std::get<5>(reference_)), ublas::begin(std::get<5>(reference_)),  ublas::begin(ublas::detail::to_extents( std::get<5>(span_vectors_) ) ) ) );

}


using test_types = std::tuple<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;


BOOST_FIXTURE_TEST_CASE_TEMPLATE( offset_test, layout, test_types, fixture_span_vector_shape )
{
  namespace ublas = boost::numeric::ublas;


	{
    auto s = std::get<0>(span_vectors_);
    auto w = ublas::to_strides( std::get<0>(extents_), layout{} );
    auto o = ublas::detail::to_offset(w,s);
		BOOST_CHECK_EQUAL( o, 0  );
	}

	{
    auto s = std::get<1>(span_vectors_);
    auto w = ublas::to_strides( std::get<1>(extents_), layout{} );
    auto o = ublas::detail::to_offset(w,s);
		BOOST_CHECK_EQUAL( o, 0  );
	}

	{
    auto s = std::get<2>(span_vectors_);
    auto w = ublas::to_strides( std::get<2>(extents_), layout{} );
    auto o = ublas::detail::to_offset(w,s);
		BOOST_CHECK_EQUAL( o, 0  );
	}

	{
    auto s = std::get<3>(span_vectors_);
    auto w = ublas::to_strides( std::get<3>(extents_), layout{} );
    auto o = ublas::detail::to_offset(w,s);
		BOOST_CHECK_EQUAL( o, s[0].first()*w[0] + s[1].first()*w[1]  );
	}

	{
    auto s = std::get<4>(span_vectors_);
    auto w = ublas::to_strides( std::get<4>(extents_), layout{} );
    auto o = ublas::detail::to_offset(w,s);
		BOOST_CHECK_EQUAL( o, s[0].first()*w[0] + s[1].first()*w[1] + s[2].first()*w[2]   );
	}


	{
    auto s = std::get<5>(span_vectors_);
    auto w = ublas::to_strides( std::get<5>(extents_), layout{} );
    auto o = ublas::detail::to_offset(w,s);
    BOOST_CHECK_EQUAL( o, s[0].first()*w[0] + s[1].first()*w[1] + s[2].first()*w[2] +  s[3].first()*w[3] );
	}

}


#if 0



BOOST_FIXTURE_TEST_CASE_TEMPLATE( span_strides_test, layout, test_types, fixture_span_vector_shape )
{

	/*A(:)*/
	/*A(0,0)*/
	/*A(0,:)*/
	/*A(1,1:2)*/
	/*A(1:3,1,1:2)*/
	/*A(1:3,1,0:1,2:4)*/


	// auto span_strides(strides_type const& strides, std::vector<span_type> const& spans)

	using namespace boost::numeric;
  using strides = ublas::basic_strides<std::size_t,layout>;

	for(unsigned k = 0; k < span_vectors_.size(); ++k)
	{
		auto s = span_vectors_[k];
		auto w = strides( extents_[k] );
		auto ss = ublas::detail::span_strides(  w, s  );
		for(unsigned i = 0; i < w.size(); ++i)
			BOOST_CHECK_EQUAL( ss[i], w[i]*s[i].step() );
	}

}

#endif

BOOST_AUTO_TEST_SUITE_END()
