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


#include <iostream>
#include <algorithm>
#include <vector>
#include <boost/numeric/ublas/tensor/algorithms.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>
#include <boost/numeric/ublas/tensor/strides.hpp>
#include "utility.hpp"

#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_SUITE ( test_tensor_algorithms,
                        * boost::unit_test::depends_on("test_extents")
                        * boost::unit_test::depends_on("test_strides"))

// BOOST_AUTO_TEST_SUITE ( test_tensor_algorithms)


using test_types  = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;
using test_types2 = std::tuple<int,long,float,double,std::complex<float>>;

struct fixture
{
	using extents_type = boost::numeric::ublas::dynamic_extents<>;
	fixture()
	  : extents {
	      extents_type{1,1}, // 1
	      extents_type{1,2}, // 2
	      extents_type{2,1}, // 3
	      extents_type{2,3}, // 4
	      extents_type{2,3,1}, // 5
	      extents_type{4,1,3}, // 6
	      extents_type{1,2,3}, // 7
	      extents_type{4,2,3}, // 8
	      extents_type{4,2,3,5} } // 9
	{
	}
	std::vector<extents_type> extents;
};




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_algorithms_copy, value,  test_types2, fixture )
{
	using namespace boost::numeric;
	using value_type   = value;
	using vector_type  = std::vector<value_type>;


	for(auto const& n : extents) {

		auto a  = vector_type(product(n));
		auto b  = vector_type(product(n));
		auto c  = vector_type(product(n));

		auto wa = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);
		auto wb = ublas::strides_t<ublas::dynamic_extents<>,ublas::last_order> (n);
		auto wc = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);

		auto v = value_type{};
		for(auto i = 0ul; i < a.size(); ++i, v+=1){
			a[i]=v;
		}

		ublas::copy( n.size(), n.data(), b.data(), wb.data(), a.data(), wa.data() );
		ublas::copy( n.size(), n.data(), c.data(), wc.data(), b.data(), wb.data() );

		for(auto i = 1ul; i < c.size(); ++i)
			BOOST_CHECK_EQUAL( c[i], a[i] );

		using size_type = typename ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>::value_type;
		size_type const*const p0 = nullptr;
		BOOST_CHECK_THROW( ublas::copy( n.size(), p0, c.data(), wc.data(), b.data(), wb.data() ), std::runtime_error );
		BOOST_CHECK_THROW( ublas::copy( n.size(), n.data(), c.data(), p0, b.data(), wb.data() ), std::runtime_error );
		BOOST_CHECK_THROW( ublas::copy( n.size(), n.data(), c.data(), wc.data(), b.data(), p0 ), std::runtime_error );

		value_type* c0 = nullptr;
		BOOST_CHECK_THROW( ublas::copy( n.size(), n.data(), c0, wc.data(), b.data(), wb.data() ), std::runtime_error );
	}

	// special case rank == 0
	{
		auto n = ublas::dynamic_extents<>{};

		auto a  = vector_type(product(n));
		auto b  = vector_type(product(n));
		auto c  = vector_type(product(n));


		auto wa = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);
		auto wb = ublas::strides_t<ublas::dynamic_extents<>,ublas::last_order> (n);
		auto wc = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);

		ublas::copy( n.size(), n.data(), b.data(), wb.data(), a.data(), wa.data() );
		ublas::copy( n.size(), n.data(), c.data(), wc.data(), b.data(), wb.data() );



		BOOST_CHECK_NO_THROW( ublas::copy( n.size(), n.data(), c.data(), wc.data(), b.data(), wb.data() ) );

	}





}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_algorithms_copy_exceptions, value,  test_types2, fixture )
{
	using namespace boost::numeric;
	using value_type   = value;
	using vector_type  = std::vector<value_type>;

	for(auto const& n : extents) {

		value_type* a  = nullptr;
		auto c  = vector_type(product(n));

		auto wa = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);
		auto wc = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);

		BOOST_REQUIRE_THROW( ublas::copy( n.size(), n.data(), c.data(), wc.data(), a, wa.data() ), std::runtime_error );
		
	}

	for(auto const& n : extents) {

		value_type* a  = nullptr;
		value_type* c  = nullptr;

		auto wa = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);
		auto wc = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);

		BOOST_REQUIRE_THROW( ublas::copy( n.size(), n.data(), c, wc.data(), a, wa.data() ), std::runtime_error );

	}

	for(auto const& n : extents) {

			auto a  = vector_type(product(n));
			value_type* c  = nullptr;

			auto wa = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);
			auto wc = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);

			BOOST_REQUIRE_THROW( ublas::copy( n.size(), n.data(), c, wc.data(), a.data(), wa.data() ), std::runtime_error );
			
	}

	for(auto const& n : extents) {

			auto a  = vector_type(product(n));
			auto c  = vector_type(product(n));

			size_t* wa = nullptr;
			auto wc = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);

			BOOST_REQUIRE_THROW( ublas::copy( n.size(), n.data(), c.data(), wc.data(), a.data(), wa ), std::runtime_error );
			
	}

	for(auto const& n : extents) {

			auto a  = vector_type(product(n));
			auto c  = vector_type(product(n));

			size_t* wc = nullptr;
			auto wa = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);

			BOOST_REQUIRE_THROW( ublas::copy( n.size(), n.data(), c.data(), wc, a.data(), wa.data() ), std::runtime_error );
			
	}

	for(auto const& n : extents) {

			auto a  = vector_type(product(n));
			auto c  = vector_type(product(n));
			
			size_t* m = nullptr;
			auto wc = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);
			auto wa = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);

			BOOST_REQUIRE_THROW( ublas::copy( n.size(), m, c.data(), wc.data(), a.data(), wa.data() ), std::runtime_error );
			
	}
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_algorithms_transform, value,  test_types2, fixture )
{
	using namespace boost::numeric;
	using value_type   = value;
	using vector_type  = std::vector<value_type>;


	for(auto const& n : extents) {

		auto a  = vector_type(product(n));
		auto b  = vector_type(product(n));
		auto c  = vector_type(product(n));

		auto wa = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);
		auto wb = ublas::strides_t<ublas::dynamic_extents<>,ublas::last_order> (n);
		auto wc = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);

		auto v = value_type{};
		for(auto i = 0ul; i < a.size(); ++i, v+=1){
			a[i]=v;
		}

		ublas::transform( n.size(), n.data(), b.data(), wb.data(), a.data(), wa.data(), [](value_type const& a){ return a + value_type(1);} );
		ublas::transform( n.size(), n.data(), c.data(), wc.data(), b.data(), wb.data(), [](value_type const& a){ return a - value_type(1);} );

        using size_type = typename ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>::value_type;

        size_type zero = 0;
		ublas::transform(zero, n.data(), c.data(), wc.data(), b.data(), wb.data(), [](value_type const& a){ return a + value_type(1);} );

		value_type* c0 = nullptr;
        const size_type* s0 = nullptr;
        size_type const*const p0 = nullptr;

        BOOST_CHECK_THROW(ublas::transform( n.size(), n.data(), c0, wb.data(), a.data(), wa.data(), [](value_type const& a){ return a + value_type(1);} ), std::runtime_error);
        BOOST_CHECK_THROW(ublas::transform( n.size(), n.data(), b.data(), s0, a.data(), wa.data(), [](value_type const& a){ return a + value_type(1);} ), std::runtime_error);
        BOOST_CHECK_THROW(ublas::transform( n.size(), p0, b.data(), wb.data(), a.data(), wa.data(), [](value_type const& a){ return a + value_type(1);} ), std::runtime_error);


        for(auto i = 1ul; i < c.size(); ++i)
			BOOST_CHECK_EQUAL( c[i], a[i] );

	}
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_algorithms_transform_exceptions, value,  test_types2, fixture )
{
	using namespace boost::numeric;
	using value_type   = value;
	using vector_type  = std::vector<value_type>;

	for(auto const& n : extents) {

		value_type* a  = nullptr;
		auto c  = vector_type(product(n));

		auto wa = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);
		auto wc = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);

		BOOST_REQUIRE_THROW( ublas::transform( n.size(), n.data(), c.data(), wc.data(), a, wa.data(), [](value_type const& a){ return a + value_type(1);} ), std::runtime_error );
		
	}

	for(auto const& n : extents) {

		value_type* a  = nullptr;
		value_type* c  = nullptr;

		auto wa = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);
		auto wc = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);

		BOOST_REQUIRE_THROW( ublas::transform( n.size(), n.data(), c, wc.data(), a, wa.data(), [](value_type const& a){ return a + value_type(1);} ), std::runtime_error );

	}

	for(auto const& n : extents) {

			auto a  = vector_type(product(n));
			value_type* c  = nullptr;

			auto wa = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);
			auto wc = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);

			BOOST_REQUIRE_THROW( ublas::transform( n.size(), n.data(), c, wc.data(), a.data(), wa.data(), [](value_type const& a){ return a + value_type(1);} ), std::runtime_error );
			
	}

	for(auto const& n : extents) {

			auto a  = vector_type(product(n));
			auto c  = vector_type(product(n));

			size_t* wa = nullptr;
			auto wc = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);

			BOOST_REQUIRE_THROW( ublas::transform( n.size(), n.data(), c.data(), wc.data(), a.data(), wa, [](value_type const& a){ return a + value_type(1);} ), std::runtime_error );
			
	}

	for(auto const& n : extents) {

			auto a  = vector_type(product(n));
			auto c  = vector_type(product(n));

			size_t* wc = nullptr;
			auto wa = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);

			BOOST_REQUIRE_THROW( ublas::transform( n.size(), n.data(), c.data(), wc, a.data(), wa.data(), [](value_type const& a){ return a + value_type(1);} ), std::runtime_error );
			
	}

	for(auto const& n : extents) {

			auto a  = vector_type(product(n));
			auto c  = vector_type(product(n));
			
			size_t* m = nullptr;
			auto wc = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);
			auto wa = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);

			BOOST_REQUIRE_THROW( ublas::transform( n.size(), m, c.data(), wc.data(), a.data(), wa.data(), [](value_type const& a){ return a + value_type(1);} ), std::runtime_error );
			
	}
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_algorithms_accumulate, value,  test_types2, fixture )
{
	using namespace boost::numeric;
	using value_type   = value;
	using vector_type  = std::vector<value_type>;


	for(auto const& n : extents) {

		auto const s = product(n);

		auto a  = vector_type(product(n));
		//		auto b  = vector_type(product(n));
		//		auto c  = vector_type(product(n));

		auto wa = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);
		//		auto wb = ublas::strides_t<ublas::dynamic_extents<>,ublas::last_order> (n);
		//		auto wc = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);

		auto v = value_type{};
		for(auto i = 0ul; i < a.size(); ++i, v+=value_type(1)){
			a[i]=v;
		}

		auto acc = ublas::accumulate( n.size(), n.data(), a.data(), wa.data(), v);

		BOOST_CHECK_EQUAL( acc, value_type( static_cast< inner_type_t<value_type> >( s*(s+1) / 2 ) )  );

                using size_type = typename ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>::value_type;
                size_type zero = 0;
                ublas::accumulate(zero, n.data(), a.data(), wa.data(),v);

                value_type* c0 = nullptr;
                size_type const*const p0 = nullptr;

                BOOST_CHECK_THROW(ublas::accumulate( n.size(), n.data(), c0, wa.data(), v), std::runtime_error);
                BOOST_CHECK_THROW(ublas::accumulate( n.size(), n.data(), a.data(), p0, v), std::runtime_error);
                BOOST_CHECK_THROW(ublas::accumulate( n.size(), p0, a.data(), wa.data(), v), std::runtime_error);


                auto acc2 = ublas::accumulate( n.size(), n.data(), a.data(), wa.data(), v,
		                               [](auto const& l, auto const& r){return l + r; });

                BOOST_CHECK_EQUAL( acc2, value_type( static_cast< inner_type_t<value_type> >( s*(s+1) / 2 ) )  );

                ublas::accumulate(zero, n.data(), a.data(), wa.data(), v, [](auto const& l, auto const& r){return l + r; });

                BOOST_CHECK_THROW(ublas::accumulate( n.size(), n.data(), c0, wa.data(), v,[](auto const& l, auto const& r){return l + r; }), std::runtime_error);
                BOOST_CHECK_THROW(ublas::accumulate( n.size(), n.data(), a.data(), p0, v, [](auto const& l, auto const& r){return l + r; }), std::runtime_error);
                BOOST_CHECK_THROW(ublas::accumulate( n.size(), p0, a.data(), wa.data(),v, [](auto const& l, auto const& r){return l + r; }), std::runtime_error);

	}
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_algorithms_accumulate_exceptions, value,  test_types2, fixture )
{
	using namespace boost::numeric;
	using value_type   = value;
	using vector_type  = std::vector<value_type>;

	for(auto const& n : extents) {

		value_type* a  = nullptr;

		auto wa = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);

		BOOST_REQUIRE_THROW( ublas::accumulate( n.size(), n.data(), a, wa.data(), value_type{0} ), std::runtime_error );
		
	}

	for(auto const& n : extents) {

		value_type* a  = nullptr;

		auto wa = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);

		BOOST_REQUIRE_THROW( ublas::accumulate( n.size(), n.data(), a, wa.data(), value_type{0},[](value_type const& a,value_type const& b){ return a + b;} ), std::runtime_error );
		
	}

	for(auto const& n : extents) {

		auto a  = vector_type(product(n));

		auto wa = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);
		size_t p = 0u;
		BOOST_CHECK_EQUAL ( ublas::accumulate( p, n.data(), a.data(), wa.data(), value_type{0} ), value_type{0} );
		
	}

	for(auto const& n : extents) {

		auto a  = vector_type(product(n));

		auto wa = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);
		size_t p = 0u;
		BOOST_CHECK_EQUAL( ublas::accumulate( p, n.data(), a.data(), wa.data(), value_type{0}, [](value_type const& a,value_type const& b){ return a + b;} ), value_type{0} );
		
	}

	for(auto const& n : extents) {

			auto a  = vector_type(product(n));

			size_t* wa = nullptr;

			BOOST_REQUIRE_THROW( ublas::accumulate( n.size(), n.data(), a.data(), wa, value_type{0} ), std::runtime_error );
			
	}

	for(auto const& n : extents) {

			auto a  = vector_type(product(n));

			auto wa = ublas::strides_t<ublas::dynamic_extents<>,ublas::first_order>(n);

			size_t* m = nullptr;

			BOOST_REQUIRE_THROW( ublas::accumulate( n.size(), m, a.data(), wa.data(), value_type{0}, [](value_type const& a,value_type const& b){ return a + b;} ), std::runtime_error );
			
	}

}


template<class V>
void init(std::vector<V>& a)
{
	auto v = V(1);
	for(auto i = 0u; i < a.size(); ++i, ++v){
		a[i] = v;
	}
}

template<class V>
void init(std::vector<std::complex<V>>& a)
{
	auto v = std::complex<V>(1,1);
	for(auto i = 0u; i < a.size(); ++i){
		a[i] = v;
		v.real(v.real()+1);
		v.imag(v.imag()+1);
	}
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_algorithms_trans, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using vector_type  = std::vector<value_type>;
	using extents_type = ublas::dynamic_extents<>;
	using strides_type = ublas::strides_t<extents_type,layout_type>;
	using size_type = typename extents_type::value_type;
	using permutation_type = std::vector<size_type>;


	for(auto const& n : extents) {

		auto p   = n.size();
		auto s   = product(n);

		auto pi  = permutation_type(p);
		auto a   = vector_type(s);
		auto b1  = vector_type(s);
		auto b2  = vector_type(s);
		auto c1  = vector_type(s);
		auto c2  = vector_type(s);

		auto wa = strides_type(n);

		init(a);

		// so wie last-order.
		for(auto i = size_type(0), j = p; i < n.size(); ++i, --j)
			pi[i] = j;

		auto nc = typename extents_type::base_type (p);
		for(auto i = 0u; i < p; ++i)
			nc[pi[i]-1] = n[i];

		auto wc = strides_type(extents_type(nc));
		auto wc_pi = typename strides_type::base_type (p);
		for(auto i = 0u; i < p; ++i)
			wc_pi[pi[i]-1] = wc[i];

		ublas::copy ( p, n.data(),            c1.data(), wc_pi.data(), a.data(), wa.data());
		ublas::trans( p, n.data(), pi.data(), c2.data(), wc.data(),    a.data(), wa.data() );

		if(!std::is_compound_v<value_type>)
			for(auto i = 0ul; i < s; ++i)
				BOOST_CHECK_EQUAL( c1[i], c2[i] );


		auto nb = typename extents_type::base_type (p);
		for(auto i = 0u; i < p; ++i)
			nb[pi[i]-1] = nc[i];

		auto wb = strides_type (extents_type(nb));
		auto wb_pi = typename strides_type::base_type (p);
		for(auto i = 0u; i < p; ++i)
			wb_pi[pi[i]-1] = wb[i];

		ublas::copy ( p, nc.data(),            b1.data(), wb_pi.data(), c1.data(), wc.data());
		ublas::trans( p, nc.data(), pi.data(), b2.data(), wb.data(),    c2.data(), wc.data() );

		if(!std::is_compound_v<value_type>)
			for(auto i = 0ul; i < s; ++i)
				BOOST_CHECK_EQUAL( b1[i], b2[i] );

		for(auto i = 0ul; i < s; ++i)
			BOOST_CHECK_EQUAL( a[i], b2[i] );

		size_type zero = 0;
                ublas::trans( zero, n.data(), pi.data(), c2.data(), wc.data(), a.data(), wa.data() );
                ublas::trans( zero, nc.data(), pi.data(), b2.data(), wb.data(), c2.data(), wc.data() );

                value_type *c0 = nullptr;
                size_type const*const s0 = nullptr;

                BOOST_CHECK_THROW(ublas::trans( p, n.data(), pi.data(), c0, wc.data(),  a.data(), wa.data()), std::runtime_error);
                BOOST_CHECK_THROW(ublas::trans( p, s0, pi.data(), c2.data(),wc.data(),  a.data(), wa.data()), std::runtime_error);
                BOOST_CHECK_THROW(ublas::trans( p, n.data(), pi.data(), c2.data(), s0,  a.data(), wa.data()), std::runtime_error);
                BOOST_CHECK_THROW(ublas::trans( p, n.data(), s0, c2.data(), wc.data(),  a.data(), wa.data()), std::runtime_error);

        }
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_tensor_algorithms_trans_exceptions, value,  test_types, fixture )
{
	using namespace boost::numeric;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
	using vector_type  = std::vector<value_type>;
	using extents_type = ublas::dynamic_extents<>;
	using strides_type = ublas::strides_t<extents_type, layout_type>;
	using size_type = typename extents_type::value_type;
	using permutation_type = std::vector<size_type>;

	for(auto const& n : extents) {

		auto p   = n.size();
		auto s   = product(n);

		auto pi  = permutation_type(p);
		value_type* a   = nullptr;
		auto c  = vector_type(s);

		auto wa = strides_type(n);

		auto nc = typename extents_type::base_type (p);
		auto wc = strides_type(n);
		auto wc_pi = typename strides_type::base_type (p);

		BOOST_REQUIRE_THROW( ublas::trans( p, nc.data(), pi.data(), a, wa.data(),    c.data(), wc.data() ), std::runtime_error );
		
	}

	for(auto const& n : extents) {

		auto p   = n.size();
		auto s   = product(n);

		auto pi  = permutation_type(p);
		value_type* a   = nullptr;
		auto c  = vector_type(s);

		auto wa = strides_type(n);
		auto nc = typename extents_type::base_type (p);
			
		auto wc = strides_type(n);

		BOOST_REQUIRE_THROW( ublas::trans( p, nc.data(), pi.data(), c.data(), wc.data(),    a, wa.data() ), std::runtime_error );
		
	}

	for(auto const& n : extents) {

		auto p   = n.size();

		auto pi  = permutation_type(p);
		value_type* a   = nullptr;
		value_type* c   = nullptr;

		auto wa = strides_type(n);
		auto nc = typename extents_type::base_type (p);
			
		auto wc = strides_type(n);

		BOOST_REQUIRE_THROW( ublas::trans( p, nc.data(), pi.data(), c, wc.data(),    a, wa.data() ), std::runtime_error );
		
	}

	for(auto const& n : extents) {

		auto p   = n.size();
		auto s   = product(n);

		auto pi  = permutation_type(p);
		auto a  = vector_type(s);
		auto c  = vector_type(s);

		auto wa = strides_type(n);

		auto nc = typename extents_type::base_type (p);
			
		size_t* wc = nullptr;

		BOOST_REQUIRE_THROW( ublas::trans( p, nc.data(), pi.data(), c.data(), wc,    a.data(), wa.data() ), std::runtime_error );
		
	}

	for(auto const& n : extents) {

		auto p   = n.size();
		auto s   = product(n);

		auto pi  = permutation_type(p);
		auto a  = vector_type(s);
		auto c  = vector_type(s);

		auto wc = strides_type(n);
		auto nc = typename extents_type::base_type (p);
			
		size_t* wa = nullptr;

		BOOST_REQUIRE_THROW( ublas::trans( p, nc.data(), pi.data(), c.data(), wc.data(),    a.data(), wa ), std::runtime_error );
		
	}

	for(auto const& n : extents) {

		auto p   = n.size();
		auto s   = product(n);

		auto pi  = permutation_type(p);
		auto a  = vector_type(s);
		auto c  = vector_type(s);

		size_t* wc = nullptr;

		auto nc = typename extents_type::base_type (p);
			
		size_t* wa = nullptr;

		BOOST_REQUIRE_THROW( ublas::trans( p, nc.data(), pi.data(), c.data(), wc,    a.data(), wa ), std::runtime_error );
		
	}

	for(auto const& n : extents) {

		auto p   = n.size();
		auto s   = product(n);

		size_type* pi  = nullptr;
		auto a  = vector_type(s);
		auto c  = vector_type(s);

		auto wa = strides_type(n);

		auto nc = typename extents_type::base_type (p);
		auto wc = strides_type(n);

		BOOST_REQUIRE_THROW( ublas::trans( p, nc.data(), pi, c.data(), wc.data(),    a.data(), wa.data() ), std::runtime_error );
		
	}

	for(auto const& n : extents) {

		auto p   = n.size();
		auto s   = product(n);

		auto pi  = permutation_type(p);
		auto a  = vector_type(s);
		auto c  = vector_type(s);

		auto wa = strides_type(n);
		size_t* nc = nullptr;
			
		auto wc = strides_type(n);

		BOOST_REQUIRE_THROW( ublas::trans( p, nc, pi.data(), c.data(), wc.data(),    a.data(), wa.data() ), std::runtime_error );
		
	}

	for(auto const& n : extents) {

		size_type p   = 1;
		auto s   = product(n);

		auto pi  = permutation_type(p);
		auto a  = vector_type(s);
		auto c  = vector_type(s);

		auto wa = strides_type(n);
		auto nc = typename extents_type::base_type (p);
			
		auto wc = strides_type(n);

		ublas::trans( p, nc.data(), pi.data(), c.data(), wc.data(),    a.data(), wa.data() );
		
	}

}


BOOST_AUTO_TEST_SUITE_END()
