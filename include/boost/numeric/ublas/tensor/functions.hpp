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


#ifndef _BOOST_UBLAS_TENSOR_FUNCTIONS_HPP_
#define _BOOST_UBLAS_TENSOR_FUNCTIONS_HPP_


#include <stdexcept>
#include <vector>
#include <algorithm>
#include <numeric>

#include <boost/numeric/ublas/tensor/multiplication.hpp>
#include <boost/numeric/ublas/tensor/algorithms.hpp>
#include <boost/numeric/ublas/tensor/expression.hpp>
#include <boost/numeric/ublas/tensor/expression_evaluation.hpp>
#include <boost/numeric/ublas/tensor/detail/type_traits.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>
#include <boost/numeric/ublas/tensor/strides.hpp>

// dynamic_functions
namespace boost::numeric::ublas
{

	namespace detail{
		// MSVC 14.1 bug
		// cannot use ::_size inside constexpr function
		// basic_extents does not contains _size member because of which
		// it complains about it

		template<typename E>
		constexpr auto extents_result_type_tensor_times_vector( E const& ){
			static_assert(is_static_rank<E>::value, 
				"boost::numeric::ublas::extents_result_type_tensor_times_vector() : invalid type, type should be an extents");
			using size_type = typename E::size_type;
			auto ret = dynamic_extents< std::max( size_type( E::_size - 1 ), size_type(2) )>();
			ret.fill(typename E::value_type(1));
			return ret;
		}

		template<typename T>
		constexpr auto extents_result_type_tensor_times_vector( basic_extents<T> const& e ){
			using size_type = typename basic_extents<T>::size_type;
			return dynamic_extents<>{ typename dynamic_extents<>::base_type(std::max( size_type( e.size() - 1), size_type(2) ),1) } ;
		}
		
		template<typename E>
		constexpr auto extents_result_type_tensor_times_matrix( E const& a ){
			static_assert(is_static_rank<E>::value, 
				"boost::numeric::ublas::extents_result_type_tensor_times_matrix() : invalid type, type should be an extents");
			return dynamic_extents<E::_size>(a);
		}

		template<typename T>
		constexpr auto extents_result_type_tensor_times_matrix( basic_extents<T> const& e ){
			return dynamic_extents<>{ e } ;
		}

		template<typename T, T... E1, T... E2>
		constexpr auto extents_result_type_outer_prod( basic_static_extents<T,E1...> const&, basic_static_extents<T,E2...> const& ){
			return dynamic_extents<sizeof...(E1) + sizeof...(E2)>();
		}

		template<typename T, T... E, size_t R>
		constexpr auto extents_result_type_outer_prod( basic_static_extents<T,E...> const&, basic_fixed_rank_extents<T,R> const& ){
			return dynamic_extents<sizeof...(E) + R>();
		}

		template<typename T, T... E, size_t R>
		constexpr auto extents_result_type_outer_prod( basic_fixed_rank_extents<T,R> const&, basic_static_extents<T,E...> const& ){
			return dynamic_extents<R + sizeof...(E)>();
		}

		template<typename E1, typename E2>
		auto extents_result_type_outer_prod( E1 const& e1, E2 const& e2){
			return dynamic_extents<>( std::vector<typename E1::value_type>( e1.size() + e2.size(), 1 ) );
		}

	} // namespace detail
	
	/** @brief Computes the m-mode tensor-times-vector product
	 *
	 * Implements C[i1,...,im-1,im+1,...,ip] = A[i1,i2,...,ip] * b[im]
	 *
	 * @note calls ublas::ttv
	 *
	 * @param[in] m contraction dimension with 1 <= m <= p
	 * @param[in] a tensor object A with order p
	 * @param[in] b vector object B
	 *
	 * @returns tensor object C with order p-1, the same storage format and allocator type as A
	*/
	template <class V, class E, class F, class A1, class A2>
	inline decltype(auto) prod(tensor<V, E, F, A1> const &a, vector<V, A2> const &b, const std::size_t m)
	{

		using tensor_type = tensor<V, E, F, A1>;
		using extents_type = typename tensor_type::extents_type;
		using value_type = typename tensor_type::value_type;
		using size_type = typename extents_type::size_type;

		auto const p = std::size_t(a.rank());

		if (m == 0)
			throw std::length_error(
				"error in boost::numeric::ublas::prod(ttv): "
				"contraction mode must be greater than zero.");

		if (p < m)
			throw std::length_error(
				"error in boost::numeric::ublas::prod(ttv): rank of tensor must be "
				"greater than or equal to the modus.");

		if (a.empty())
			throw std::length_error(
				"error in boost::numeric::ublas::prod(ttv): first "
				"argument tensor should not be empty.");

		if (b.size() == 0)
			throw std::length_error(
				"error in boost::numeric::ublas::prod(ttv): second "
				"argument vector should not be empty.");

		auto nc = detail::extents_result_type_tensor_times_vector(a.extents());
		auto nb = std::vector<typename extents_type::value_type>{b.size(), 1};
		auto a_extents = a.extents();
		for (auto i = size_type(0), j = size_type(0); i < p; ++i)
			if (i != m - 1)
				nc[j++] = a_extents.at(i);


		auto c = tensor<value_type, decltype(nc), F, std::vector<V> >(nc, value_type{});
		auto bb = &(b(0));
		ttv(m, p,
			c.data(), c.extents().data(), c.strides().data(),
			a.data(), a.extents().data(), a.strides().data(),
			bb, nb.data(), nb.data());
		return c;
	}

	/** @brief Computes the m-mode tensor-times-matrix product
	 *
	 * Implements C[i1,...,im-1,j,im+1,...,ip] = A[i1,i2,...,ip] * B[j,im]
	 *
	 * @note calls ublas::ttm
	 *
	 * @param[in] a tensor object A with order p
	 * @param[in] b vector object B
	 * @param[in] m contraction dimension with 1 <= m <= p
	 *
	 * @returns tensor object C with order p, the same storage format and allocator type as A
	*/
	template <class V, class E, class F, class A1, class A2>
	inline decltype(auto) prod(tensor<V, E, F, A1> const &a, matrix<V, F, A2> const &b, const std::size_t m)
	{
		using tensor_type = tensor<V, E, F, A1>;
		using dynamic_strides_type = strides_t<dynamic_extents<>,F>;
		using value_type = typename tensor_type::value_type;

		auto const p = a.rank();

		if (m == 0)
			throw std::length_error(
				"error in boost::numeric::ublas::prod(ttm): "
				"contraction mode must be greater than zero.");

		if (p < m || m > a.extents().size())
			throw std::length_error(
				"error in boost::numeric::ublas::prod(ttm): rank "
				"of the tensor must be greater equal the modus.");

		if (a.empty())
			throw std::length_error(
				"error in boost::numeric::ublas::prod(ttm): first "
				"argument tensor should not be empty.");

		if (b.size1() * b.size2() == 0)
			throw std::length_error(
				"error in boost::numeric::ublas::prod(ttm): second "
				"argument matrix should not be empty.");

		auto nc = detail::extents_result_type_tensor_times_matrix(a.extents());
		auto nb = dynamic_extents<>{b.size1(), b.size2()};

		auto wb = dynamic_strides_type(nb);

		nc[m - 1] = nb[0];

		auto c = tensor(nc, value_type{});

		auto bb = &(b(0, 0));
		ttm(m, p,
			c.data(), c.extents().data(), c.strides().data(),
			a.data(), a.extents().data(), a.strides().data(),
			bb, nb.data(), wb.data());

		return c;
	}

	/** @brief Computes the q-mode tensor-times-tensor product
	 *
	 * Implements C[i1,...,ir,j1,...,js] = sum( A[i1,...,ir+q] * B[j1,...,js+q]  )
	 *
	 * @note calls ublas::ttt
	 *
	 * na[phia[x]] = nb[phib[x]] for 1 <= x <= q
	 *
	 * @param[in]	 phia one-based permutation tuple of length q for the first
	 * input tensor a
	 * @param[in]	 phib one-based permutation tuple of length q for the second
	 * input tensor b
	 * @param[in]  a  left-hand side tensor with order r+q
	 * @param[in]  b  right-hand side tensor with order s+q
	 * @result     tensor with order r+s
	*/
	template <class V, class E1, class E2, class F, class A1, class A2>
	inline decltype(auto) prod(tensor<V, E1, F, A1> const &a, tensor<V, E2, F, A2> const &b,
										std::vector<std::size_t> const &phia, std::vector<std::size_t> const &phib)
	{

		using tensor_type = tensor<V, E1, F, A1>;
		using extents_type_1 = E1;
		using value_type = typename tensor_type::value_type;
		using size_type = typename extents_type_1::value_type;

		auto const pa = a.rank();
		auto const pb = b.rank();

		auto const q = size_type(phia.size());

		if (pa == 0ul)
			throw std::runtime_error("error in ublas::prod: order of left-hand side tensor must be greater than 0.");
		if (pb == 0ul)
			throw std::runtime_error("error in ublas::prod: order of right-hand side tensor must be greater than 0.");
		if (pa < q)
			throw std::runtime_error("error in ublas::prod: number of contraction dimensions cannot be greater than the order of the left-hand side tensor.");
		if (pb < q)
			throw std::runtime_error("error in ublas::prod: number of contraction dimensions cannot be greater than the order of the right-hand side tensor.");

		if (q != phib.size())
			throw std::runtime_error("error in ublas::prod: permutation tuples must have the same length.");

		if (pa < phia.size())
			throw std::runtime_error("error in ublas::prod: permutation tuple for the left-hand side tensor cannot be greater than the corresponding order.");
		if (pb < phib.size())
			throw std::runtime_error("error in ublas::prod: permutation tuple for the right-hand side tensor cannot be greater than the corresponding order.");

		auto const &na = a.extents();
		auto const &nb = b.extents();

		for (auto i = 0ul; i < q; ++i)
			if (na.at(phia.at(i) - 1) != nb.at(phib.at(i) - 1))
				throw std::runtime_error("error in ublas::prod: permutations of the extents are not correct.");

		auto const r = pa - q;
		auto const s = pb - q;

		std::vector<std::size_t> phia1(pa), phib1(pb);
		std::iota(phia1.begin(), phia1.end(), 1ul);
		std::iota(phib1.begin(), phib1.end(), 1ul);

		std::vector<std::size_t> nc(std::max(r + s, size_type(2)), size_type(1));

		for (auto i = 0ul; i < phia.size(); ++i)
			*std::remove(phia1.begin(), phia1.end(), phia.at(i)) = phia.at(i);

		//phia1.erase( std::remove(phia1.begin(), phia1.end(), phia.at(i)),  phia1.end() )  ;

		assert(phia1.size() == pa);

		for (auto i = 0ul; i < r; ++i)
			nc[i] = na[phia1[i] - 1];

		for (auto i = 0ul; i < phib.size(); ++i)
			*std::remove(phib1.begin(), phib1.end(), phib.at(i)) = phib.at(i);
		//phib1.erase( std::remove(phib1.begin(), phib1.end(), phia.at(i)), phib1.end() )  ;

		assert(phib1.size() == pb);

		for (auto i = 0ul; i < s; ++i)
			nc[r + i] = nb[phib1[i] - 1];

		//	std::copy( phib.begin(), phib.end(), phib1.end()  );

		assert(phia1.size() == pa);
		assert(phib1.size() == pb);

		auto c = tensor(dynamic_extents<>(nc), value_type{});
		
		ttt(pa, pb, q,
			phia1.data(), phib1.data(),
			c.data(), c.extents().data(), c.strides().data(),
			a.data(), a.extents().data(), a.strides().data(),
			b.data(), b.extents().data(), b.strides().data());

		return c;
	}

	// template<class V, class F, class A1, class A2, std::size_t N, std::size_t M>
	// auto operator*( tensor_index<V,F,A1,N> const& lhs, tensor_index<V,F,A2,M>
	// const& rhs)

	/** @brief Computes the q-mode tensor-times-tensor product
	 *
	 * Implements C[i1,...,ir,j1,...,js] = sum( A[i1,...,ir+q] * B[j1,...,js+q]  )
	 *
	 * @note calls ublas::ttt
	 *
	 * na[phi[x]] = nb[phi[x]] for 1 <= x <= q
	 *
	 * @param[in]	 phi one-based permutation tuple of length q for bot input
	 * tensors
	 * @param[in]  a  left-hand side tensor with order r+q
	 * @param[in]  b  right-hand side tensor with order s+q
	 * @result     tensor with order r+s
	*/
	template <class V, class E1, class E2, class F, class A1, class A2>
	inline decltype(auto) prod(tensor<V, E1, F, A1> const &a, tensor<V, E2, F, A2> const &b,
										std::vector<std::size_t> const &phi)
	{
		return prod(a, b, phi, phi);
	}

	/** @brief Computes the inner product of two tensors
	 *
	 * Implements c = sum(A[i1,i2,...,ip] * B[i1,i2,...,jp])
	 *
	 * @note calls inner function
	 *
	 * @param[in] a tensor object A
	 * @param[in] b tensor object B
	 *
	 * @returns a value type.
	*/
	template <class V, class E1, class E2, class F, class A1, class A2>
	inline decltype(auto) inner_prod(tensor<V, E1, F, A1> const &a, tensor<V, E2, F, A2> const &b)
	{
		using value_type = typename tensor<V, E1, F, A1>::value_type;

		if (a.rank() != b.rank())
			throw std::length_error("error in boost::numeric::ublas::inner_prod: Rank of both tensors must be the same.");

		if (a.empty() || b.empty())
			throw std::length_error("error in boost::numeric::ublas::inner_prod: Tensors should not be empty.");

		if (a.extents() != b.extents())
			throw std::length_error("error in boost::numeric::ublas::inner_prod: Tensor extents should be the same.");
		
		return inner(a.rank(), a.extents().data(),
					a.data(), a.strides().data(),
					b.data(), b.strides().data(), value_type{0});
	}

	/** @brief Computes the outer product of two tensors
	 *
	 * Implements C[i1,...,ip,j1,...,jq] = A[i1,i2,...,ip] * B[j1,j2,...,jq]
	 *
	 * @note calls outer function
	 *
	 * @param[in] a tensor object A
	 * @param[in] b tensor object B
	 *
	 * @returns tensor object C with the same storage format F and allocator type A1
	*/
	template <class V, class E1, class E2, class F, class A1, class A2,
		std::enable_if_t<
			!( detail::is_static<E1>::value &&
			detail::is_static<E2>::value )
			,int> = 0
	>
	inline decltype(auto) outer_prod(tensor<V, E1, F, A1> const &a, tensor<V, E2, F, A2> const &b)
	{

		if (a.empty() || b.empty())
			throw std::runtime_error(
				"error in boost::numeric::ublas::outer_prod: "
				"tensors should not be empty.");

		auto nc = detail::extents_result_type_outer_prod(a.extents(), b.extents());

		auto a_extents = a.extents();
		auto b_extents = b.extents();

		for(auto i = 0u; i < a.rank(); ++i)
			nc.at(i) = a_extents.at(i);

		for(auto i = 0u; i < b.rank(); ++i)
			nc.at(a.rank()+i) = b_extents.at(i);
		
		auto c = tensor(nc, V{});

		outer(c.data(), c.rank(), c.extents().data(), c.strides().data(),
			a.data(), a.rank(), a_extents.data(), a.strides().data(),
			b.data(), b.rank(), b_extents.data(), b.strides().data());

		return c;
	}

	/** @brief Transposes a tensor according to a permutation tuple
	 *
	 * Implements C[tau[i1],tau[i2]...,tau[ip]] = A[i1,i2,...,ip]
	 *
	 * @note calls trans function
	 *
	 * @param[in] a    tensor object of rank p
	 * @param[in] tau  one-based permutation tuple of length p
	 * @returns        a transposed tensor object with the same storage format F and allocator type A
	*/
	template <class V, class E, class F, class A>
	inline decltype(auto) trans(tensor<V, E, F, A> const &a, std::vector<std::size_t> const &tau)
	{
		using result_tensor_type = tensor<V, dynamic_extents<>, F, std::vector<V>>;
		//	using strides_type = typename tensor_type::strides_type;

		if (a.empty())
			return result_tensor_type{};

		auto const p = a.rank();
		auto const &na = a.extents();

		auto nc = typename dynamic_extents<>::base_type(p);
		for (auto i = 0u; i < p; ++i)
			nc.at(tau.at(i) - 1) = na.at(i);

		//	auto wc = strides_type(extents_type(nc));
		auto c = result_tensor_type(dynamic_extents<>(nc));

		trans(a.rank(), a.extents().data(), tau.data(),
			c.data(), c.strides().data(),
			a.data(), a.strides().data());

		//	auto wc_pi = typename strides_type::base_type (p);
		//	for(auto i = 0u; i < p; ++i)
		//		wc_pi.at(tau.at(i)-1) = c.strides().at(i);

		// copy(a.rank(),
		//		 a.extents().data(),
		//		 c.data(), wc_pi.data(),
		//		 a.data(), a.strides().data() );

		return c;
	}
	/**
	 *
	 * @brief Computes the frobenius nor of a tensor
	 *
	 * @note Calls accumulate on the tensor.
	 *
	 * implements
	 * k = sqrt( sum_(i1,...,ip) A(i1,...,ip)^2 )
	 *
	 * @tparam V the data type of tensor
	 * @tparam F the format of tensor storage
	 * @tparam A the array_type of tensor
	 * @param a the tensor whose norm is expected of rank p.
	 * @return the frobenius norm of a tensor.
	 */
	template <class V, class E, class F, class A>
	inline decltype(auto) norm(tensor<V, E, F, A> const &a)
	{
		using tensor_type = tensor<V, E, F, A>;
		using value_type = typename tensor_type::value_type;
		
		static_assert(std::is_default_constructible<V>::value,
					"Value type of tensor must be default construct able in order "
					"to call boost::numeric::ublas::norm");
		if (a.empty())
		{
			throw std::runtime_error(
				"error in boost::numeric::ublas::norm: tensors should not be empty.");
		}

		return std::sqrt(accumulate(a.order(), a.extents().data(), a.data(), a.strides().data(), value_type{},
									[](auto const &l, auto const &r) { return l + r * r; }));
	}


	/** @brief Computes the complex conjugate component of tensor elements within a tensor expression
	 *
	 * @param[in] lhs tensor expression
	 * @returns   unary tensor expression
	*/
	template<class V, class E, class A, class F, class D>
	auto conj(detail::tensor_expression<tensor<std::complex<V>,E,F,A>,D> const& expr)
	{
		return detail::make_unary_tensor_expression<tensor<std::complex<V>,E,F,A>> (expr(), [] (auto const& l) { return std::conj( l ); } );
	}

	/** @brief Computes the complex conjugate component of tensor elements within a tensor expression
	 *
	 * @param[in] expr tensor expression
	 * @returns   complex tensor
	*/
	template<class T, class D>
	auto conj(detail::tensor_expression<T,D> const& expr)
	{
		using tensor_type = T;
		using value_type = typename tensor_type::value_type;
		using extents_type = typename tensor_type::extents_type;
		using layout_type = typename tensor_type::layout_type;
		using array_type = typename tensor_type::array_type;

		using new_value_type = std::complex<value_type>;
		using new_array_type = typename storage_traits<array_type>::template rebind<new_value_type>;

		using tensor_complex_type = tensor<new_value_type,extents_type,layout_type, new_array_type>;

		if( detail::retrieve_extents( expr  ).empty() )
			throw std::runtime_error("error in boost::numeric::ublas::conj: tensors should not be empty.");

		auto a = tensor_type( expr );
		auto c = tensor_complex_type( a.extents() );

		std::transform( a.begin(), a.end(),  c.begin(), [](auto const& l){ return std::conj(l) ; }  );

		return c;
	}

	/** @brief Extract the real component of tensor elements within a tensor expression
	 *
	 * @param[in] lhs tensor expression
	 * @returns   unary tensor expression
	*/
	template<class T, class D>
	auto real(detail::tensor_expression<T,D> const& expr) {
		return detail::make_unary_tensor_expression<T> (expr(), [] (auto const& l) { return std::real( l ); } );
	}

	/** @brief Extract the real component of tensor elements within a tensor expression
	 *
	 * @param[in] lhs tensor expression
	 * @returns   unary tensor expression
	*/
	template<class V, class E, class A, class F, class D>
	auto real(detail::tensor_expression<tensor<std::complex<V>,E,F,A>,D> const& expr)
	{
		using tensor_complex_type = tensor<std::complex<V>,E,F,A>;
		using tensor_type = tensor<V,E,F,typename storage_traits<A>::template rebind<V>>;

		if( detail::retrieve_extents( expr  ).empty() )
			throw std::runtime_error("error in boost::numeric::ublas::real: tensors should not be empty.");

		auto a = tensor_complex_type( expr );
		auto c = tensor_type( a.extents() );

		std::transform( a.begin(), a.end(),  c.begin(), [](auto const& l){ return std::real(l) ; }  );

		return c;
	}


	/** @brief Extract the imaginary component of tensor elements within a tensor expression
	 *
	 * @param[in] lhs tensor expression
	 * @returns   unary tensor expression
	*/
	template<class T, class D>
	auto imag(detail::tensor_expression<T,D> const& lhs) {
		return detail::make_unary_tensor_expression<T> (lhs(), [] (auto const& l) { return std::imag( l ); } );
	}


	/** @brief Extract the imag component of tensor elements within a tensor expression
	 *
	 * @param[in] lhs tensor expression
	 * @returns   unary tensor expression
	*/
	template<class V, class E, class A, class F, class D>
	auto imag(detail::tensor_expression<tensor<std::complex<V>,E,F,A>,D> const& expr)
	{
		using tensor_complex_type = tensor<std::complex<V>,E,F,A>;
		using tensor_type = tensor<V,E,F,typename storage_traits<A>::template rebind<V>>;

		if( detail::retrieve_extents( expr  ).empty() )
			throw std::runtime_error("error in boost::numeric::ublas::real: tensors should not be empty.");

		auto a = tensor_complex_type( expr );
		auto c = tensor_type( a.extents() );

		std::transform( a.begin(), a.end(),  c.begin(), [](auto const& l){ return std::imag(l) ; }  );

		return c;
	}

}

// static functions
namespace boost::numeric::ublas
{

	namespace detail{

		template<size_t M, size_t I, typename T, T... E, T... R>
		inline
		constexpr auto extents_result_type_tensor_times_vector(basic_static_extents<T>, 
			basic_static_extents<T, E...>, basic_static_extents<T, R...>)
		{
			return basic_static_extents<T, R..., E...>{};
		}

		template<size_t M, size_t I, typename T, T E0, T... E, T O0, T... OtherE, T... R>
		inline
		constexpr auto extents_result_type_tensor_times_vector(basic_static_extents<T,E0,E...>, 
			basic_static_extents<T, O0, OtherE...>, basic_static_extents<T, R...> = basic_static_extents<T>{})
		{
			if constexpr(I != M - 1){
				return extents_result_type_tensor_times_vector<M,I + 1> 
					( basic_static_extents<T,E...>{}, basic_static_extents<T,OtherE...>{}, basic_static_extents<T, R..., E0>{} );
			}else{
				return extents_result_type_tensor_times_vector<M,I + 1>
					( basic_static_extents<T,E...>{}, basic_static_extents<T,O0,OtherE...>{}, basic_static_extents<T, R...>{} );
			}
		}


		template<size_t M, typename T, T E0, T... E>
		inline
		constexpr auto extents_result_type_tensor_times_vector(basic_static_extents<T,E0,E...> const& e)
		{
			using size_type = typename basic_static_extents<T>::size_type;
			auto ones = typename impl::make_sequence_of_ones_t< T, std::max( size_type(2), sizeof...(E) ) >::extents_type{};
			return extents_result_type_tensor_times_vector<M,0>(e, ones);
		}

		template<size_t I, size_t NE, typename T, T E0, T... E, T... OtherE>
		inline
		constexpr auto static_extents_set_at
			( basic_static_extents<T,E0,E...> const&, basic_static_extents<T,OtherE...> = basic_static_extents<T>{}){
			static_assert( I < sizeof...(E) + 1, "boost::numeric::ublas::detail::static_extents_set_at(): out of bound");
			if constexpr( sizeof...(E) == 0 ){
				if constexpr( I == 0 ){
					return basic_static_extents<T,OtherE..., NE>{};
				}else{
					return basic_static_extents<T,OtherE...,E0>{};
				}
			}else{
				if constexpr(I == 0){
					return basic_static_extents<T,OtherE..., NE, E...>{};
				}else{
					return static_extents_set_at<I - 1, NE>( basic_static_extents<T,E...>{}, basic_static_extents<T,OtherE..., E0>{} );
				}
			}
		}

		namespace impl{
			template<typename T, T... E1, T... E2>
			struct concat< basic_static_extents<T, E1...>, basic_static_extents<T, E2...> >{
				using type = basic_static_extents<T, E1..., E2...>;
			};
		}

	} // namespace detail
	
	/** @brief Computes the m-mode tensor-times-vector product
	 *
	 * Implements C[i1,...,im-1,im+1,...,ip] = A[i1,i2,...,ip] * b[im]
	 *
	 * @note calls ublas::ttv
	 *
	 * @tparam 	  m contraction dimension with 1 <= m <= p
	 * @param[in] a tensor object A with order p
	 * @param[in] b vector object B
	 *
	 * @returns tensor object C with order p-1, the same storage format and allocator type as A
	*/
	template <size_t M, class V, class E, class F, class A1, class A2, 
		std::enable_if_t<detail::is_static<E>::value,int> = 0
	>
	inline decltype(auto) prod(tensor<V, E, F, A1> const &a, vector<V, A2> const &b)
	{

		using tensor_type = tensor<V, E, F, A1>;
		using extents_type = typename tensor_type::extents_type;
		using value_type = typename tensor_type::value_type;

		auto const p = std::size_t(a.rank());

		static_assert( M != 0, 
				"error in boost::numeric::ublas::prod(ttv): "
				"contraction mode must be greater than zero.");

		static_assert( extents_type::_size >= M,
				"error in boost::numeric::ublas::prod(ttv): rank of tensor must be "
				"greater than or equal to the modus.");

		static_assert(extents_type::_size != 0,
				"error in boost::numeric::ublas::prod(ttv): first "
				"argument tensor should not be empty.");

		if (b.size() == 0)
			throw std::length_error(
				"error in boost::numeric::ublas::prod(ttv): second "
				"argument vector should not be empty.");

		auto nc = detail::extents_result_type_tensor_times_vector<M>(a.extents());
		auto nb = std::vector<typename extents_type::value_type>{b.size(), 1};

		auto c = tensor(nc, value_type{});
		auto bb = &(b(0));

		auto& a_static_extents = a.extents().base();
		auto& c_static_extents = c.extents().base();

		auto& a_static_strides = a.strides().base();
		auto& c_static_strides = c.strides().base();

		ttv(M, p,
			c.data(), c_static_extents.data(), c_static_strides.data(),
			a.data(), a_static_extents.data(), a_static_strides.data(),
			bb, nb.data(), nb.data());

		return c;
	}

	/** @brief Computes the m-mode tensor-times-matrix product
	 *
	 * Implements C[i1,...,im-1,j,im+1,...,ip] = A[i1,i2,...,ip] * B[j,im]
	 *
	 * @note calls ublas::ttm
	 *
	 * @tparam    m contraction dimension with 1 <= m <= p
	 * @param[in] a tensor object A with order p
	 * @param[in] b vector object B
	 *
	 * @returns tensor object C with order p, the same storage format and allocator type as A
	*/
	template <size_t M, size_t MatricRow, class V, class E, class F, class A1, class A2,
		std::enable_if_t<detail::is_static<E>::value,int> = 0
	>
	inline decltype(auto) prod(tensor<V, E, F, A1> const &a, matrix<V, F, A2> const &b)
	{
		using tensor_type = tensor<V, E, F, A1>;
		using dynamic_strides_type = strides_t<dynamic_extents<>,F>;
		using value_type = typename tensor_type::value_type;

		auto const p = a.rank();

		static_assert(M != 0,
				"error in boost::numeric::ublas::prod(ttm): "
				"contraction mode must be greater than zero.");

		static_assert( E::_size >= M ,
				"error in boost::numeric::ublas::prod(ttm): rank "
				"of the tensor must be greater equal the modus.");

		static_assert(E::_size,
				"error in boost::numeric::ublas::prod(ttm): first "
				"argument tensor should not be empty.");

		if (b.size1() * b.size2() == 0)
			throw std::length_error(
				"error in boost::numeric::ublas::prod(ttm): second "
				"argument matrix should not be empty.");

		auto nc = detail::static_extents_set_at< M - 1, MatricRow >( a.extents() );
		auto nb = dynamic_extents<>{b.size1(), b.size2()};

		auto wb = dynamic_strides_type(nb);

		auto c = tensor(nc, value_type{});

		auto bb = &(b(0, 0));

		auto& a_static_extents = a.extents().base();
		auto& c_static_extents = c.extents().base();

		auto& a_static_strides = a.strides().base();
		auto& c_static_strides = c.strides().base();
		ttm(M, p,
			c.data(), c_static_extents.data(), c_static_strides.data(),
			a.data(), a_static_extents.data(), a_static_strides.data(),
			bb, nb.data(), wb.data());

		return c;
	}

	/** @brief Computes the outer product of two tensors
	 *
	 * Implements C[i1,...,ip,j1,...,jq] = A[i1,i2,...,ip] * B[j1,j2,...,jq]
	 *
	 * @note calls outer function
	 *
	 * @param[in] a tensor object A
	 * @param[in] b tensor object B
	 *
	 * @returns tensor object C with the same storage format F and allocator type A1
	*/
	template <class V, class E1, class E2, class F, class A1, class A2,
		std::enable_if_t<
			detail::is_static<E1>::value &&
			detail::is_static<E2>::value
			,int> = 0
	>
	inline decltype(auto) outer_prod(tensor<V, E1, F, A1> const &a, tensor<V, E2, F, A2> const &b)
	{
		if (a.empty() || b.empty())
			throw std::runtime_error(
				"error in boost::numeric::ublas::outer_prod: "
				"tensors should not be empty.");
		using extents_type1 = std::decay_t< decltype(a.extents()) >;
		using extents_type2 = std::decay_t< decltype(b.extents()) >;
		auto nc = detail::impl::concat_t<extents_type1, extents_type2>{};

		auto a_extents = a.extents();
		auto b_extents = b.extents();

		
		auto c = tensor(nc, V{});

		auto& a_static_extents = a_extents.base();
		auto& a_static_strides = a.strides().base();
		
		auto& b_static_extents = b_extents.base();
		auto& b_static_strides = b.strides().base();
		
		auto c_static_extents = c.extents().base();
		auto c_static_strides = c.strides().base();

		outer(c.data(), c.rank(), c_static_extents.data(), c_static_strides.data(),
			a.data(), a.rank(), a_static_extents.data(), a_static_strides.data(),
			b.data(), b.rank(), b_static_extents.data(), b_static_strides.data());

		return c;
	}

}


#endif
