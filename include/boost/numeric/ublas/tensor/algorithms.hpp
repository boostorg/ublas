//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB, Ettlingen Germany
//


#ifndef _BOOST_UBLAS_TENSOR_ALGORITHMS_
#define _BOOST_UBLAS_TENSOR_ALGORITHMS_


#include <stdexcept>
#include <complex>

namespace boost {
namespace numeric {
namespace ublas {
namespace detail {
namespace recursive {


/** @brief Copies a tensor to another tensor with different layouts
 *
 * Implements C[i1,i2,...,ip] = A[i1,i2,...,ip]
 *
 * @note is used in function trans or for tensor views
 *
 * @param[in]  r  zero-based recursion level starting with r=p-1
 * @param[in]  n  pointer to the extents of input or output tensor
 * @param[out] c  pointer to the output tensor
 * @param[in] wc pointer to the strides of output tensor c
 * @param[in] a  pointer to the first input tensor
 * @param[in] wa pointer to the strides of input tensor a
*/
template <class	PointerOut, class PointerIn, class SizeType>
void copy(SizeType r,    SizeType const*const n,
					PointerOut c,  SizeType const*const wc,
					PointerIn a,   SizeType const*const wa)
{
	if(r > 0)
		for(auto d = 0u; d < n[r]; c += wc[r], a += wa[r], ++d)
			copy(r-1, n, c, wc, a, wa);
	else
		for(auto d = 0u; d < n[0]; c += wc[0], a += wa[0], ++d)
			*c = *a;
}


/** @brief Transposes a tensor
 *
 * Implements C[tau[i1],tau[i2],...,tau[ip]] = A[i1,i2,...,ip]
 *
 * @note is used in function trans
 *
 * @param[in]  r zero-based recursion level starting with r=p-1
 * @param[in] na pointer to the extents of input tensor a of length p
 * @param[in] pi pointer to a one-based permutation tuple of length p
 * @param[out] c pointer to the output tensor
 * @param[in] wc pointer to the strides of output tensor c
 * @param[in]  a pointer to the input tensor
 * @param[in] wa pointer to the strides of input tensor a
*/
template <class	PointerOut, class PointerIn, class SizeType>
void trans(SizeType const r,   SizeType const*const n, SizeType const*const pi,
					 PointerOut c, SizeType const*const wc,
					 PointerIn a,  SizeType const*const wa)
{
	if(r > 0){
		for(auto d = 0u; d < n[r]; c += wc[pi[r]-1], a += wa[r], ++d){
			trans(r-1, n, pi, c, wc, a, wa);
		}
	}
	else{
		for(auto d = 0u; d < n[0]; c += wc[pi[0]-1], a += wa[0], ++d){
			*c = *a;
		}
	}
}


/** @brief Transposes a tensor
 *
 * Implements C[tau[i1],tau[i2],...,tau[ip]] = A[i1,i2,...,ip]
 *
 * @note is used in function trans
 *
 * @param[in]  r zero-based recursion level starting with r=p-1
 * @param[in] na pointer to the extents of input tensor a of length p
 * @param[in] pi pointer to a one-based permutation tuple of length p
 * @param[out] c pointer to the output tensor
 * @param[in] wc pointer to the strides of output tensor c
 * @param[in]  a pointer to the input tensor
 * @param[in] wa pointer to the strides of input tensor a
*/
template <class	ValueType, class SizeType>
void trans(SizeType const r, SizeType const*const n, SizeType const*const pi,
					 std::complex<ValueType>* c, SizeType const*const wc,
					 std::complex<ValueType>* a,  SizeType const*const wa)
{
	if(r > 0)
		for(auto d = 0u; d < n[r]; c += wc[pi[r]-1], a += wa[r], ++d)
			trans(r-1, n, pi, c, wc, a, wa);
	else
		for(auto d = 0u; d < n[0]; c += wc[pi[0]-1], a += wa[0], ++d)
			*c = std::conj(*a);
}


} // recursive
} // detail
} // ublas
} // numeric
} // boost





namespace boost {
namespace numeric {
namespace ublas {



/** @brief Copies a tensor to another tensor with different layouts
 *
 * Implements C[i1,i2,...,ip] = A[i1,i2,...,ip]
 *
 * @param[in]  p rank of input and output tensor
 * @param[in]  n pointer to the extents of input or output tensor of length p
 * @param[in] pi pointer to a one-based permutation tuple of length p
 * @param[out] c pointer to the output tensor
 * @param[in] wc pointer to the strides of output tensor c
 * @param[in]  a pointer to the input tensor
 * @param[in] wa pointer to the strides of input tensor a
*/
template <class	PointerOut, class PointerIn, class SizeType>
void copy(const SizeType p, SizeType const*const n,
					PointerOut c, SizeType const*const wc,
					PointerIn a,  SizeType const*const wa)
{
	static_assert( std::is_pointer<PointerOut>::value & std::is_pointer<PointerIn>::value,
								 "Static error in boost::numeric::ublas::copy: Argument types for pointers are not pointer types.");
	if( p == 0 )
		return;

	if(c == nullptr || a == nullptr)
		throw std::length_error("Error in boost::numeric::ublas::copy: Pointers shall not be null pointers.");

	detail::recursive::copy( p-1, n, c, wc, a, wa );

}


/** @brief Transposes a tensor
 *
 * Implements C[tau[i1],tau[i2],...,tau[ip]] = A[i1,i2,...,ip]
 *
 * @note is used in function trans
 *
 * @param[in]  p rank of input and output tensor
 * @param[in] na pointer to the extents of the input tensor a of length p
 * @param[in] pi pointer to a one-based permutation tuple of length p
 * @param[out] c pointer to the output tensor
 * @param[in] wc pointer to the strides of output tensor c
 * @param[in]  a pointer to the input tensor
 * @param[in] wa pointer to the strides of input tensor a
*/

template <class PointerOut, class PointerIn, class SizeType>
void trans( SizeType const p,  SizeType const*const na, SizeType const*const pi,
			PointerOut c,      SizeType const*const wc,
			PointerIn a,       SizeType const*const wa)
{

	static_assert( std::is_pointer<PointerOut>::value & std::is_pointer<PointerIn>::value,
				   "Static error in boost::numeric::ublas::trans: Argument types for pointers are not pointer types.");

	if( p < 2)
		return;

	if(c == nullptr || a == nullptr)
		throw std::length_error("Error in boost::numeric::ublas::trans: Pointers shall not be null pointers.");

	detail::recursive::trans( p-1, na, pi, c, wc, a, wa );
}


/** @brief Transposes a tensor
 *
 * Implements C[tau[i1],tau[i2],...,tau[ip]] = A[i1,i2,...,ip]
 *
 * @note is used in function trans
 *
 * @param[in]  p rank of input and output tensor
 * @param[in] na pointer to the extents of the input tensor a of length p
 * @param[in] pi pointer to a one-based permutation tuple of length p
 * @param[out] c pointer to the output tensor
 * @param[in] wc pointer to the strides of output tensor c
 * @param[in]  a pointer to the input tensor
 * @param[in] wa pointer to the strides of input tensor a
*/

template <class ValueType, class SizeType>
void trans( SizeType const p,  SizeType const*const na, SizeType const*const pi,
			std::complex<ValueType>* c,      SizeType const*const wc,
			std::complex<ValueType>* a,       SizeType const*const wa)
{

//	static_assert( std::is_pointer<PointerOut>::value & std::is_pointer<PointerIn>::value,
//					 "Static error in boost::numeric::ublas::trans: Argument types for pointers are not pointer types.");

	if( p < 2)
		return;

	if(c == nullptr || a == nullptr)
		throw std::length_error("Error in boost::numeric::ublas::trans: Pointers shall not be null pointers.");

	detail::recursive::trans( p-1, na, pi, c, wc, a, wa );
}

}
}
}

#endif
