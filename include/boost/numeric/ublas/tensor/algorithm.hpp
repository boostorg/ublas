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



#ifndef _BOOST_UBLAS_TENSOR_ALGORITHM_
#define _BOOST_UBLAS_TENSOR_ALGORITHM_

#include <type_traits>

namespace boost { namespace numeric { namespace ublas {
struct first_order;
}}}


namespace boost { namespace numeric { namespace ublas { namespace detail {

template<class layout_type>
struct recursion
{
	static constexpr bool is_first_order = std::is_same_v<layout_type,first_order>;

	static constexpr inline auto cmpr(const std::size_t r, const std::size_t p)
	{
		if constexpr (is_first_order) return r > 0;
		else                          return r < p;
	}


	static constexpr inline auto nextr(const std::size_t r)
	{
		if constexpr (is_first_order) return r - 1;
		else                          return r + 1;
	}

	static constexpr inline auto lastr(const std::size_t p)
	{
		if constexpr (is_first_order) return 0;
		else                          return p-1;
	}


	template<class size_type, class value_type, class unary_op>
	static void run (const std::size_t p,
					const std::size_t r,
					const size_type *const n,
					const size_type *const wa,
					const size_type *const wc,
					const value_type* a,
					value_type *c,
					unary_op op)
	{
		if(cmpr(r,p))
			for(auto i = 0ul; i < n[r]; ++i, a+=wa[r], c+=wc[r] )
				run(p, nextr(r), n, wa, wc, a, c, op);
		else
			for(auto i = 0ul; i < n[0]; ++i, ++a, ++c )
				op(*c,*a);
	}
};

}}}}



namespace boost { namespace numeric { namespace ublas {

template<class size_type, class value_type, class layout_type>
void copy(size_type         const p,
		  size_type  const*const  n,
		  size_type  const*const wa,
		  size_type  const*const wc,
		  value_type const*  a,
		  value_type      *  c,
		  layout_type l = layout_type() )
{
	detail::recursion<layout_type>::run( p, p-1, n, wa, wc, a, c, [](value_type& c, value_type const& a){c = a;});
}

}}}






template <class pointer_t_c, class pointer_t_a, class pointer_t_b>
void
ttm_recursion_notzero(
        const size_t m,  size_t r,
        pointer_t_c c, const size_t* nc, const size_t* wc,
        pointer_t_a a, const size_t* na, const size_t* wa,
        pointer_t_b b, const size_t* nb, const size_t* wb)
{

    if(r == m) {
        ttm_recursion_notzero(m, r-1, c, nc, wc,    a, na, wa,    b, nb, wb);
    }


    else if(r == 0){
        for(size_t i0 = 0u; i0 < nc[0]; c += wc[0], a += wa[0], ++i0) {

            auto cm = c;
            auto b0 = b;

            // r == m
            for(size_t i0 = 0u; i0 < nc[m]; cm += wc[m], b0 += wb[0], ++i0){

                auto am = a;
                auto b1 = b0;

                for(size_t i1 = 0u; i1 < nb[1]; am += wa[m], b1 += wb[1], ++i1){
                    *cm += *am * *b1;
                }
            }
        }
    }

    else{
        for(size_t i = 0u; i < na[r]; c += wc[r], a += wa[r], ++i)
            ttm_recursion_notzero(m, r-1, c, nc, wc,    a, na, wa,    b, nb, wb);
    }
}








template <class pointer_t_c, class pointer_t_a, class pointer_t_b>
void
ttm_recursion_zero(
        size_t r,
        pointer_t_c c, const size_t* nc, const size_t* wc,
        pointer_t_a a, const size_t* na, const size_t* wa,
        pointer_t_b b, const size_t* nb, const size_t* wb)
{

    if(r > 1){
        for(size_t i = 0u; i < na[r]; c += wc[r], a += wa[r], ++i)
            ttm_recursion_zero(r-1, c, nc, wc,    a, na, wa,    b, nb, wb);
    }
    else{
        for(size_t i1 = 0u; i1 < nc[1]; c += wc[1], a += wa[1], ++i1) {
            auto cm = c;
            auto b0 = b;
            // r == m == 0
            for(size_t i0 = 0u; i0 < nc[0]; cm += wc[0], b0 += wb[0], ++i0){

                auto am = a;
                auto b1 = b0;
                for(size_t i1 = 0u; i1 < nb[1]; am += wa[0], b1 += wb[1], ++i1){

                    *cm += *am * *b1;
                }
            }
        }
    }
}


template <class pointer_t_a, class pointer_t_b>
typename std::remove_cv<typename std::remove_pointer<pointer_t_a>::type>::type
inner_recursive( const size_t r, const size_t* n, pointer_t_a a, const size_t* wa, pointer_t_b b, const size_t* wb)
{
	using value_t   = typename std::remove_cv<typename std::remove_pointer<pointer_t_a>::type>::type;
	using value_t_b = typename std::remove_cv<typename std::remove_pointer<pointer_t_b>::type>::type;
	static_assert(std::is_same<value_t, value_t_b>::value, "static assert in inner_recursive: both pointer types must point to the same type.");

	value_t sum = value_t();

    if(r == 0){
        for(size_t i0 = 0u; i0 < n[0]; a += wa[0], b += wb[0], ++i0)
            sum += *a * *b;
    }
    else{
        for(size_t ir = 0u; ir < n[r]; a += wa[r], b += wb[r], ++ir)
             sum += inner_recursive(r-1, n,   a, wa,    b, wb);
    }
    return sum;
}


template <class pointer_t_c, class pointer_t_a, class pointer_t_b>
void
outer_2x2(
        size_t const pa,
        size_t const rc, pointer_t_c c, const size_t*   , const size_t* wc,
        size_t const ra, pointer_t_a a, const size_t* na, const size_t* wa,
        size_t const rb, pointer_t_b b, const size_t* nb, const size_t* wb)
{
    assert(rc == 3);
    assert(ra == 1);
    assert(rb == 1);

	for(size_t ib1 = 0u; ib1 < nb[1]; b += wb[1], c += wc[pa+1], ++ib1)
    {
        auto c2 = c;
        auto b0 = b;
		for(size_t ib0 = 0u; ib0 < nb[0]; b0 += wb[0], c2 += wc[pa], ++ib0)
        {
            const auto b = *b0;
            auto c1 = c2;
            auto a1 = a;                        
            for(size_t ia1 = 0u; ia1 < na[1]; a1 += wa[1], c1 += wc[1], ++ia1)
            {
                auto a0 = a1;
                auto c0 = c1;
                for(size_t ia0 = 0u; ia0 < na[0]; a0 += wa[0], c0 += wc[0], ++ia0){
                    *c0 = *a0 * b;
                }
            }
        }
    }
}

template<class pointer_t_c, class pointer_t_a, class pointer_t_b>
void
outer_recursion(
        size_t const pa,
        size_t const rc, pointer_t_c c, const size_t* nc, const size_t* wc,
        size_t const ra, pointer_t_a a, const size_t* na, const size_t* wa,
        size_t const rb, pointer_t_b b, const size_t* nb, const size_t* wb)
{
    if(rb > 1) // ra > 1 &&
    {
        for(size_t ib = 0u; ib < nb[rb]; b += wb[rb], c += wc[rc], ++ib)
			 outer_recursion(pa, rc-1, c, nc, wc,    ra, a, na, wa,    rb-1, b, nb, wb);
    }
    else if(ra > 1) //  && rb == 1
    {
        for(size_t ia = 0u; ia < na[ra]; a += wa[ra], c += wc[ra], ++ia)
			 outer_recursion(pa, rc-1, c, nc, wc,   ra-1, a, na, wa,   rb, b, nb, wb);
    }
    else
    {
        assert(ra == 1 && rb == 1 && rc == 3);
		outer_2x2(pa, rc, c, nc, wc,   ra, a, na, wa,    rb, b, nb, wb);
    }
}


template <class pointer_t_c, class pointer_t_a, class pointer_t_b>
void
ttt_recursion(
        size_t k, size_t const ar, size_t  br, size_t const m,
        pointer_t_c c, const size_t* nc, const size_t* wc,
        pointer_t_a a, const size_t* na, const size_t* wa,
        pointer_t_b b, const size_t* nb, const size_t* wb)
{
    if(k < ar)
    {
        assert(nc[k] == na[k]);
        for(size_t ic = 0u; ic < nc[k]; a += wa[k], c += wc[k], ++ic)
            ttt_recursion(k+1, ar, br, m,    c, nc, wc,   a, na, wa,   b, nb, wb);
    }
    else if(k < ar+br)
    {
        assert(nc[k] == nb[k-ar]);
        for(size_t ic = 0u; ic < nc[k]; b += wb[k-ar], c += wc[k], ++ic)
            ttt_recursion(k+1, ar, br, m,    c, nc, wc,   a, na, wa,   b, nb, wb);
    }
    else if(k < ar+br+m-1)
    {
        assert(na[k-br] == nb[k-ar]);
        for(size_t ia = 0u; ia < na[k-br]; a += wa[k-br], b += wb[k-ar], ++ia)
            ttt_recursion(k+1, ar, br, m,    c, nc, wc,   a, na, wa,   b, nb, wb);
    }
    else
    {
        assert(na[k-br] == nb[k-ar]);
        for(size_t ia = 0u; ia < na[k-br]; a += wa[k-br], b += wb[k-ar], ++ia)
            *c += *a * *b;
    }
}


} // namespace detail
} // namespace ublas
} // namespace numeric
} // namespace boost


namespace boost {
namespace numeric {
namespace ublas {

/*
template <class pointer_t_c, class pointer_t_a>
void
transpose(size_t p, const size_t* tau,
		  pointer_t_c c, const size_t* nc, const size_t* wc,
		  pointer_t_a a, const size_t* na, const size_t* wa)
{
	if(c == nullptr || a == nullptr)
		throw std::length_error("Error in fhg::transpose(...) : data pointers shall not be null pointers.");

	if(nc == nullptr || na == nullptr)
		throw std::length_error("Error in fhg::transpose(...) : extent pointers shall not be null pointers.");

	if(wc == nullptr || wa == nullptr)
		throw std::length_error("Error in fhg::transpose(...) : stride pointers shall not be null pointers.");

	if(tau == nullptr)
		throw std::length_error("Error in fhg::transpose(...) : transpose permutation pointer shall not be a null pointer.");

	if(p < 2)
		throw std::length_error("Error in fhg::transpose(...) : rank cannot be less than 2.");


	// compute extents of C
	for(size_t i = 0u; i < p; ++i)
		if( nc[i] != na[tau[i]-1] )
			throw std::length_error("Error in fhg::transpose(...) : extents of left and right hand side tensor do not match.");

	size_t* vc = new size_t[p];
	for(size_t i = 0u; i < p; ++i)
		vc[tau[i]-1] = wc[i];

	if( std::equal(vc, vc+p, wa)  ){
		const size_t nna = std::accumulate(na, na+p, 1ul, std::multiplies<size_t>());
		std::copy(a,a+nna,c);
	}
	else{
		::fhg::copy_pointer(p, na, c, vc, a, wa);
	}


	delete[] vc;
}
*/



/*
template <class pointer_t_c, class pointer_t_a>
void
transpose(size_t p, const size_t* tau,
          pointer_t_c c, const size_t* nc, const size_t* wc,
          pointer_t_a a, const size_t* na, const size_t* wa)
{
	if(c == nullptr || a == nullptr)
		throw std::length_error("Error in fhg::transpose(...) : data pointers shall not be null pointers.");

	if(nc == nullptr || na == nullptr)
		throw std::length_error("Error in fhg::transpose(...) : extent pointers shall not be null pointers.");

	if(wc == nullptr || wa == nullptr)
		throw std::length_error("Error in fhg::transpose(...) : stride pointers shall not be null pointers.");

	if(tau == nullptr)
		throw std::length_error("Error in fhg::transpose(...) : transpose permutation pointer shall not be a null pointer.");

	if(p < 2)
		throw std::length_error("Error in fhg::transpose(...) : rank cannot be less than 2.");


	// compute extents of C
	for(size_t i = 0u; i < p; ++i)
		if( nc[i] != na[tau[i]-1] )
			throw std::length_error("Error in fhg::transpose(...) : extents of left and right hand side tensor do not match.");

	size_t* vc = new size_t[p];
	for(size_t i = 0u; i < p; ++i)
		vc[tau[i]-1] = wc[i];

	if( std::equal(vc, vc+p, wa)  ){
		const size_t nna = std::accumulate(na, na+p, 1ul, std::multiplies<size_t>());
		std::copy(a,a+nna,c);
	}
	else{
		::fhg::copy_pointer(p, na, c, vc, a, wa);
	}


	delete[] vc;
}
*/


template <class pointer_t_c, class pointer_t_a, class pointer_t_b>
void
tensor_times_vector(size_t m, size_t p,
                    pointer_t_c c, const size_t* nc, const size_t* wc,
                    pointer_t_a a, const size_t* na, const size_t* wa,
                    pointer_t_b b, const size_t* nb, const size_t* /*wb*/)
{

	if( p < m )
        throw std::length_error("pointer_t_cError in tensor times vector function. Rank must be greater equal the modus.");

    if( p == 0)
        throw std::length_error("Error intensor times vector function. Rank must be greater than zero.");

    if(c == nullptr || a == nullptr || b == nullptr)
        throw std::length_error("Error in tensor times vector function. Pointers shall not be null pointers.");

	for(size_t i = 0; i < m-1; ++i){
        if(na[i] != nc[i])
            throw std::length_error("Error in tensor times vector function. Extents (except of dimension mode) of A and C must be equal.");
    }

    for(size_t i = m; i < p; ++i){
        if(na[i] != nc[i-1])
            throw std::length_error("Error in tensor times vector function. Extents (except of dimension mode) of A and C must be equal.");
    }

	const auto max = std::max(nb[0], nb[1]);
	if(  na[m-1] != max)
        throw std::length_error("Error in tensor times vector function. Extent of dimension mode of A and b must be equal.");


    if((m != 1) && (p > 2))
        detail::ttv_recursion_notzero(m-1, p-1, p-2, c, nc, wc,    a, na, wa,   b);
    else if ((m == 1) && (p > 2))
        detail::ttv_recursion_zero(p-1, c, nc, wc,    a, na, wa,   b);
    else
        detail::mtv(m-1, c, nc, wc,    a, na, wa,   b);

}

template <class pointer_t_c, class pointer_t_a, class pointer_t_b>
void
tensor_times_matrix(const size_t m, const size_t p,
                    pointer_t_c c, const size_t* nc, const size_t* wc,
                    pointer_t_a a, const size_t* na, const size_t* wa, const size_t* taua,
                    pointer_t_b b, const size_t* nb, const size_t* wb, const size_t* taub)
{
    if( p < m )
        throw std::length_error("Error in tensor times matrix function. Rank must be greater equal the modus.");

    if( p == 0)
        throw std::length_error("Error in tensor times matrix function. Rank must be greater than zero.");

    if(c == nullptr || a == nullptr || b == nullptr)
        throw std::length_error("Error in tensor times matrix function. Pointers shall not be null pointers.");

    for(size_t i = 0; i < m-1; ++i){
        if(na[taua[i]-1] != nc[i])
            throw std::length_error("Error in tensor times matrix function. Extents (except of dimension mode) of A and C must be equal.");
    }

    for(size_t i = m; i < p; ++i){
        if(na[taua[i]-1] != nc[i])
            throw std::length_error("Error in tensor times matrix function. Extents (except of dimension mode) of A and C must be equal.");
    }

    if(na[taua[m-1]-1] != nb[taub[1]-1])
        throw std::length_error("Error in tensor times matrix function. 2nd Extent of B and M-th Extent of A must be the equal.");

    if(nc[m-1] != nb[taub[0]-1])
        throw std::length_error("Error in tensor times matrix function. 1nd Extent of B and M-th Extent of C must be the equal.");


    // need to compute the inverse of the transposition tuple tau
	size_t* taua_1 = new size_t[p];
	size_t* taub_1 = new size_t[2];
    for(auto i = 0u; i < p; ++i){
        taua_1[taua[i]-1] = i+1;
    }
    for(auto i = 0u; i < 2; ++i){
        taub_1[taub[i]-1] = i+1;
	}

    //compute v_a from w_a directly
	size_t* va = new size_t[p];
	size_t* vb = new size_t[2];
//	size_t va[p], vb[2];
    for(size_t i = 0u; i < p; ++i){
        va[taua_1[i]-1] = wa[i];
    }
    for(size_t i = 0u; i < 2; ++i){
        vb[taub_1[i]-1] = wb[i];
    }

	delete[] taua_1;
	delete[] taub_1;

    if(m != 1) {
        detail::ttm_recursion_notzero(m-1, p-1, c, nc, wc,    a, na, va,   b, nb, vb);
    }
    else {
        detail::ttm_recursion_zero(p-1, c, nc, wc,    a, na, va,   b, nb, vb);
    }

	delete[] va;
	delete[] vb;
}




template <class pointer_t_c, class pointer_t_a, class pointer_t_b>
void
tensor_times_matrix(const size_t m, const size_t p,
                    pointer_t_c c, const size_t* nc, const size_t* wc,
                    pointer_t_a a, const size_t* na, const size_t* wa,
                    pointer_t_b b, const size_t* nb, const size_t* wb)
{

    static_assert( (std::is_pointer<pointer_t_c>::value && std::is_pointer<pointer_t_a>::value) && std::is_pointer<pointer_t_b>::value,
                       "Static error at tensor_times_matrix function: argument types for pointers must be pointer types.");

    if( p < m )
        throw std::length_error("Error in tensor times matrix function. Rank must be greater equal the modus.");

    if( p == 0)
        throw std::length_error("Error in tensor times matrix function. Rank must be greater than zero.");

    if(c == nullptr || a == nullptr || b == nullptr)
        throw std::length_error("Error in tensor times matrix function. Pointers shall not be null pointers.");

    for(size_t i = 0; i < m-1; ++i){
        if(na[i] != nc[i])
            throw std::length_error("Error in tensor times matrix function. Extents (except of dimension mode) of A and C must be equal.");
    }

    for(size_t i = m; i < p; ++i){
        if(na[i] != nc[i])
            throw std::length_error("Error in tensor times matrix function. Extents (except of dimension mode) of A and C must be equal.");
    }

    if(na[m-1] != nb[1])
        throw std::length_error("Error in tensor times matrix function. 2nd Extent of B and M-th Extent of A must be the equal.");

    if(nc[m-1] != nb[0])
        throw std::length_error("Error in tensor times matrix function. 1nd Extent of B and M-th Extent of C must be the equal.");

    if(m != 1)
        detail::ttm_recursion_notzero(m-1, p-1, c, nc, wc,    a, na, wa,   b, nb, wb);
    else
        detail::ttm_recursion_zero(p-1, c, nc, wc,    a, na, wa,   b, nb, wb);

}





template <class pointer_t_a, class pointer_t_b>
typename std::remove_cv<typename std::remove_pointer<pointer_t_a>::type>::type
inner(const size_t p,
      pointer_t_a a, const size_t* na, const size_t* wa,
      pointer_t_b b, const size_t* /*nb*/, const size_t* wb)
{
	static_assert( std::is_pointer<pointer_t_a>::value && std::is_pointer<pointer_t_b>::value,
	               "Static error at inner function: argument types for pointers must be pointer types.");

    if(p<2)
        throw std::length_error("Error when performing inner product: Rank must be greater than zero.");

    if(a == nullptr || b == nullptr)
        throw std::length_error("Error when performing inner product: Pointers shall not be null pointers.");

    return detail::inner_recursive(p-1, na, a, wa,    b, wb);

}




template <class pointer_t_c, class pointer_t_a, class pointer_t_b>
void
outer(size_t pc,  size_t pa,  size_t pb,
      pointer_t_c c, const size_t* nc, const size_t* wc,
      pointer_t_a a, const size_t* na, const size_t* wa,
      pointer_t_b b, const size_t* nb, const size_t* wb)
{

    if(pa < 2 || pb < 2)
        throw std::length_error("Error when performing outer product: number of extents of A and B must be equal or greater than two.");

    if((pa + pb) != pc)
        throw std::length_error("Error when performing outer product: number of extents of A plus B must be equal to the number of extents of C.");

    if(a == nullptr || b == nullptr || c == nullptr)
	{
		throw std::length_error("Error when performing outer product: Pointers shall not be null pointers.");
	}


	detail::outer_recursion(pa, pc-1, c, nc, wc,   pa-1, a, na, wa,   pb-1, b, nb, wb);

}


template <class pointer_t_c, class pointer_t_a, class pointer_t_b>
void
tensor_times_tensor(
        const size_t pc, const size_t pa, const size_t pb,
        const size_t m,	const size_t ar, const size_t br,
        const size_t* phia, const size_t* phib,
        pointer_t_c c, const size_t* nc, const size_t* wc,
        pointer_t_a a, const size_t* na, const size_t* wa,
        pointer_t_b b, const size_t* nb, const size_t* wb)
{

//    const auto pc = nc.size();
//    const auto pa = na.size();
//    const auto pb = nb.size();

    if(pa < 2 || pb < 2 || pc < 2)
        throw std::length_error("Error when performing tensor tensor product: number of extents of A, B and C must be equal or greater than two.");

//    if(ar+m != phia.size())
//        throw std::length_error("Error when performing tensor tensor product: length of permutation tuple of A must be equal to r + m.");

//    if(ar+m != na.size())
//        throw std::length_error("Error when performing tensor tensor product: length of extent tuple of A must be equal to r + m.");

//    if(ar+m != wa.size())
//        throw std::length_error("Error when performing tensor tensor product: length of stride tuple of A must be equal to r + m.");

//    if(br+m != phib.size())
//        throw std::length_error("Error when performing tensor tensor product: length of permutation tuple of B must be equal to s + m.");

//    if(br+m != nb.size())
//        throw std::length_error("Error when performing tensor tensor product: length of extent tuple of B must be equal to s + m.");

//    if(br+m != wb.size())
//        throw std::length_error("Error when performing tensor tensor product: length of stride tuple of B must be equal to s + m.");

//    if(ar+br != nc.size() && ar!=0 && br!=0)
//        throw std::length_error("Error when performing tensor tensor product: length of extent tuple of C must be equal to r + s.");

//    if(ar+br != wc.size() && ar!=0 && br!=0)
//        throw std::length_error("Error when performing tensor tensor product: length of stride tuple of C must be equal to r + s.");

    if(a == nullptr || b == nullptr || c == nullptr)
	{throw std::length_error("Error when performing tensor tensor product: Pointers shall not be null pointers.");}

	size_t* na_ = new size_t[pa];
	size_t* nb_ = new size_t[pb];
	size_t* wa_ = new size_t[pa];
	size_t* wb_ = new size_t[pb];
//	size_t na_[pa], nb_[pb];
//	size_t wa_[pa], wb_[pb];

    for(size_t k = 0; k < ar; ++k){
        na_[k] = na[phia[k]-1];
        wa_[k] = wa[phia[k]-1];
    }

    for(size_t k = 0; k < br; ++k){
        nb_[k] = nb[phib[k]-1];
        wb_[k] = wb[phib[k]-1];
    }

    for(size_t k = 0; k < m; ++k){
        na_[k+ar] = na[phia[k+ar]-1];
        wa_[k+ar] = wa[phia[k+ar]-1];
        nb_[k+br] = nb[phib[k+br]-1];
        wb_[k+br] = wb[phib[k+br]-1];
    }

    detail::ttt_recursion(0, ar, br, m,    c, nc, wc,   a, na_, wa_,   b, nb_, wb_);

	delete[] na_;
	delete[] nb_;
	delete[] wa_;
	delete[] wb_;

}





#endif
