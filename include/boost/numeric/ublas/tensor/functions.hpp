//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2021, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//


#ifndef BOOST_UBLAS_TENSOR_FUNCTIONS_HPP
#define BOOST_UBLAS_TENSOR_FUNCTIONS_HPP

#include "multiplication.hpp"
#include "tensor_engine.hpp"
#include "detail/strides_functions.hpp"

#include "traits/type_traits_extents.hpp"
#include "traits/type_traits_strides.hpp"

#include "fixed_rank_strides.hpp"
#include "prod/reshape.hpp"
#include "prod/prod_dynamic.hpp"
#include "prod/prod_static.hpp"
#include "prod/prod_static_rank.hpp"
#include "prod/inner_prod.hpp"
#include "prod/outer_prod.hpp"
#include "prod/norm.hpp"
#include "prod/imag.hpp"
#include "prod/real.hpp"
#include "prod/conj.hpp"
#include "prod/trans.hpp"
#include "prod/tensor_times_vector.hpp"

//#include "fixed_rank_extents.hpp"

namespace boost::numeric::ublas{
    
    template<typename T> class tensor_core;

} // namespace boost::numeric::ublas


namespace boost::numeric::ublas
{

    


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
    template <typename TensorEngine, typename A>
    inline decltype(auto) prod(
      tensor_core< TensorEngine > const &a,
      matrix<typename tensor_core< TensorEngine >::value_type,
             typename tensor_core< TensorEngine >::layout_type , A> const &b,
        const std::size_t m)
    {

        using tensor_type   = tensor_core< TensorEngine >;
        using extents_type  = typename tensor_type::extents_type;        
        using value_type    = typename tensor_type::value_type;
        using layout_type   = typename tensor_type::layout_type;


        static_assert(  
            std::is_same_v< 
                typename tensor_core<TensorEngine>::resizable_tag, 
                storage_resizable_container_tag 
            >,
            "error in boost::numeric::ublas::prod(tensor_core const&, matrix const& ): "
            "tensor container should be resizable"
        );

        static_assert(  
            is_dynamic_v<extents_type>,
            "error in boost::numeric::ublas::prod(tensor_core const&, matrix const& ): "
            "extents type should be dynamic"
        );

        auto const p = a.rank();

        if (m == 0ul)
            throw std::length_error(
                "error in boost::numeric::ublas::prod(ttm): "
                "contraction mode must be greater than zero.");

        if (p < m || m > ublas::size(a.extents()))
            throw std::length_error(
                "error in boost::numeric::ublas::prod(ttm): rank "
                "of the tensor must be greater equal the modus.");

        if (a.empty())
            throw std::length_error(
                "error in boost::numeric::ublas::prod(ttm): first "
                "argument tensor should not be empty.");

        if (b.size1() * b.size2() == 0ul)
            throw std::length_error(
                "error in boost::numeric::ublas::prod(ttm): second "
                "argument matrix should not be empty.");

        auto nc = a.extents();
        auto nb = extents<2>{b.size1(), b.size2()};
        auto wb = basic_fixed_rank_strides<std::size_t,2,layout_type>(nb);//strides_type(nb);
//        compute_strides(nb,wb);

        nc[m-1] = nb[0];

        auto c = tensor_type(nc, value_type{});

        auto bb = &(b(0, 0));
        ttm(m, p,
            c.data(), data(c.extents()), c.strides().data(),
            a.data(), data(a.extents()), a.strides().data(),
            bb, data(nb), wb.data());

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
     * @param[in]  a  left-hand side tensor with order r+q
     * @param[in]  b  right-hand side tensor with order s+q
     * @param[in]  phi one-based permutation tuple of length q for both input
     * tensors can be of type std::vector<std::size_t> or std::array<std::size_t,N>
     * @result     tensor with order r+s
    */
    template <typename TensorEngine1, typename TensorEngine2, typename PermuType = std::vector<size_t>>
    inline decltype(auto) prod(
      tensor_core< TensorEngine1 > const &a,
      tensor_core< TensorEngine2 > const &b,
      PermuType const &phi)
    {
        return prod(a, b, phi, phi);
    }













} // namespace boost::numeric::ublas

#endif
