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
#include "prod/tensor_times_matrix.hpp"

//#include "fixed_rank_extents.hpp"

namespace boost::numeric::ublas{
    
    template<typename T> class tensor_core;

} // namespace boost::numeric::ublas


namespace boost::numeric::ublas
{


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
