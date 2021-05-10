//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2021, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//


#ifndef BOOST_NUMERIC_UBLAS_TENSOR_PROD_STATIC_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_PROD_STATIC_HPP

#include "../multiplication.hpp"
#include "../tensor_engine.hpp"
#include "../detail/strides_functions.hpp"

#include "../traits/type_traits_extents.hpp"
#include "../traits/type_traits_strides.hpp"

#include "../fixed_rank_strides.hpp"

namespace boost::numeric::ublas{

    

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
    template <typename TensorEngine1, typename TensorEngine2,
        std::enable_if_t<
            is_static_v< typename tensor_core< TensorEngine1 >::extents_type > &&
            is_static_v< typename tensor_core< TensorEngine2 >::extents_type >
            ,int> = 0
    >
    inline decltype(auto) outer_prod(tensor_core< TensorEngine1 > const &a, tensor_core< TensorEngine2 > const &b)
    {
        if (a.empty() || b.empty())
            throw std::runtime_error(
                "error in boost::numeric::ublas::outer_prod: "
                "tensors should not be empty.");

        using extents_type1 = std::decay_t< decltype(a.extents()) >;
        using extents_type2 = std::decay_t< decltype(b.extents()) >;
        using array_type    = typename tensor_core< TensorEngine1 >::array_type;
        using value_type    = typename tensor_core< TensorEngine1 >::value_type;
        using layout_type   = typename tensor_core< TensorEngine1 >::layout_type;

        static_assert(
            std::is_same_v<value_type, typename tensor_core< TensorEngine2 >::value_type>,
            "error in boost::numeric::ublas::outer_prod(tensor_core< TensorEngine1 > const&, tensor_core< TensorEngine2 > const&): "
            "Both the tensor should have the same value_type"
        );
        
        auto nc = detail::impl::concat_t<extents_type1, extents_type2>{};

        auto a_extents = a.extents();
        auto b_extents = b.extents();
        
        
        using c_extents_type = std::decay_t<decltype(nc)>;
        
        using t_engine = tensor_engine<
            c_extents_type,
            layout_type,
            strides<c_extents_type>,
            rebind_storage_size_t<c_extents_type,array_type>
        >;

        auto c = t_engine(value_type{});

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

} // namespace boost::numeric::ublas

#endif  // BOOST_NUMERIC_UBLAS_TENSOR_PROD_STATIC_HPP
