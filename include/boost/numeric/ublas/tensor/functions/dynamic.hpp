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


#ifndef BOOST_NUMERIC_UBLAS_TENSOR_FUNCTIONS_DYNAMIC_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_FUNCTIONS_DYNAMIC_HPP

#include "../multiplication.hpp"
#include "../tensor_engine.hpp"
#include "../detail/strides_functions.hpp"

#include "../traits/type_traits_extents.hpp"
#include "../traits/type_traits_strides.hpp"

#include "../fixed_rank_strides.hpp"

namespace boost::numeric::ublas{

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
    template <typename TensorEngine, typename A, 
        typename ExtentsType = typename tensor_core< TensorEngine >::extents_type,
        std::enable_if_t< is_dynamic_rank_v< ExtentsType >, void >* = nullptr
    >
    inline decltype(auto) prod( tensor_core< TensorEngine > const &a, 
        vector<typename tensor_core< TensorEngine >::value_type, A> const &b, 
        const std::size_t m)
    {

        using tensor_type   = tensor_core< TensorEngine >;
        using extents_type  = typename tensor_type::extents_type;
        using value_type    = typename tensor_type::value_type;
        using array_type    = typename tensor_type::array_type;
        using layout_type   = typename tensor_type::layout_type;

        auto const p = a.rank();

        static_assert(  
            std::is_same_v< 
                typename tensor_core<TensorEngine>::resizable_tag, 
                storage_resizable_container_tag 
            >,
            "error in boost::numeric::ublas::prod(tensor_core const&, vector const& ): "
            "tensor container should be resizable"
        );

        static_assert(  
            is_dynamic_v<extents_type>,
            "error in boost::numeric::ublas::prod(tensor_core const&, vector const& ): "
            "extents type should be dynamic"
        );

        if (m == 0ul)
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

        if (b.empty())
            throw std::length_error(
                "error in boost::numeric::ublas::prod(ttv): second "
                "argument vector should not be empty.");

        using extents_value_type = typename extents_type::value_type;

        auto const& na = a.extents();

        auto compute_nc = [](auto const& na){
            using size_type = typename extents_type::size_type;
            using extents_base_type = typename extents_type::base_type;
            auto const sz = std::max( ublas::size(na) - 1, size_type(2) );
            auto arr = extents_base_type(sz,1);
            return extents_type{ std::move(arr) } ;
        };

        auto nc = compute_nc(na);
        auto nb = std::vector<extents_value_type>{b.size(), extents_value_type(1)};

        for (auto i = 0ul, j = 0ul; i < p; ++i)
            if (i != m - 1)
                nc[j++] = na.at(i);

        using c_extents_type = std::decay_t< decltype(nc) >;

        using t_engine = tensor_engine< 
            c_extents_type,  
            layout_type,
            strides<c_extents_type>,
            array_type
        >;
        
        auto c = tensor_core<t_engine>( nc, value_type{} );
        auto bb = &(b(0));
        ttv(m, p,
            c.data(), data(c.extents()), c.strides().data(),
            a.data(), data(a.extents()), a.strides().data(),
            bb, data(nb), data(nb));
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
    template <typename TensorEngine1, typename TensorEngine2,
        typename ExtentsType1 = typename tensor_core< TensorEngine1 >::extents_type,
        typename ExtentsType2 = typename tensor_core< TensorEngine2 >::extents_type,
        std::enable_if_t<
            (is_dynamic_rank_v<ExtentsType1> || is_dynamic_rank_v<ExtentsType2>) &&
            !(is_static_v<ExtentsType1> && is_static_v<ExtentsType2>)
            ,void>* = nullptr
    >
    inline decltype(auto) outer_prod(tensor_core< TensorEngine1 > const &a, tensor_core< TensorEngine2 > const &b)
    {
        using tensor_type   = tensor_core< TensorEngine1 >;
        using value_type    = typename tensor_type::value_type;
        using layout_type   = typename tensor_type::layout_type;
        using array_type    = typename tensor_type::array_type;

        static_assert( 
            std::is_same_v< 
                typename tensor_core<TensorEngine1>::resizable_tag, 
                typename tensor_core<TensorEngine2>::resizable_tag 
            > && 
            std::is_same_v< 
                typename tensor_core<TensorEngine1>::resizable_tag, 
                storage_resizable_container_tag 
            >,
            "error in boost::numeric::ublas::outer_prod(tensor_core const&, tensor_core const&): "
            "Both the tensor storage should have the same type of storage and both should be resizable"
        );

        static_assert(
            std::is_same_v<value_type, typename tensor_core< TensorEngine2 >::value_type>,
            "error in boost::numeric::ublas::outer_prod(tensor_core< TensorEngine1 > const&, tensor_core< TensorEngine2 > const&): "
            "Both the tensor should have the same value_type"
        );

        if (a.empty() || b.empty())
            throw std::runtime_error(
                "error in boost::numeric::ublas::outer_prod: "
                "tensors should not be empty.");

        auto const& na = a.extents();
        auto const& nb = b.extents();

        auto create_nc = [](auto const& na, auto const& nb){
            using extents_type = extents<>;
            auto nc = typename extents_type::base_type(ublas::size(na)+ublas::size(nb));
            auto nci = std::copy(ublas::begin(na),ublas::end(na), std::begin(nc));
            std::copy(ublas::begin(nb),ublas::end(nb), nci);
            return extents_type(nc);
        };

        auto nc  = create_nc(na,nb);

        using c_extents_type = std::decay_t< decltype(nc) >;

        using t_engine = tensor_engine< 
            c_extents_type,  
            layout_type,
            strides<c_extents_type>,
            array_type
        >;

        auto c = tensor_core<t_engine>( nc, value_type{} );

        outer(c.data(), c.rank(), data(c.extents()), c.strides().data(),
            a.data(), a.rank(), data(na), a.strides().data(),
            b.data(), b.rank(), data(nb), b.strides().data());

        return c;
    }

} // namespace boost::numeric::ublas

#endif   // BOOST_NUMERIC_UBLAS_TENSOR_FUNCTIONS_DYNAMIC_HPP
