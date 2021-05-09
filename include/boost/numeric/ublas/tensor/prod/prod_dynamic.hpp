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


#ifndef BOOST_NUMERIC_UBLAS_TENSOR_PROD_DYNAMIC_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_PROD_DYNAMIC_HPP

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


    /** @brief Computes the q-mode tensor-times-tensor product
     *
     * Implements C[i1,...,ir,j1,...,js] = sum( A[i1,...,ir+q] * B[j1,...,js+q]  )
     *
     * @note calls ublas::ttt
     *
     * na[phia[x]] = nb[phib[x]] for 1 <= x <= q
     *
     * @param[in]  a  left-hand side tensor with order r+q
     * @param[in]  b  right-hand side tensor with order s+q
     * @param[in]  phia one-based permutation tuple of length q for the first
     * input tensor a can be of type std::vector<std::size_t> or std::array<std::size_t,N>
     * @param[in]  phib one-based permutation tuple of length q for the second
     * input tensor b can be of type std::vector<std::size_t> or std::array<std::size_t,N>
     * @result     tensor with order r+s
    */
    template <typename TensorEngine1, typename TensorEngine2, typename PermuType = std::vector<size_t>,
        typename ExtentsType1 = typename tensor_core< TensorEngine1 >::extents_type,
        typename ExtentsType2 = typename tensor_core< TensorEngine2 >::extents_type,
        std::enable_if_t<
            (is_dynamic_rank_v<ExtentsType1> || is_dynamic_rank_v<ExtentsType2> || !is_bounded_array_v<PermuType>) &&
            !(is_static_v<ExtentsType1> && is_static_v<ExtentsType2>)
            ,void>* = nullptr
    >
    inline decltype(auto) prod(tensor_core< TensorEngine1 > const &a, tensor_core< TensorEngine2 > const &b,
                                PermuType const &phia, PermuType const &phib)
    {
        using tensor_type       = tensor_core< TensorEngine1 >;
        using extents_type      = typename tensor_type::extents_type;       
        using value_type        = typename tensor_type::value_type;
        using layout_type       = typename tensor_type::layout_type;
        using size_type         = typename extents_type::size_type;
        using array_type        = typename tensor_type::array_type;

        static_assert( 
            std::is_same_v< 
                typename tensor_core<TensorEngine1>::resizable_tag, 
                typename tensor_core<TensorEngine2>::resizable_tag 
            > && 
            std::is_same_v< 
                typename tensor_core<TensorEngine1>::resizable_tag, 
                storage_resizable_container_tag 
            >,
            "error in boost::numeric::ublas::prod(tensor_core const&, tensor_core const&, "
            "PermuType const&, PermuType const& ): "
            "Both the tensor storage should have the same type of storage and both should be resizable"
        );

        static_assert(
            std::is_same_v<value_type, typename tensor_core< TensorEngine2 >::value_type>,
            "error in boost::numeric::ublas::prod(tensor_core< TensorEngine1 > const&, tensor_core< TensorEngine2 > const&, "
            "PermuType const&, PermuType const&): "
            "Both the tensor should have the same value_type"
        );

        auto const pa = a.rank();
        auto const pb = b.rank();

        auto const q = static_cast<size_type>(phia.size());

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

        std::size_t const r = pa - q;
        std::size_t const s = pb - q;

        std::vector<std::size_t> phia1(pa);
        std::vector<std::size_t> phib1(pb);
        std::iota(phia1.begin(), phia1.end(), 1ul);
        std::iota(phib1.begin(), phib1.end(), 1ul);

        auto create_nc = [](auto const& na, auto const& nb, auto const& phia, auto const& phib){
            using extents_base_type = typename extents<>::base_type;
            size_type const size = ( ublas::size(na) + ublas::size(nb) ) - ( phia.size() + phib.size() );
            return extents<>( extents_base_type ( std::max(size, size_type(2)), size_type(1) ) );
        };

        auto nc = create_nc(na,nb,phia,phib);

        for (auto i = 0ul; i < phia.size(); ++i)
            *std::remove(phia1.begin(), phia1.end(), phia.at(i)) = phia.at(i);

        //phia1.erase( std::remove(phia1.begin(), phia1.end(), phia.at(i)),  phia1.end() )  ;

        for (auto i = 0ul; i < r; ++i)
            nc[i] = na[phia1[i] - 1];

        for (auto i = 0ul; i < phib.size(); ++i)
            *std::remove(phib1.begin(), phib1.end(), phib.at(i)) = phib.at(i);
        //phib1.erase( std::remove(phib1.begin(), phib1.end(), phia.at(i)), phib1.end() )  ;

        for (auto i = 0ul; i < s; ++i)
            nc[r + i] = nb[phib1[i] - 1];

        // std::copy( phib.begin(), phib.end(), phib1.end()  );
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
        assert(phia1.size() == pa);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
        assert(phib1.size() == pb);

        using c_extents_type = std::decay_t< decltype(nc) >;

        using t_engine = tensor_engine< 
            c_extents_type,
            layout_type,
            strides<c_extents_type>,
            array_type
        >;

        auto c = tensor_core<t_engine>( nc, value_type{} );
        
        ttt(pa, pb, q,
            phia1.data(), phib1.data(),
            c.data(), data(c.extents()), c.strides().data(),
            a.data(), data(a.extents()), a.strides().data(),
            b.data(), data(b.extents()), b.strides().data());

        return c;
    }



} // namespace boost::numeric::ublas

#endif   // BOOST_NUMERIC_UBLAS_TENSOR_PROD_DYNAMIC_HPP
