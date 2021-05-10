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


#ifndef BOOST_NUMERIC_UBLAS_TENSOR_PROD_STATIC_RANK_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_PROD_STATIC_RANK_HPP

#include "../multiplication.hpp"
#include "../tensor_engine.hpp"
#include "../detail/strides_functions.hpp"

#include "../traits/type_traits_extents.hpp"
#include "../traits/type_traits_strides.hpp"

#include "../fixed_rank_strides.hpp"


namespace boost::numeric::ublas{

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
            (is_static_rank_v<ExtentsType1> && is_static_rank_v<ExtentsType2>) &&
            (is_dynamic_v<ExtentsType1> && is_dynamic_v<ExtentsType2>) &&
            is_bounded_array_v<PermuType>
            ,void>* = nullptr
    >
    inline decltype(auto) prod(tensor_core< TensorEngine1 > const &a,
                           tensor_core< TensorEngine2 > const &b,
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
            using lextents_type      = std::decay_t< decltype(na) >;
            using rextents_type      = std::decay_t< decltype(nb) >;
            using array_type1        = std::decay_t< decltype(phia) >;
            using array_type2        = std::decay_t< decltype(phib) >;
            constexpr auto const N   = extent_of_rank_one_array_v<array_type1> + extent_of_rank_one_array_v<array_type2>;
            constexpr auto const sz  = size(lextents_type{}) + size(rextents_type{}) - N;
            constexpr auto const msz = std::max(size_type(sz), size_type(2));
            return extents<msz>();
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
            (is_static_rank_v<ExtentsType1> && is_static_rank_v<ExtentsType2>) &&
            (is_dynamic_v<ExtentsType1> && is_dynamic_v<ExtentsType2>)
            ,void>* = nullptr
    >
    inline decltype(auto) outer_prod(tensor_core< TensorEngine1 > const &a, tensor_core< TensorEngine2 > const &b)
    {
        using tensor_type   = tensor_core< TensorEngine1 >;
        using value_type    = typename tensor_type::value_type;
        using layout_type   = typename tensor_type::layout_type;
        using array_type   = typename tensor_type::array_type;

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
            using na_type = std::decay_t< decltype(na) >;
            using nb_type = std::decay_t< decltype(nb) >;
            constexpr auto na_size = size(na_type{});
            constexpr auto nb_size = size(nb_type{});

            using extents_type = extents<na_size+nb_size>;
            auto nc = typename extents_type::base_type{};
            auto nci = std::copy(ublas::begin(na), ublas::end(na), std::begin(nc));
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

#endif // BOOST_NUMERIC_UBLAS_TENSOR_PROD_STATIC_RANK_HPP
