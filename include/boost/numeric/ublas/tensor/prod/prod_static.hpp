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

    namespace detail{
        template<typename T, std::size_t N>
        constexpr auto array_of_ones() noexcept{
            std::array<T,N> ones{};
            std::fill(ones.begin(), ones.end(), T{1});
            return ones;
        }

        template<std::size_t M, typename ExtentsType>
        constexpr auto extents_result_tensor_times_vector_impl(ExtentsType const& e) noexcept{
            static_assert(size(ExtentsType{}) > 0ul, "extents cannot be empty!");
            using extents_type = typename ExtentsType::value_type;
            constexpr auto sz = size(ExtentsType{}) - 1ul;
            auto res = array_of_ones<extents_type,sz>();
            
            auto j = 0ul;
            for(auto i = 0ul; i < sz; ++i){
                if(i != M - 1ul) res[j++] = e[i];
            }
            return res;
        }

        template<std::size_t M, typename ExtentsType, std::size_t... Is>
        constexpr auto extents_result_tensor_times_vector_helper([[maybe_unused]] ExtentsType const& /*e*/, [[maybe_unused]] std::index_sequence<Is...> /*is*/) noexcept{
            using extents_type = typename ExtentsType::value_type;
            constexpr auto res_arr = extents_result_tensor_times_vector_impl<M>(ExtentsType{});
            return basic_static_extents<extents_type, ( ..., res_arr[Is] ) >{};
        }

        template<std::size_t M, typename ExtentsType>
        constexpr auto extents_result_tensor_times_vector([[maybe_unused]] ExtentsType const& /*e*/) noexcept{
            static_assert(is_static_v<ExtentsType>);
            return extents_result_tensor_times_vector_helper<M>(ExtentsType{}, std::make_index_sequence<size(ExtentsType{})>{});
        }

        template<std::size_t I, std::size_t Value, typename ExtentsType>
        constexpr auto static_extents_set_at_impl(ExtentsType const& e) noexcept{
            using extents_type = typename ExtentsType::value_type;
            
            auto res = e.base();
            
            res[I] = static_cast<extents_type>(Value);

            return res;
        }

        template<std::size_t I, std::size_t Value, typename ExtentsType, std::size_t... Is>
        constexpr auto static_extents_set_at_helper( [[maybe_unused]] ExtentsType const& /*e*/, [[maybe_unused]] std::index_sequence<Is...> /*is*/){
            using extents_type = typename ExtentsType::value_type;
            constexpr auto res_arr = static_extents_set_at_impl<I,Value>(ExtentsType{});
            return basic_static_extents<extents_type, ( ..., res_arr[Is] ) >{};
        }

        template<std::size_t I, std::size_t Value, typename ExtentsType>
        constexpr auto static_extents_set_at( [[maybe_unused]] ExtentsType const& /*e*/){
            static_assert(is_static_v<ExtentsType>);
            static_assert( I < size(ExtentsType{}), "boost::numeric::ublas::detail::static_extents_set_at(ExtentsType const&): out of bound");
            return static_extents_set_at_helper<I,Value>(ExtentsType{}, std::make_index_sequence<size(ExtentsType{})>{});
        }

    } // namespace detail
    
    /** @brief Computes the m-mode tensor-times-vector product
     *
     * Implements C[i1,...,im-1,im+1,...,ip] = A[i1,i2,...,ip] * b[im]
     *
     * @note calls ublas::ttv
     *
     * @tparam    M contraction dimension with 1 <= m <= p
     * @param[in] a tensor object A with order p
     * @param[in] b vector object B
     *
     * @returns tensor object C with order p-1, the same storage format and allocator type as A
    */
    template <size_t M, typename TensorType, typename A>
    inline decltype(auto) prod(tensor_core< TensorType > const &a
        , vector<typename tensor_core< TensorType >::value_type, A> const &b)
    {
        using tensor_type   = tensor_core< TensorType >;
        using array_type    = typename tensor_type::array_type;
        using extents_type  = typename tensor_type::extents_type;
        using value_type    = typename tensor_type::value_type;
        using layout_type   = typename tensor_type::layout_type;

        auto const p = std::size_t(a.rank());

        static_assert( M != 0ul, 
                "error in boost::numeric::ublas::prod(ttv): "
                "contraction mode must be greater than zero.");

        static_assert( extents_type::_size >= M,
                "error in boost::numeric::ublas::prod(ttv): rank of tensor must be "
                "greater than or equal to the modus.");

        static_assert(extents_type::_size != 0,
                "error in boost::numeric::ublas::prod(ttv): first "
                "argument tensor should not be empty.");

        if (b.size() == 0ul)
            throw std::length_error(
                "error in boost::numeric::ublas::prod(ttv): second "
                "argument vector should not be empty.");

        using extents_value_type = typename extents_type::value_type;

        auto nc = detail::extents_result_tensor_times_vector<M>(a.extents());
        auto nb = std::vector<extents_value_type>{b.size(), extents_value_type(1)};
        using c_extents_type = std::decay_t<decltype(nc)>;
        
        using t_engine = tensor_engine<
            c_extents_type,
            layout_type,
            strides<c_extents_type>,
            rebind_storage_size_t<c_extents_type,array_type>
        >;
        
        auto c = t_engine(value_type{});
        auto bb = &(b(0));

        auto& a_static_extents = a.extents().base();
        auto& c_static_extents = c.extents().base();

        auto& a_static_strides = a.strides().base();
        auto& c_static_strides = c.strides().base();

        ttv(M, p,
            c.data(), c_static_extents.data(), c_static_strides.data(),
            a.data(), a_static_extents.data(), a_static_strides.data(),
            bb, data(nb), data(nb));

        return c;
    }

    /** @brief Computes the m-mode tensor-times-matrix product
     *
     * Implements C[i1,...,im-1,j,im+1,...,ip] = A[i1,i2,...,ip] * B[j,im]
     *
     * @note calls ublas::ttm
     *
     * @tparam    M contraction dimension with 1 <= M <= p
     * @tparam    MatrixDimension is a non contracting dimension
     * @param[in] a tensor object A with order p
     * @param[in] b vector object B
     *
     * @returns tensor object C with order p, the same storage format and allocator type as A
    */
    template <size_t M, size_t MatrixDimension, typename TensorType, typename A>
    inline decltype(auto) prod(tensor_core< TensorType > const &a, 
        matrix<typename tensor_core< TensorType >::value_type, typename tensor_core< TensorType >::layout_type, A> const &b)
    {
        using tensor_type   = tensor_core< TensorType >;
        using extents_type  = typename tensor_type::extents_type;
        using layout_type   = typename tensor_type::layout_type;
        using value_type    = typename tensor_type::value_type;
        using array_type    = typename tensor_type::array_type;
        using dynamic_strides_type = strides_t<extents<>, layout_type>;
        
        auto const p = a.rank();

        static_assert(M != 0ul,
                "error in boost::numeric::ublas::prod(ttm): "
                "contraction mode must be greater than zero.");

        static_assert( extents_type::_size >= M ,
                "error in boost::numeric::ublas::prod(ttm): rank "
                "of the tensor must be greater equal the modus.");

        static_assert( extents_type::_size,
                "error in boost::numeric::ublas::prod(ttm): first "
                "argument tensor should not be empty.");

        if (b.size1() * b.size2() == 0ul)
            throw std::length_error(
                "error in boost::numeric::ublas::prod(ttm): second "
                "argument matrix should not be empty.");


        auto const& na = a.extents();
        auto nc_base = detail::static_extents_set_at< M - 1, MatrixDimension >( na );
        auto nb = extents<>{b.size1(), b.size2()};

        auto wb = dynamic_strides_type(nb);
        
        using c_extents_type = std::decay_t<decltype(nc_base)>;
        
        using t_engine = tensor_engine<
            c_extents_type,
            layout_type,
            strides<c_extents_type>,
            rebind_storage_size_t<c_extents_type,array_type>
        >;
        auto c = t_engine(value_type{});

        auto bbdata = &(b(0, 0));

        auto const& nc = c.extents();
        auto const& wa = a.strides();
        auto const& wc = c.strides();

        ttm(M, p,
            c.data(), data(nc), wc.data(),
            a.data(), data(na), wa.data(),
            bbdata  , data(nb), wb.data());

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
