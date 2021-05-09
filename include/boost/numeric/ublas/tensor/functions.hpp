//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
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

//#include "fixed_rank_extents.hpp"

namespace boost::numeric::ublas{
    
    template<typename T> class tensor_core;

} // namespace boost::numeric::ublas


namespace boost::numeric::ublas
{
    namespace detail{

        template<typename T>
        struct is_complex : std::false_type{};

        template<typename T>
        struct is_complex< std::complex<T> > : std::true_type{};

        template<typename T>
        inline static constexpr bool is_complex_v = is_complex<T>::value;

    } // namespace detail
    


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
    template <typename TensorEngine,typename PermuType = std::vector<std::size_t> >
    inline decltype(auto) trans(tensor_core< TensorEngine > const &a, PermuType const &tau)
    {

        using tensor_type   = tensor_core< TensorEngine >;
        using layout_type   = typename tensor_type::layout_type;
        using array_type    = typename tensor_type::array_type;
        using extents_type  = typename tensor_type::extents_type;
        
        static_assert(
            is_dynamic_v< extents_type > ,
            "error in boost::numeric::ublas::trans(tensor_core< TensorEngine > const &a, "
            "PermuType const &tau): "
            "Tensor should have dynamic extents"
        );

        using t_engine = tensor_engine< 
            extents_type,
            layout_type,
            strides<extents_type>,
            array_type
        >;

        auto const p = a.rank();
        auto const &na = a.extents();
        typename extents_type::base_type nc;

        if constexpr( is_dynamic_rank_v<extents_type> ){
            nc.resize(p);
        }

        for (auto i = 0u; i < p; ++i)
            nc.at(tau.at(i) - 1) = na.at(i);

        auto c = tensor_core<t_engine>( extents_type( std::move(nc) ) );
        
        if (a.empty())
            return c;

        trans(a.rank(), data(a.extents()), tau.data(),
            c.data(), c.strides().data(),
            a.data(), a.strides().data());

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
    template <typename TensorEngine>
    inline decltype(auto) norm(tensor_core< TensorEngine > const &a)
    {
        using tensor_type = tensor_core< TensorEngine >;
        using value_type = typename tensor_type::value_type;
        
        static_assert(std::is_default_constructible<value_type>::value,
                    "Value type of tensor must be default construct able in order "
                    "to call boost::numeric::ublas::norm");

        if (a.empty())
        {
            throw std::runtime_error(
                "error in boost::numeric::ublas::norm: tensors should not be empty.");
        }
        return std::sqrt(accumulate(a.order(), data(a.extents()), a.data(), a.strides().data(), value_type{},
                                    [](auto const &l, auto const &r) { return l + r * r; }));
    }


    /** @brief Computes the complex conjugate component of tensor elements within a tensor expression
     *
     * @param[in] lhs tensor expression
     * @returns   unary tensor expression
    */
    template<typename TensorEngine, class D,
        std::enable_if_t< detail::is_complex_v<typename tensor_core< TensorEngine >::value_type>, int > = 0
    >
    auto conj(detail::tensor_expression< tensor_core<TensorEngine>, D > const& expr)
    {
        return detail::make_unary_tensor_expression< tensor_core<TensorEngine> > (expr(), [] (auto const& l) { return std::conj( l ); } );
    }

    /** @brief Computes the complex conjugate component of tensor elements within a tensor expression
     *
     * @param[in] expr tensor expression
     * @returns   complex tensor
    */
    template<class T, class D>
    auto conj(detail::tensor_expression<T,D> const& expr)
    {
        using old_tensor_type   = T;
        using value_type    = typename old_tensor_type::value_type;
        using layout_type   = typename old_tensor_type::layout_type;
        using array_type    = typename old_tensor_type::array_type;
        using extents_type  = typename old_tensor_type::extents_type;

        using complex_type = std::complex<value_type>;
        using storage_traits_t = storage_traits<array_type>;

        using t_engine = tensor_engine< 
            extents_type,
            layout_type,
            strides<extents_type>,
            typename storage_traits_t::template rebind<complex_type>
        >;

        using tensor_type = tensor_core<t_engine>;

        if( ublas::empty( detail::retrieve_extents( expr  ) ) )
            throw std::runtime_error("error in boost::numeric::ublas::conj: tensors should not be empty.");

        auto a = old_tensor_type( expr );
        auto c = tensor_type( a.extents() );

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
    template<typename TensorEngine, class D,
        std::enable_if_t< detail::is_complex_v<typename tensor_core< TensorEngine >::value_type>, int > = 0
    >
    auto real(detail::tensor_expression< tensor_core< TensorEngine > ,D > const& expr)
    {
        
        using old_tensor_type   = tensor_core< TensorEngine >;
        using complex_type  = typename old_tensor_type::value_type;
        using value_type    = typename complex_type::value_type;
        using layout_type   = typename old_tensor_type::layout_type;
        using array_type    = typename old_tensor_type::array_type;
        using extents_type  = typename old_tensor_type::extents_type;
        using storage_traits_t = storage_traits<array_type>;
        
        using t_engine = tensor_engine< 
            extents_type,
            layout_type,
            strides<extents_type>,
            typename storage_traits_t::template rebind<value_type>
        >;

        using tensor_type = tensor_core<t_engine>;

        if( ublas::empty ( detail::retrieve_extents( expr  ) ) )
            throw std::runtime_error("error in boost::numeric::ublas::real: tensors should not be empty.");

        auto a = old_tensor_type( expr );
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
    template<typename TensorEngine, class D,
        std::enable_if_t< detail::is_complex_v<typename tensor_core< TensorEngine >::value_type>, int > = 0
    >
    auto imag(detail::tensor_expression< tensor_core< TensorEngine > ,D> const& expr)
    {
        using old_tensor_type   = tensor_core< TensorEngine >;
        using complex_type  = typename old_tensor_type::value_type;
        using value_type    = typename complex_type::value_type;
        using layout_type   = typename old_tensor_type::layout_type;
        using array_type    = typename old_tensor_type::array_type;
        using extents_type  = typename old_tensor_type::extents_type;
        using storage_traits_t = storage_traits<array_type>;
        
        using t_engine = tensor_engine<
            extents_type,
            layout_type,
            strides<extents_type>,
            typename storage_traits_t::template rebind<value_type>
        >;

        using tensor_type = tensor_core<t_engine>;

        if( ublas::empty( detail::retrieve_extents( expr  ) ) )
            throw std::runtime_error("error in boost::numeric::ublas::real: tensors should not be empty.");

        auto a = old_tensor_type( expr );
        auto c = tensor_type( a.extents() );

        std::transform( a.begin(), a.end(),  c.begin(), [](auto const& l){ return std::imag(l) ; }  );

        return c;
    }

} // namespace boost::numeric::ublas

#endif
