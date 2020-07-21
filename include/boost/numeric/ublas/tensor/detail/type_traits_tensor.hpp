//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google
//

#ifndef BOOST_UBLAS_TENSOR_TYPE_TRAITS_TENSOR_HPP
#define BOOST_UBLAS_TENSOR_TYPE_TRAITS_TENSOR_HPP

#include <boost/numeric/ublas/tensor/detail/basic_type_traits.hpp>
#include <boost/numeric/ublas/tensor/detail/type_traits_extents.hpp>
#include <boost/numeric/ublas/tensor/detail/storage_traits.hpp>
#include <boost/numeric/ublas/functional.hpp>

namespace boost::numeric::ublas{
    
    template<typename T> class tensor_core;

} // namespace boost::numeric::ublas


namespace boost::numeric::ublas {

    template<typename T>
    struct is_valid_tensor: std::false_type{};
    
    template<typename T>
    struct is_valid_tensor< tensor_core<T> >: std::true_type{};

    template<typename T>
    inline static constexpr bool is_valid_tensor_v = is_valid_tensor<T>::value;

    template<typename E, typename A, typename ValueType>
    struct rebind_storage{
        using type = typename storage_traits<A>::template rebind<ValueType>;
    };

    namespace detail{

        template<typename A, std::size_t N, typename T>
        struct rebind_static_storage_helper;

        template<typename A, std::size_t N>
        struct rebind_static_storage_helper<A,N,storage_static_container_tag>
        {
            using type = typename storage_traits<A>::template rebind_size<N>;
        };
        
        template<typename A, std::size_t N>
        struct rebind_static_storage_helper<A,N,storage_resizable_container_tag>
        {
            using type = A;
        };

        template<typename A, std::size_t N>
        using rebind_static_storage_helper_t = typename rebind_static_storage_helper<A,N, typename storage_traits<A>::resizable_tag >::type;
        
    } // namespace detail
    
    template<typename ValueType, typename A, typename T, T... Ns>
    struct rebind_storage< basic_static_extents<T,Ns...>, A, ValueType >
        : std::conditional<
            std::is_same_v<
                typename storage_traits<A>::resizable_tag, 
                storage_static_container_tag
            >,
            detail::rebind_static_storage_helper_t<
                typename storage_traits<A>::template rebind<ValueType>,
                ( ... * Ns)
            >,
            typename storage_traits<A>::template rebind<ValueType>
        >
    {};

    template<typename ValueType, typename A, typename T>
    struct rebind_storage< basic_static_extents<T>, A, ValueType >
    {
        using type = typename storage_traits<A>::template rebind<ValueType>;
    };

    template<typename E, typename A, typename ValueType>
    using rebind_storage_t = typename rebind_storage<E,A,ValueType>::type;

} // namespace boost::numeric::ublas

#endif
