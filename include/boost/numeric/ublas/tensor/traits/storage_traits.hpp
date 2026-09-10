//
// 	Copyright (c) 2019, Amit Singh, amitsingh19975@gmail.com
// 	Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com

//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of Google
//


#ifndef BOOST_UBLAS_TRAITS_STORAGE_HPP
#define BOOST_UBLAS_TRAITS_STORAGE_HPP

#include <array>
#include <vector>

#include "../tags.hpp"

namespace boost::numeric::ublas
{

template <class storage_type>
struct container_traits;


template <class V, class A>
struct container_traits<std::vector<V,A>>
{
    using container_type  = std::vector<V,A>;

    using size_type       = typename container_type::size_type;
    using difference_type = typename container_type::difference_type;
    using value_type      = typename container_type::value_type;

    using reference       = typename container_type::reference;
    using const_reference = typename container_type::const_reference;

    using pointer         = typename container_type::pointer;
    using const_pointer   = typename container_type::const_pointer;

    using iterator        = typename container_type::iterator;
    using const_iterator  = typename container_type::const_iterator;

    using reverse_iterator        = typename container_type::reverse_iterator;
    using const_reverse_iterator  = typename container_type::const_reverse_iterator;

    using container_tag = storage_seq_container_tag;
    using resizable_tag = storage_resizable_container_tag;

    template<class U>
    using rebind = std::vector<U, typename std::allocator_traits<A>::template rebind_alloc<U>>;
};


template <class V, std::size_t N>
struct container_traits<std::array<V,N>>
{
    using container_type  = std::array<V,N>;

    using size_type       = typename container_type::size_type;
    using difference_type = typename container_type::difference_type;
    using value_type      = typename container_type::value_type;

    using reference       = typename container_type::reference;
    using const_reference = typename container_type::const_reference;

    using pointer         = typename container_type::pointer;
    using const_pointer   = typename container_type::const_pointer;

    using iterator        = typename container_type::iterator;
    using const_iterator  = typename container_type::const_iterator;

    using reverse_iterator        = typename container_type::reverse_iterator;
    using const_reverse_iterator  = typename container_type::const_reverse_iterator;

    using container_tag = storage_seq_container_tag;
    using resizable_tag = storage_static_container_tag;

    template<class U>
    using rebind = std::array<U,N>;

    template<std::size_t M>
    using rebind_size = std::array<V,M>;
};

} // namespace boost::numeric::ublas

namespace boost::numeric::ublas
{

template <class ExtentsType, ExtentsType... E>
class basic_static_extents;

    namespace detail{
        template<typename E, typename A, typename Tag>
        struct rebind_storage_size_helper{
            using type = A;
        };

        template<typename T, typename C, T E0,T... Es>
        struct rebind_storage_size_helper<basic_static_extents<T,E0,Es...>, C, storage_static_container_tag>{
            using type = typename container_traits<C>::template rebind_size< E0 * (Es * ...) >;
        };

        template<typename T, typename C>
        struct rebind_storage_size_helper<basic_static_extents<T>, C, storage_static_container_tag>{
            using type = typename container_traits<C>::template rebind_size< 0 >;
        };
    } //namespace detail

    template<typename E, typename C>
    struct rebind_storage_size
        : detail::rebind_storage_size_helper<E,C,
            typename container_traits<C>::resizable_tag
        >
    {};

    template<typename E, typename C>
    using rebind_storage_size_t = typename rebind_storage_size<E,C>::type;
    
} // namespace boost::numeric::ublas


#endif // BOOST_UBLAS_TRAITS_STORAGE_HPP
