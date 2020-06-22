//
// 	Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//


#ifndef BOOST_UBLAS_TRAITS_STORAGE_HPP
#define BOOST_UBLAS_TRAITS_STORAGE_HPP

#include <vector>
#include <array>
#include <boost/numeric/ublas/tensor/tags.hpp>

namespace boost {
namespace numeric {
namespace ublas {


template <class A>
struct storage_traits;


template <class V, class A>
struct storage_traits<std::vector<V,A>>
{
    using array_type      = std::vector<V,A>;

    using size_type       = typename array_type::size_type;
    using difference_type = typename array_type::difference_type;
    using value_type      = typename array_type::value_type;

    using reference       = typename array_type::reference;
    using const_reference = typename array_type::const_reference;

    using pointer         = typename array_type::pointer;
    using const_pointer   = typename array_type::const_pointer;

    using iterator        = typename array_type::iterator;
    using const_iterator  = typename array_type::const_iterator;

    using reverse_iterator        = typename array_type::reverse_iterator;
    using const_reverse_iterator  = typename array_type::const_reverse_iterator;

    using container_tag = storage_seq_container_tag;
    using resizable_tag = storage_resizable_container_tag;

    template<class U>
    using rebind = std::vector<U, typename std::allocator_traits<A>::template rebind_alloc<U>>;
};


template <class V, std::size_t N>
struct storage_traits<std::array<V,N>>
{
    using array_type      = std::array<V,N>;

    using size_type       = typename array_type::size_type;
    using difference_type = typename array_type::difference_type;
    using value_type      = typename array_type::value_type;

    using reference       = typename array_type::reference;
    using const_reference = typename array_type::const_reference;

    using pointer         = typename array_type::pointer;
    using const_pointer   = typename array_type::const_pointer;

    using iterator        = typename array_type::iterator;
    using const_iterator  = typename array_type::const_iterator;

    using reverse_iterator        = typename array_type::reverse_iterator;
    using const_reverse_iterator  = typename array_type::const_reverse_iterator;

    using container_tag = storage_seq_container_tag;
    using resizable_tag = storage_static_container_tag;

    template<class U>
    using rebind = std::array<U,N>;

    template<std::size_t M>
    using rebind_size = std::array<V,M>;
};

} // ublas
} // numeric
} // boost


#endif // BOOST_UBLAS_TRAITS_STORAGE_HPP
