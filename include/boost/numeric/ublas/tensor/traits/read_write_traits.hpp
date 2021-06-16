//
//  Copyright (c) 2020, Kannav Mehta, kmkannavkmehta@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef BOOST_UBLAS_TENSOR_READ_WRITE_TYPE_TRAITS_HPP
#define BOOST_UBLAS_TENSOR_READ_WRITE_TYPE_TRAITS_HPP

#include <type_traits>
#include <cstddef>

#include "../tags.hpp"

namespace boost::numeric::ublas::detail {

template<typename T>
struct is_read_write : std::false_type {};

template<>
struct is_read_write<read_write_tag> : std::true_type {};

template<typename T>
inline static constexpr bool is_read_write_v = is_read_write<T>::value;

} // namespace boost::numeric::ublas

#endif
