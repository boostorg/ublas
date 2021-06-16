//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef BOOST_UBLAS_TENSOR_TAGS_HPP
#define BOOST_UBLAS_TENSOR_TAGS_HPP

namespace boost::numeric::ublas{
struct tensor_tag{};
struct storage_resizable_container_tag{};
struct storage_static_container_tag{};
struct storage_seq_container_tag{};
struct storage_non_seq_container_tag{};
struct read_write_tag{};
struct read_only_tag{};
} // namespace boost::numeric::ublas


namespace boost::numeric::ublas::detail::tag  {
struct unit_access    {};
struct non_unit_access{};
} // namespace boost::numeric::ublas::detail::tag


#endif // BOOST_UBLAS_TENSOR_TAGS_HPP

