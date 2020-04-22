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

#ifndef _BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_HPP_
#define _BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_HPP_

#include <boost/numeric/ublas/tensor/detail/extents_functions.hpp>
#include <boost/numeric/ublas/tensor/dynamic_extents.hpp>
#include <boost/numeric/ublas/tensor/fixed_rank_extents.hpp>
#include <boost/numeric/ublas/tensor/static_extents.hpp>

namespace boost::numeric::ublas {

    template <class LExtents, class RExtents,
        std::enable_if_t<is_extents_v<LExtents> && is_extents_v<RExtents>, int> = 0
    >
    constexpr bool operator==(LExtents const& lhs, RExtents const& rhs){
        static_assert(is_extents_v<RExtents> && is_extents_v<LExtents>,
            "boost::numeric::ublas::operator==() : invalid type, type should be an extents");
        if( lhs.size() != rhs.size() ){
            return false;
        }else{
            return std::equal(lhs.begin(), lhs.end(), rhs.begin());
        }
    }

    template <class LExtents, class RExtents,
        std::enable_if_t<is_extents_v<LExtents> && is_extents_v<RExtents>, int> = 0
    >
    constexpr bool operator!=(LExtents const& lhs, RExtents const& rhs){
        static_assert(is_extents_v<RExtents> && is_extents_v<LExtents>,
            "boost::numeric::ublas::operator!=() : invalid type, type should be an extents");
        return !(lhs == rhs);
    }

    template <class Extents,
        std::enable_if_t<is_extents_v<Extents>, int> = 0
    >
    std::ostream& operator<<(std::ostream& os, Extents const& e){
        static_assert(is_extents_v<Extents>,
            "boost::numeric::ublas::operator<<() : invalid type, type should be an extents");
        return os<<to_string(e);
    }


    namespace detail{
        template<std::size_t... N>
        struct dynamic_extents_impl;

        template <std::size_t N> struct dynamic_extents_impl<N> {
            using type = basic_fixed_rank_extents<std::size_t, N>;
        };

        template <> struct dynamic_extents_impl<> {
            using type = basic_extents<std::size_t>;
        };

    } // namespace detail


    template<std::size_t... E>
    using dynamic_extents = typename detail::dynamic_extents_impl<E...>::type;


}


#endif
