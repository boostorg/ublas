//
//  Copyright (c) 2018-2019, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB, Ettlingen, Germany
//


#ifndef BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_HPP

#include <boost/numeric/ublas/tensor/extents_functions.hpp>
#include <boost/numeric/ublas/tensor/dynamic_extents.hpp>
#include <boost/numeric/ublas/tensor/static_extents.hpp>

namespace boost::numeric::ublas {

template <class T, ptrdiff_t R, ptrdiff_t... E>
constexpr bool operator==(basic_extents<T> const& lhs, basic_static_extents<T,R,E...> const& rhs){
    if( rhs.size() != lhs.size() ){
        return false;
    }

    for(auto i = 0u; i < rhs.size(); i++){
        if(rhs.at(i) != lhs.at(i)){
            return false;
        }
    }
    return true;
}

template <class T, ptrdiff_t R, ptrdiff_t... E>
constexpr bool operator==(basic_static_extents<T,R,E...> const& lhs, basic_extents<T> const& rhs){
    return (rhs == lhs);
}

template <class T, ptrdiff_t R, ptrdiff_t... E>
constexpr bool operator!=(basic_extents<T> const& lhs, basic_static_extents<T,R,E...> const& rhs){
    return !(lhs == rhs);
}

template <class T, ptrdiff_t R, ptrdiff_t... E>
constexpr bool operator!=(basic_static_extents<T,R,E...> const& lhs, basic_extents<T> const& rhs){
    return rhs != lhs;
}


}

#endif