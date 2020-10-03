//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef BOOST_UBLAS_TENSOR_TYPE_TRAITS_TENSOR_HPP
#define BOOST_UBLAS_TENSOR_TYPE_TRAITS_TENSOR_HPP

#include <boost/numeric/ublas/tensor/traits/basic_type_traits.hpp>
#include <boost/numeric/ublas/tensor/traits/type_traits_extents.hpp>
#include <boost/numeric/ublas/tensor/traits/storage_traits.hpp>

namespace boost::numeric::ublas{
    
    template<typename T> class tensor_core;

} // namespace boost::numeric::ublas


namespace boost::numeric::ublas {

    /// @brief Checks if the type is valid tensor
    template<typename T>
    struct is_valid_tensor: std::false_type{};
    
    template<typename T>
    struct is_valid_tensor< tensor_core<T> >: std::true_type{};

    template<typename T>
    inline static constexpr bool is_valid_tensor_v = is_valid_tensor<T>::value;

} // namespace boost::numeric::ublas

#endif
