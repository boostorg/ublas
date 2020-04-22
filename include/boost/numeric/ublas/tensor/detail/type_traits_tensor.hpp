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

namespace boost::numeric::ublas {

    template<typename T>
    struct tensor_traits;

    template<typename T>
    struct is_valid_tensor : std::false_type{};

    template<typename T>
    inline static constexpr bool is_valid_tensor_v = is_valid_tensor<T>::value;

    template<typename T, typename...Ts>
    struct tensor_rebind;

    template<typename T, typename...Ts>
    using tensor_rebind_t = typename tensor_rebind<T,Ts...>::type;


    template<typename V, typename E, typename F>
    struct result_tensor;

    template<typename V, typename E, typename F>
    using result_tensor_t = typename result_tensor<V,E,F>::type;
    
    struct tensor_tag {};

    struct dynamic_tensor_tag : tensor_tag{};
    struct static_tensor_tag : tensor_tag{};

} // namespace boost::numeric::ublas

#endif
