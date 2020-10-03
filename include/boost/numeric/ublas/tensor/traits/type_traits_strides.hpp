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

#ifndef BOOST_UBLAS_TENSOR_TYPE_TRAITS_STRIDES_HPP
#define BOOST_UBLAS_TENSOR_TYPE_TRAITS_STRIDES_HPP

namespace boost::numeric::ublas {

    /// @brief checks if the type is tensor strides or not
    template <class E>
    struct is_strides : std::false_type {};

    template <class E>
    inline static constexpr bool const is_strides_v = is_strides<E>::value;

    template<typename ExtentsType>
    struct strides;

    /** @brief type alias of result of strides::type
     *
     * @tparam E extents type either basic_extents or basic_static_extents
     *
     * @tparam Layout either first_order or last_order
     *
     */
    template <class E, class Layout>
    using strides_t = typename strides<E>::template type<Layout>;

} // namespace boost::numeric::ublas

#include <boost/numeric/ublas/tensor/traits/type_traits_static_strides.hpp>
#include <boost/numeric/ublas/tensor/traits/type_traits_dynamic_strides.hpp>
#include <boost/numeric/ublas/tensor/traits/type_traits_fixed_rank_strides.hpp>

#endif
