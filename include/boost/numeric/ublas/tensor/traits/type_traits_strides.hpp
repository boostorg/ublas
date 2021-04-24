//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019, Amit Singh, amitsingh19975@gmail.com
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

    template<typename ExtentsType>
    struct strides;

    template <class E, class Layout>
    using strides_t = typename strides<E>::template type<Layout>;

} // namespace boost::numeric::ublas


#endif
