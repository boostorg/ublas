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

#ifndef BOOST_UBLAS_TENSOR_IMPL_HPP
#define BOOST_UBLAS_TENSOR_IMPL_HPP

#include <boost/numeric/ublas/tensor/tensor_engine.hpp>
#include <boost/numeric/ublas/tensor/layout.hpp>
#include <boost/numeric/ublas/tensor/detail/storage_traits.hpp>

namespace boost::numeric::ublas{

    template<typename ValueType, typename Layout = layout::first_order>
    using dynamic_tensor = tensor_core< 
        tensor_engine<
            dynamic_extents<>,
            Layout,
            strides< dynamic_extents<> >,
            std::vector< ValueType, std::allocator<ValueType> >
        > 
    >;
    

    template<typename ValueType, typename ExtentsType, typename Layout = layout::first_order>
    using static_tensor = tensor_core<
        tensor_engine< 
            ExtentsType,
            Layout,
            strides<ExtentsType>,
            std::array< ValueType, product(ExtentsType{}) >
        > 
    >;

    template<typename ValueType, std::size_t N, typename Layout = layout::first_order>
    using fixed_rank_tensor = tensor_core< 
        tensor_engine<
            dynamic_extents<N>,
            Layout,
            strides< dynamic_extents<N> >,
            std::vector< ValueType, std::allocator<ValueType> >
        > 
    >;

} // namespace boost::numeric::ublas


#endif
