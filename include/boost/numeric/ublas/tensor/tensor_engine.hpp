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

#ifndef BOOST_UBLAS_TENSOR_ENGINE_IMPL_HPP
#define BOOST_UBLAS_TENSOR_ENGINE_IMPL_HPP

#include <boost/numeric/ublas/tensor/tensor_core.hpp>

namespace boost::numeric::ublas{
    
    template<typename...>
    struct tensor_engine;

    template<typename ExtentsType, typename LayoutType, typename StrideType, typename StorageType>
    struct tensor_engine<ExtentsType, LayoutType, StrideType, StorageType>{
        using extents_type 	        = ExtentsType;
        
        static_assert(is_extents_v<extents_type>,
            "boost::numeric::ublas::tensor_engine : please provide valid tensor extents type"
        );

        using layout_type 	        = LayoutType;
        using strides_type 	        = typename StrideType::template type<layout_type>;

        static_assert(is_strides_v<strides_type>,
            "boost::numeric::ublas::tensor_engine : please provide valid tensor layout type"
        );

        using storage_traits_type   = storage_traits<StorageType>;
        
    };

    template<typename ExtentsType, typename LayoutType, typename StorageType>
    struct tensor_engine<ExtentsType, LayoutType, StorageType>
        : tensor_engine< ExtentsType, LayoutType, strides<ExtentsType>, StorageType >
    {};

} // namespace boost::numeric::ublas


#endif
