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

#ifndef BOOST_UBLAS_TENSOR_ENGINE_IMPL_HPP
#define BOOST_UBLAS_TENSOR_ENGINE_IMPL_HPP

#include <boost/numeric/ublas/tensor/tensor_core.hpp>
#include <boost/numeric/ublas/tensor/detail/storage_traits.hpp>

namespace boost::numeric::ublas{

    namespace layout{

        template<typename...>
        struct first_order;
        
        template<>
        struct first_order<>{
            
            template<typename ExtentsType>
            struct strides{
                using extents_type = ExtentsType;
                using layout_type = ::boost::numeric::ublas::first_order;
                using strides_type = strides_t<ExtentsType,layout_type>;
            };
            
        };
        
        template<typename ExtentsType>
        struct first_order<ExtentsType>{
            
            template<typename>
            struct strides{
                using extents_type = ExtentsType;
                using layout_type = ::boost::numeric::ublas::first_order;
                using strides_type = strides_t<ExtentsType,layout_type>;
            };

        };

        template<typename...>
        struct last_order;

        template<>
        struct last_order<>{
            
            template<typename ExtentsType>
            struct strides{
                using extents_type = ExtentsType;
                using layout_type = ::boost::numeric::ublas::last_order;
                using strides_type = strides_t<ExtentsType,layout_type>;
            };
            
        };
        
        template<typename ExtentsType>
        struct last_order<ExtentsType>{
            
            template<typename>
            struct strides{
                using extents_type = ExtentsType;
                using layout_type = ::boost::numeric::ublas::last_order;
                using strides_type = strides_t<ExtentsType,layout_type>;
            };

        };

        template<typename,typename>
        struct extract_strides;

        template<typename ExtentsType, typename Layout>
        struct extract_strides
        {
            using type = typename Layout::template strides<ExtentsType>;
        };

        template<typename ExtentsType, typename Layout>
        using extract_strides_t = typename extract_strides<ExtentsType,Layout>::type;
        
    } // namespace layout

    
    template<typename...>
    struct tensor_engine;

    template<typename ExtentsType, typename LayoutType, typename StorageType>
    struct tensor_engine<ExtentsType, LayoutType, StorageType>{
        using extents_type 	        = ExtentsType;
        
        static_assert(is_extents_v<extents_type>,
            "boost::numeric::ublas::tensor_engine : please provide valid tensor extents type"
        );

        using layout_type 	        = typename layout::extract_strides_t<extents_type,LayoutType>::layout_type;
        using strides_type 	        = typename layout::extract_strides_t<extents_type,LayoutType>::strides_type;

        static_assert(is_strides_v<strides_type>,
            "boost::numeric::ublas::tensor_engine : please provide valid tensor layout type"
        );

        using storage_traits        = storage_traits<StorageType>;
        
    };
    
    template<typename LayoutType, typename StorageType>
    struct tensor_engine<LayoutType, StorageType>{
        using extents_type 	        = typename layout::extract_strides_t<void,LayoutType>::extents_type;
        
        static_assert(is_extents_v<extents_type>,
            "boost::numeric::ublas::tensor_engine : please provide valid tensor extents type"
        );

        using layout_type 	        = typename layout::extract_strides_t<void,LayoutType>::layout_type;
        using strides_type 	        = typename layout::extract_strides_t<void,LayoutType>::strides_type;

        static_assert(is_strides_v<strides_type>,
            "boost::numeric::ublas::tensor_engine : please provide valid tensor layout type"
        );

        using storage_traits        = storage_traits<StorageType>;
        
    };

    template<typename ValueType, typename Layout = first_order>
    using dynamic_tensor = tensor_core< 
        tensor_engine<
            dynamic_extents<>,
            std::conditional_t<
                std::is_same_v< Layout, first_order >,
                layout::first_order<dynamic_extents<>>,
                layout::last_order<dynamic_extents<>>
            >,
            std::vector< ValueType, std::allocator<ValueType> >
        > 
    >;
    
    namespace detail{
        
        template<typename E, typename L>
        struct select_static_strides{
            static_assert(is_static_v<E>,
                "boost::numeric::ublas::tensor_engine : please provide valid static tensor extents type"
            );

            static_assert( always_false_v<E>, "boost::numeric::ublas::detail::select_static_strides" 
                "Extents should be static tensor extents"
            );
        };
        
        template<typename T, typename L, T... Ns>
        struct select_static_strides< basic_static_extents<T,Ns...>, L >
            : std::conditional< 
                std::is_same_v< L, first_order >,
                layout::first_order< basic_static_extents<T,Ns...> >,
                layout::last_order< basic_static_extents<T,Ns...> >
            >
        {};

        template<typename E, typename L>
        using select_static_strides_t = typename select_static_strides<E,L>::type;

    } // namespace detail
    

    template<typename ValueType, typename ExtentsType, typename Layout = first_order>
    using static_tensor = tensor_core<
        tensor_engine< 
            ExtentsType,
            detail::select_static_strides_t<ExtentsType,Layout>,
            std::array< ValueType, static_product_v<ExtentsType> >
        > 
    >;

    template<typename ValueType, std::size_t N, typename Layout = first_order>
    using fixed_rank_tensor = tensor_core< 
        tensor_engine<
            dynamic_extents<N>,
            std::conditional_t<
                std::is_same_v< Layout, first_order >,
                layout::first_order<dynamic_extents<N>>,
                layout::last_order<dynamic_extents<N>>
            >,
            std::vector< ValueType, std::allocator<ValueType> >
        > 
    >;

} // namespace boost::numeric::ublas


#endif
