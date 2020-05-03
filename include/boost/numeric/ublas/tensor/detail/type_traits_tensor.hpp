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

#include <boost/numeric/ublas/tensor/detail/basic_type_traits.hpp>
#include <boost/numeric/ublas/tensor/detail/type_traits_extents.hpp>

namespace boost::numeric::ublas{
    template<typename T> class basic_tensor;
    
    template< typename T, typename F > struct dynamic_tensor;

    template< typename T, std::size_t R, typename F > struct fixed_rank_tensor;

    template< typename T, typename E, typename F > struct static_tensor;
} // namespace boost::numeric::ublas


namespace boost::numeric::ublas {

    template<typename T>
    struct tensor_traits;

    template<typename T>
    struct is_valid_tensor: std::is_base_of< basic_tensor<T>, T >{};

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

namespace boost::numeric::ublas{
    
    template<typename T, typename F, typename NewValue>
    struct tensor_rebind< dynamic_tensor<T, F>, NewValue > {
        using type = dynamic_tensor< NewValue, F >;
    };

    template<typename T, typename F, typename NewValue, typename NewLayout>
    struct tensor_rebind< dynamic_tensor<T, F>, NewValue, NewLayout > {
        using type = dynamic_tensor< NewValue, NewLayout >;
    };

    template<typename T, typename F>
    struct is_static< dynamic_tensor<T, F> > : std::false_type{};

    template<typename T, typename F>
    struct is_static_rank< dynamic_tensor<T, F> > : std::false_type{};

    template<typename T, typename F>
    struct is_dynamic< dynamic_tensor<T, F> > : std::true_type{};

    template<typename T, typename F>
    struct is_dynamic_rank< dynamic_tensor<T, F> > : std::true_type{};

    template<typename V, typename F>
    struct result_tensor< V, dynamic_extents<>, F >{
        using type = dynamic_tensor< V, F >;
    };

} // namespace boost::numeric::ublas::detail


namespace boost::numeric::ublas{
    
    template<typename T, std::size_t R, typename F, typename NewValue>
    struct tensor_rebind< fixed_rank_tensor<T, R, F>, NewValue > {
        using type = fixed_rank_tensor< NewValue, R, F >;
    };
    
    template<typename T, std::size_t R, typename F, typename NewValue, typename NewExtents>
    struct tensor_rebind< fixed_rank_tensor<T, R, F>, NewValue, NewExtents > {
        using type = fixed_rank_tensor< NewValue, NewExtents::_size, F >;
    };
    
    template<typename T, std::size_t R, typename F, typename NewValue, typename NewExtents, typename NewLayout>
    struct tensor_rebind< fixed_rank_tensor<T, R, F>, NewValue, NewExtents, NewLayout > {
        using type = fixed_rank_tensor< NewValue, NewExtents::_size, NewLayout >;
    };

    template<typename T, std::size_t R, typename F>
    struct is_static< fixed_rank_tensor<T, R, F> > : std::false_type{};
    
    template<typename T, std::size_t R, typename F>
    struct is_static_rank< fixed_rank_tensor<T, R, F> > : std::true_type{};
    
    template<typename T, std::size_t R, typename F>
    struct is_dynamic< fixed_rank_tensor<T, R, F> > : std::true_type{};
    
    template<typename T, std::size_t R, typename F>
    struct is_dynamic_rank< fixed_rank_tensor<T, R, F> > : std::false_type{};

    template<typename V, typename F, typename T, std::size_t R>
    struct result_tensor< V, basic_fixed_rank_extents<T,R>, F >{
        using type = fixed_rank_tensor< V, R, F >;
    };

} // namespace boost::numeric::ublas::detail

namespace boost::numeric::ublas{

    template<typename T, typename E, typename F, typename NewValue>
    struct tensor_rebind< static_tensor<T, E, F>, NewValue > {
        using type = static_tensor< NewValue, E, F >;
    };

    template<typename T, typename E, typename F, typename NewValue, typename NewExtents>
    struct tensor_rebind< static_tensor<T, E, F>, NewValue, NewExtents > {
        using type = static_tensor< NewValue, NewExtents, F >;
    };

    template<typename T, typename E, typename F, typename NewValue, typename NewExtents, typename NewLayout>
    struct tensor_rebind< static_tensor<T, E, F>, NewValue, NewExtents, NewLayout > {
        using type = static_tensor< NewValue, NewExtents, NewLayout >;
    };


    template<typename T, typename E, typename F>
    struct is_static< static_tensor<T, E, F> > : std::true_type{};
    
    template<typename T, typename E, typename F>
    struct is_static_rank< static_tensor<T, E, F> > : std::true_type{};
    
    template<typename T, typename E, typename F>
    struct is_dynamic< static_tensor<T, E, F> > : std::false_type{};
    
    template<typename T, typename E, typename F>
    struct is_dynamic_rank< static_tensor<T, E, F> > : std::false_type{};

    template<typename V, typename F, typename T, T... Es>
    struct result_tensor< V, basic_static_extents<T,Es...>, F >{
        using type = static_tensor< V, basic_static_extents<T,Es...>, F >;
    };

} // namespace boost::numeric::ublas::detail


#endif
