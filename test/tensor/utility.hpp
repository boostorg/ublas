//  Copyright (c) 2018-2019
//  Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer and Google in producing this work
//  which started as a Google Summer of Code project.
//

#ifndef _BOOST_UBLAS_TEST_TENSOR_UTILITY_
#define _BOOST_UBLAS_TEST_TENSOR_UTILITY_

#include <boost/numeric/ublas/tensor/shape_helper.hpp>
#include <utility>

template<class ... types>
struct zip_helper;

template<class type1, class ... types3>
struct zip_helper<std::tuple<types3...>, type1>
{
	template<class ... types2>
	struct with
	{
		using type = std::tuple<types3...,std::pair<type1,types2>...>;
	};
	template<class ... types2>
	using with_t = typename with<types2...>::type;
};


template<class type1, class ... types3, class ... types1>
struct zip_helper<std::tuple<types3...>, type1, types1...>
{
	template<class ... types2>
	struct with
	{
		using next_tuple = std::tuple<types3...,std::pair<type1,types2>...>;
		using type       = typename zip_helper<next_tuple, types1...>::template with<types2...>::type;
	};

	template<class ... types2>
	using with_t = typename with<types2...>::type;
};

template<class ... types>
using zip = zip_helper<std::tuple<>,types...>;


template < ptrdiff_t index, typename S >
struct get_impl;

template < ptrdiff_t el  >
struct get_impl< 0, boost::numeric::ublas::detail::basic_shape< el > >{
    constexpr ptrdiff_t operator()() const noexcept{
        return el;
    }
};

template < ptrdiff_t el, ptrdiff_t ...Extents >
struct get_impl<0, boost::numeric::ublas::detail::basic_shape< el, Extents... >>{
    constexpr ptrdiff_t operator()() const noexcept{
        return el;
    }
};

template < ptrdiff_t index, ptrdiff_t el, ptrdiff_t ...Extents >
struct get_impl<index, boost::numeric::ublas::detail::basic_shape< el, Extents... >>{
    
    static_assert(boost::numeric::ublas::detail::basic_shape< el, Extents... >::rank > index && index >= 0,"");

    constexpr ptrdiff_t operator()() const noexcept{
        return get_impl<index - 1, boost::numeric::ublas::detail::basic_shape<Extents...> >()();
    }
};

template < ptrdiff_t index, class S >
ptrdiff_t get(){
    return get_impl<index, S >()();
}


template<size_t I, class CallBack, class...Ts>
struct for_each_tuple_impl{
    auto operator()(std::tuple<Ts...>& t, CallBack call_back){
        call_back(I,std::get<I>(t));
        if constexpr(sizeof...(Ts) - 1 > I){
            for_each_tuple_impl<I + 1,CallBack,Ts...> it;
            it(t,call_back);
        }
    }
};

template<class CallBack, class... Ts>
auto for_each_tuple(std::tuple<Ts...>& t, CallBack call_back){
    for_each_tuple_impl<0,CallBack,Ts...> f;
    f(t,call_back);
}


template<typename... Ts>
struct list{
    static constexpr size_t size = sizeof...(Ts);
};

template<size_t I, class CallBack, class T, class...Ts>
struct for_each_list_impl{
    constexpr decltype(auto) operator()(list<T, Ts...> l, CallBack call_back){
        using new_list = list<Ts...>;
        using value_type = T;
        call_back(I,value_type{});
        
        if constexpr(new_list::size != 0){
            for_each_list_impl<I + 1,CallBack, Ts...> it;
            it(new_list{},call_back);
        }
    }
};


template<class CallBack, class... Ts>
auto for_each_list(list<Ts...> l, CallBack call_back){
    for_each_list_impl<0,CallBack,Ts...> f;
    f(l,call_back);
}


// creates e.g.
// using test_types = zip<long,float>::with_t<first_order,last_order>; // equals
// using test_types = std::tuple< std::pair<float, first_order>, std::pair<float, last_order >, std::pair<double,first_order>, std::pair<double,last_order >
//>;
//static_assert(std::is_same< std::tuple_element_t<0,std::tuple_element_t<0,test_types2>>, float>::value,"should be float ");
//static_assert(std::is_same< std::tuple_element_t<1,std::tuple_element_t<0,test_types2>>, boost::numeric::ublas::first_order>::value,"should be boost::numeric::ublas::first_order ");

#endif
