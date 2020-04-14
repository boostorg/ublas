//
// 	Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef _BOOST_UBLAS_TEST_TENSOR_UTILITY_
#define _BOOST_UBLAS_TEST_TENSOR_UTILITY_

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

#include <complex>

// To counter msvc warninig C4244
template<typename T>
struct inner_type{
    using type = T;
};

template<typename T>
struct inner_type< std::complex<T> >{
    using type = T;
};

template<typename T>
using inner_type_t = typename inner_type<T>::type;



// creates e.g.
// using test_types = zip<long,float>::with_t<first_order,last_order>; // equals
// using test_types = std::tuple< std::pair<float, first_order>, std::pair<float, last_order >, std::pair<double,first_order>, std::pair<double,last_order >
//>;
//static_assert(std::is_same< std::tuple_element_t<0,std::tuple_element_t<0,test_types2>>, float>::value,"should be float ");
//static_assert(std::is_same< std::tuple_element_t<1,std::tuple_element_t<0,test_types2>>, boost::numeric::ublas::first_order>::value,"should be boost::numeric::ublas::first_order ");

#endif
