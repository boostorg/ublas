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

#ifndef _BOOST_UBLAS_TEST_TENSOR_UTILITY_
#define _BOOST_UBLAS_TEST_TENSOR_UTILITY_

#include <utility>
#include <tuple>

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



template<size_t I, class Tuple, class UnaryOp>
struct for_each_in_tuple_impl
{
  static constexpr unsigned N = std::tuple_size_v<Tuple>;
  static_assert(I < N, "Static Assert in boost::numeric::ublas::detail::for_each_tuple");

  using next_type = for_each_in_tuple_impl<I+1,Tuple, UnaryOp>;

  static void run(Tuple const& tuple, UnaryOp op)
  {
      op(I,std::get<I>(tuple));
      if constexpr(I < N-1){
        next_type::run(tuple, op);
      }
  }
};

template<class Tuple, class UnaryOp>
void for_each_in_tuple(Tuple const& tuple, UnaryOp op)
{
  if constexpr (std::tuple_size_v<Tuple> == 0u )
    return;

  for_each_in_tuple_impl<0,Tuple,UnaryOp>::run(tuple,op);
}


template<typename... Ts>
struct list{
    static constexpr size_t size = sizeof...(Ts);
};

template<size_t I, class CallBack, class T, class...Ts>
struct for_each_list_impl{
    constexpr decltype(auto) operator()(list<T, Ts...>, CallBack call_back){
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
// using test_types = zip<long,float>::with_t<layout::first_order,layout::last_order>; // equals
// using test_types = std::tuple< std::pair<float, layout::first_order>, std::pair<float, layout::last_order >, std::pair<double,layout::first_order>, std::pair<double,layout::last_order >
//>;
//static_assert(std::is_same< std::tuple_element_t<0,std::tuple_element_t<0,test_types2>>, float>::value,"should be float ");
//static_assert(std::is_same< std::tuple_element_t<1,std::tuple_element_t<0,test_types2>>, boost::numeric::ublas::layout::first_order>::value,"should be boost::numeric::ublas::layout::first_order ");

#endif
