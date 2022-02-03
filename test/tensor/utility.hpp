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
#include <functional>
#include <boost/numeric/ublas/tensor/layout.hpp>

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

template<std::size_t N, typename FnType>
constexpr auto static_for_each(FnType&& fn) noexcept{
    auto helper = [fn = std::forward<FnType>(fn)]<std::size_t... Is>(std::index_sequence<Is...>){
        (..., std::invoke(fn, std::integral_constant<std::size_t, Is>{}));
    };
    helper(std::make_index_sequence<N>{});
}

template<class UnaryOp, class ... Elements>
constexpr void for_each_in_tuple(std::tuple<Elements...> const& tuple, UnaryOp&& op)
{
  auto invoke_op_for_tuple = [&tuple, op = std::forward<UnaryOp>(op)]<typename IType>(IType id) {
    constexpr auto i = IType::value;
    std::invoke(op, id, std::get<i>(tuple));
  };

  static_for_each<sizeof...(Elements)>(std::move(invoke_op_for_tuple));
}

namespace boost::numeric::ublas
{

template<class UnaryOp, class TA, class TB, std::size_t ... is>
constexpr void for_each_in_index(std::index_sequence<is...>, TA const& a, TB const& b, UnaryOp&& op)
{
  (..., std::invoke(op,a,b,std::index_sequence<is>{}) );
}

}// namespace boost::numeric::ublas

//template<class UnaryOp, std::size_t ... is>
//void for_each_in_tuple(std::index_sequence<is...>, UnaryOp&& op)
//{
//  auto invoke_op_for_tuple = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
//    (..., std::invoke(op, Is, Is));
//  };

//  invoke_op_for_tuple(std::make_index_sequence<std::index_sequence<is...>::size()>{});
//}


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

#include <boost/multiprecision/cpp_bin_float.hpp>
namespace boost::numeric::ublas{
  using double_extended = boost::multiprecision::cpp_bin_float_double_extended;
  
  // using cpp_std_types = zip<int,float,std::complex<float>>::with_t<layout::first_order, layout::last_order>;
  using cpp_std_types = zip<int,float>::with_t<layout::first_order, layout::last_order>;

  using cpp_basic_std_types = zip<int,float>::with_t<layout::first_order, layout::last_order>;
  using layout_test_types = std::tuple<layout::first_order, layout::last_order>;
  
  // using test_types_with_no_layout = std::tuple<std::int32_t,std::int64_t,float,double,std::complex<float>>;
  using test_types_with_no_layout = std::tuple<int,float>;

  // using test_types = zip<int,float,std::complex<float>,double_extended>::with_t<layout::first_order, layout::last_order>;
  using test_types = zip<int,float>::with_t<layout::first_order, layout::last_order>;

  // NOTE: std::iota cannot fill the container with the complex number
  // because the complex number does not support increment operator(++)
  template<typename ValueType>
  constexpr auto iota(auto& c, ValueType v) noexcept{
     auto inc = ValueType{1};
      for(auto& el : c){
        el = v;
        v += inc;
      }
  }
} // namespace boost::numeric::ublas



// creates e.g.
// using test_types = zip<long,float>::with_t<layout::first_order,layout::last_order>; // equals
// using test_types = std::tuple< std::pair<float, layout::first_order>, std::pair<float, layout::last_order >, std::pair<double,layout::first_order>, std::pair<double,layout::last_order >
//>;
//static_assert(std::is_same< std::tuple_element_t<0,std::tuple_element_t<0,test_types2>>, float>::value,"should be float ");
//static_assert(std::is_same< std::tuple_element_t<1,std::tuple_element_t<0,test_types2>>, boost::numeric::ublas::layout::first_order>::value,"should be boost::numeric::ublas::layout::first_order ");

#endif
