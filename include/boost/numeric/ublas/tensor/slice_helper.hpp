//
//  Copyright (c) 2018-2019, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef _BOOST_UBLAS_TENSOR_SLICE_HELPER_HPP
#define _BOOST_UBLAS_TENSOR_SLICE_HELPER_HPP

#include <iostream>
#include <vector>
#include "fwd.hpp"
#include "slice.hpp"

namespace boost::numeric::ublas::span::detail
{

template<ptrdiff_t x>
inline static constexpr auto static_abs = x < 0 ? -x : x;

inline static constexpr auto end = std::numeric_limits<ptrdiff_t>::max();
template <typename T, ptrdiff_t f_, ptrdiff_t l_, ptrdiff_t s_, ptrdiff_t sz = ( ( (l_ - f_) / static_abs<s_> ) + 1l)>
struct normalized_slice
{
    using type = slice_helper<T, f_, l_, s_, sz>;
};

template <typename T, ptrdiff_t f_, ptrdiff_t l_, ptrdiff_t s_>
struct normalized_slice_helper
{
    constexpr decltype(auto) operator()() const
    {
        if constexpr (f_ == l_)
        {
            return normalized_slice<T, f_, l_, s_, 1l>{};
        }
        else
        {
            static_assert(s_ != 0, "Error in basic_static_span::basic_static_span : cannot have a s_ equal to zero.");
            static_assert( s_ > 0, "Error in basic_static_span::basic_static_span : cannot have a s_ less than 0");
            if constexpr ( f_ >= 0 && l_ >= 0 ){
                
                if constexpr( f_ > l_ && s_ > 0 ){
                    throw std::out_of_range("Error in basic_static_span::basic_static_span: l_ is smaller than f_");
                }

                if constexpr (l_ == detail::end)
                {
                    return normalized_slice<T, f_, l_, s_, detail::end>{};
                }
                else
                {
                    return normalized_slice<T, f_, (l_ - (l_ - f_) % static_abs<s_> ), s_>{};
                }

            }else{
                return normalized_slice<T, f_, l_, s_, 1l>{};
            }
        }
    }
};

template <typename T, ptrdiff_t f_, ptrdiff_t l_, ptrdiff_t s_>
using slice_helper_t = typename decltype(normalized_slice_helper<T, f_, l_, s_>{}())::type;


template <typename... Ts>
struct list
{
    TENSOR_AUTO_CONSTEXPR_RETURN size() const noexcept
    {
        return sizeof...(Ts);
    }
};

template <typename T>
struct is_list : std::false_type
{
};

template <typename... Ts>
struct is_list<list<Ts...>> : std::true_type
{
};

template <typename T, typename... Ts>
TENSOR_AUTO_CONSTEXPR_RETURN push_front(list<Ts...> , T ) -> list<T, Ts...>;

template <typename T, typename... Ts>
TENSOR_AUTO_CONSTEXPR_RETURN push_back(list<Ts...>, T)->list<Ts..., T>;

template <typename T, typename... Ts>
TENSOR_AUTO_CONSTEXPR_RETURN pop_front(list<T, Ts...> ) -> list<Ts...>;

template <typename T, typename... Ts>
TENSOR_AUTO_RETURN pop_and_get_front(list<T, Ts...>)
{
    return std::make_pair(T{}, list<Ts...>{});
}

template <size_t I, typename T, typename... Ts>
auto get_helper(list<T, Ts...>)
{
    if constexpr (I == 0)
    {
        return T{};
    }
    else
    {
        return get_helper<I - 1>(list<Ts...>{});
    }
}

template <size_t I, typename... Ts>
TENSOR_AUTO_CONSTEXPR_RETURN get(list<Ts...> const &l)
{
    if constexpr ( sizeof...(Ts) <= I ){
        throw std::out_of_range("boost::numeric::ublas::span::detail::get() : out of bound");
    }else{
        return get_helper<I>(l);
    }
}

template <size_t I, class CallBack, class T, class... Ts>
struct for_each_list_impl
{
    constexpr decltype(auto) operator()(list<T, Ts...> const &l, CallBack call_back)
    {
        using new_list = list<Ts...>;
        using value_type = T;
        call_back(I, value_type{});

        if constexpr (sizeof...(Ts) != 0)
        {
            for_each_list_impl<I + 1, CallBack, Ts...> it;
            it(new_list{}, call_back);
        }
    }

    template<typename U>
    constexpr decltype(auto) operator()(std::vector<basic_slice<U>> const &l, CallBack call_back)
    {
        for (auto i = 0u; i < l.size(); i++)
        {
            call_back(i, l[i]);
        }
    }
};

template <class CallBack, class... Ts>
auto for_each_list(list<Ts...> const &l, CallBack call_back)
{
    for_each_list_impl<0, CallBack, Ts...> f;
    f(l, call_back);
}

template <class CallBack, typename T>
auto for_each_list(std::vector<basic_slice<T>> const &l, CallBack call_back)
{
    for_each_list_impl<0, CallBack, int> f;
    f(l, call_back);
}

template<size_t I, typename... Ts>
TENSOR_AUTO_CONSTEXPR_RETURN get( list<Ts...> l, size_t i ){
    static_assert( I < 3, "boost::numeric::ublas::span::detail::get : invalid index");
    if ( sizeof...(Ts) <= i) throw std::out_of_range("boost::numeric::ublas::span::detail::get : out of bound");
    
    size_t val = end;
    for_each_list( l, [&]( auto const& j, auto const& s ){
        if ( i == j ){
            if constexpr ( I == 0 ) val = s.first();
            else if constexpr ( I == 1 ) val = s.last();
            else val = s.step();
        }
    });
    return val;
}


template<size_t I, typename T>
TENSOR_AUTO_CONSTEXPR_RETURN get( std::vector< basic_slice<T> > const& v, size_t i ){
    static_assert( I < 3, "boost::numeric::ublas::span::detail::get : invalid index");
    if ( v.size() <= i) throw std::out_of_range("boost::numeric::ublas::span::detail::get : out of bound");
    
    size_t val = end;
    auto s = v[i];
    if constexpr ( I == 0 ) val = s.first();
    else if constexpr ( I == 1 ) val = s.last();
    else val = s.step();
    return val;
}

template<typename... Ts>
struct slice_common_type;

template<typename T, typename... Ts>
struct slice_common_type<T, Ts...>{
    using type = ptrdiff_t;
};

template<typename U, ptrdiff_t... Args, typename... Ts>
struct slice_common_type<basic_slice<U,Args...>, Ts...>{
    using type = std::common_type_t<U, typename slice_common_type<Ts...>::type>;
};

template<>
struct slice_common_type<>{
    using type = typename slice_common_type<int>::type;
};

template<typename T>
TENSOR_AUTO_CONSTEXPR_RETURN noramlize_value(T ext, T val) {
    if ( val < 0 ){
        auto const ret = ext + val;
        if ( ret < 0 ){
            throw std::out_of_range("boost::numeric::ublas::span::detail::normalize_val : invalid slice ");
        }
        return ret;
    }else{
        return val;
    }
} 


template<ptrdiff_t ext, ptrdiff_t val>
TENSOR_AUTO_CONSTEXPR_RETURN noramlize_value() {
    if constexpr ( val < 0 ){
        constexpr auto const ret = ext + val;
        if constexpr ( ret < 0 ){
            throw std::out_of_range("boost::numeric::ublas::span::detail::normalize_val : invalid slice ");
        }
        return ret;
    }else{
        return val;
    }
} 

} // namespace boost::numeric::ublas::span::detail

#endif