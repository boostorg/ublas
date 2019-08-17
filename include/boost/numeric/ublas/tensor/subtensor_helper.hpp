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

#ifndef _BOOST_UBLAS_TENSOR_SUBTENSOR_HELPER_HPP
#define _BOOST_UBLAS_TENSOR_SUBTENSOR_HELPER_HPP

#include <iostream>
#include "meta_functions.hpp"
#include "slice.hpp"
#include "extents.hpp"
#include "strides.hpp"

namespace boost::numeric::ublas::detail
{

namespace sp = boost::numeric::ublas::span;
using namespace boost::numeric::ublas::span::detail;

template <class strides_type, class span_array>
auto span_strides(strides_type const &strides, span_array const &spans)
{
    if (strides.size() != spans.size())
        throw std::runtime_error("Error in boost::numeric::ublas::subtensor::span_strides(): tensor strides.size() != spans.size()");

    using base_type = typename strides_type::base_type;
    auto span_strides = base_type(spans.size());
    auto new_stride = strides;
    for_each_list(spans, [&](auto const &I, auto const &s) {
        if constexpr (!is_list<span_array>::value)
        {
            span_strides[I] = strides[I] * s.step();
        }
        else
        {
            new_stride.step(I) = s.step();
        }
    });

    if constexpr (!is_list<span_array>::value)
    {
        using value_type = typename strides_type::value_type;
        using layout_type = typename strides_type::layout_type;
        return boost::numeric::ublas::basic_strides<value_type, layout_type>(span_strides);
    }
    else
    {
        return new_stride;
    }
}

template <class strides_type, class span_array>
auto offset(strides_type &strides, span_array const &spans)
{
    if (strides.size() != spans.size())
        throw std::runtime_error("Error in boost::numeric::ublas::subtensor::offset(): tensor strides.size() != spans.size()");

    using extents_type = typename strides_type::extents_type;

    using base_type = typename strides_type::base_type;
    auto off = 0u;
    for_each_list(spans, [&](auto const &I, auto const &s) {
        off += strides[I] * s.first();
    });

    return off;
}

template <typename size_type>
auto transform_span(sp::basic_slice<size_type> const &s, size_type const extent)
{
    using slice_type = sp::basic_slice<size_type>;
    auto const extent0 = extent - 1;
    size_type first = s.first();
    size_type last = s.last();
    size_type size = s.size();
    size_type step = s.step();
    if (size == 0)
        return slice_type(0, extent0);
    else if (first == detail::end)
        return slice_type(extent0, extent0, step);
    else if (last >= extent)
        return slice_type(first, extent0, step);
    else
        return slice_type(first, last, step);
}

template <size_t extent, typename size_type, ptrdiff_t... Args>
auto transform_span(sp::basic_slice<size_type, Args...> const &s)
{
    using slice_type = sp::basic_slice<size_type, Args...>;
    auto constexpr extent0 = extent - 1;

    if constexpr (std::is_same_v<slice_type, sp::basic_slice<size_type>>)
        return sp::slice<0, extent0,1>{};
    else if constexpr (slice_type::first() == detail::end)
        return sp::slice<extent0, extent0, slice_type::step()>{};
    else if constexpr (slice_type::last() >= extent)
        return sp::slice<slice_type::first(), extent0, slice_type::step()>{};
    else
        return sp::slice<slice_type::first(), slice_type::last(), slice_type::step()>{};
}

struct transform_spans_impl
{
    template <typename T, ptrdiff_t R, ptrdiff_t... E>
    using static_extents = boost::numeric::ublas::basic_static_extents<T, R, E...>;

    template <std::size_t r = 0, class span_array, class extents_type, class span_type, class... span_types>
    auto operator()(extents_type const &extents,
                    span_array &spans_arr,
                    span_type const &s,
                    span_types &&... spans)
    {
        if constexpr (is_list<span_array>::value && boost::numeric::ublas::detail::is_static<extents_type>::value && is_slice<span_type, span_types...>::value)
        {
            return helper<r>(extents, spans_arr, s, std::forward<span_types>(spans)...);
        }
        else
        {
            if constexpr (is_slice<span_type>::value)
            {
                spans_arr.at(r) = transform_span(s, extents.at(r));
            }
            else
            {
                using value_type = typename extents_type::value_type;
                spans_arr.at(r) = transform_span(sp::basic_slice<value_type>{static_cast<value_type>(s)}, extents.at(r));
            }
            if constexpr (sizeof...(spans) > 0)
                this->operator()<r + 1>(extents, spans_arr, std::forward<span_types>(spans)...);
        }
    }

private:
    template <std::size_t r, class span_array, typename T, ptrdiff_t R, ptrdiff_t E, ptrdiff_t... Es, class span_type, class... span_types>
    auto helper(static_extents<T, R, E, Es...> const &extents, span_array &spans_arr, span_type const &s,
                span_types &&... spans)
    {

        auto new_span_arr = push_back(spans_arr, transform_span<E>(s));
        if constexpr (sizeof...(spans) > 0)
            return helper<r + 1>(static_extents<T, sizeof...(Es), Es...>{}, new_span_arr, std::forward<span_types>(spans)...);
        else
        {
            return new_span_arr;
        }
    }
};

template <class extents_type, class... span_types,
          std::enable_if_t<boost::numeric::ublas::detail::is_extents<extents_type>::value, int> = 0>
auto generate_span_array(extents_type const &s, span_types &&... spans)
{
    constexpr auto n = sizeof...(spans);
    if (s.size() != n)
        throw std::runtime_error("Error in boost::numeric::ublas::generate_span_vector() when creating subtensor: the number of spans does not match with the tensor rank.");
    transform_spans_impl tr;
    using value_type = typename extents_type::value_type;
    if constexpr (boost::numeric::ublas::detail::is_static<extents_type>::value && is_slice<span_types...>::value)
    {
        auto l = list{};
        if constexpr (n > 0)
            return tr.template operator()<n - n>(s, l, std::forward<span_types>(spans)...);
    }
    else
    {
        std::vector<sp::basic_slice<value_type>> span_vector(n);
        if constexpr (n > 0)
            tr.template operator()<n - n>(s, span_vector, std::forward<span_types>(spans)...);
        return span_vector;
    }
}

template <typename S, typename... Ss, ptrdiff_t... E>
auto extents_helper(list<S, Ss...> const &spans, boost::numeric::ublas::detail::basic_shape<E...> const &)
{
    if constexpr (sizeof...(Ss) > 0)
    {
        auto b = boost::numeric::ublas::detail::basic_shape<E..., S::size()>{};
        return extents_helper(list<Ss...>{}, b);
    }
    else
    {
        return static_extents<E..., S::size()>{};
    }
}

template <class span_array>
auto extents(span_array const &spans)
{
    if constexpr (is_list<span_array>::value)
    {
        if constexpr (std::is_same_v<span_array, list<>>)
        {
            return list<>{};
        }
        else
        {
            boost::numeric::ublas::detail::basic_shape<> b;
            return extents_helper(spans, b);
        }
    }
    else
    {
        using base_type = typename dynamic_extents<>::base_type;
        using span_type = sp::slice<>;
        if (spans.empty())
            return dynamic_extents<>{};
        auto extents = base_type(spans.size());
        std::transform(spans.begin(), spans.end(), extents.begin(), [](span_type const &s) { return s.size(); });
        return dynamic_extents<>(extents);
    }
}

} // namespace boost::numeric::ublas::detail

namespace boost::numeric::ublas::detail
{

template <typename E, typename... S>
struct default_span_array_impl<true, E, S...>
{
    using type = decltype(generate_span_array(E{}, S{}...));
};

template <typename E, typename... S>
struct default_span_array_impl<false, E, S...>
{
    using type = std::vector<boost::numeric::ublas::span::slice<>>;
};

template <typename E, typename... S>
struct default_span_arary
{
    using type = typename default_span_array_impl<(detail::is_static<E>::value && sizeof...(S) > 0), E, S...>::type;
};

template <typename E, typename... S>
using default_span_arary_t = typename default_span_arary<E, S...>::type;

template <typename T>
struct sub_extents
{
    using type = std::conditional_t<boost::numeric::ublas::span::detail::is_list<T>::value, decltype(extents(T{})), dynamic_extents<>>;
};

template <typename T, typename L>
struct sub_strides
{
    using type = strides_t<typename sub_extents<T>::type, L>;
};
} // namespace boost::numeric::ublas::detail

#endif