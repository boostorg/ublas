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

/** @brief calculates how strides should take steps to reach next element and sets it
 * 
 * @tparam strides_type of type either static strides or dynamic strides
 * @tparam span_array of type either std::vector or type list
 * @param strides of template type strides_type
 * @param spans of template type span_array
 * 
*/
template <class strides_type, class span_array>
auto span_strides(strides_type const &strides, span_array const &spans)
{
    if (strides.size() != spans.size())
        throw std::runtime_error("Error in boost::numeric::ublas::detail::span_strides(): tensor strides.size() != spans.size()");

    using base_type = typename strides_type::base_type;
    auto span_strides = base_type(spans.size());
    auto new_stride = strides;
    using extents_type = typename strides_type::extents_type;

    for_each_list(spans, [&](auto const &I, auto const &s) {
        if constexpr (!is_static_extents<extents_type>::value)
        {
            span_strides[I] = strides[I] * abs(s.step());
        }
        else
        {
            new_stride.step(I) = abs(s.step());
        }
    });

    if constexpr (!is_static_extents<extents_type>::value)
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

/** @brief calculates the offset for starting position
 * 
 * @tparam strides_type of type either static strides or dynamic strides
 * @tparam span_array of type either std::vector or type list
 * @param strides of template type strides_type
 * @param spans of template type span_array
 * 
*/
template <class strides_type, class span_array>
auto offset(strides_type const &strides, span_array const &spans)
{
    if (strides.size() != spans.size())
        throw std::runtime_error("Error in boost::numeric::ublas::subtensor::offset(): tensor strides.size() != spans.size()");

    auto off = 0u;
    for_each_list(spans, [&](auto const &I, auto const &s) {
        off += strides[I] * s.first();
    });

    return off;
}

/** @brief normalizes the dynamic slices within the bounds of extents
 * 
 * @tparam size_type 
 * @param s of type basic_slice<size_type>
 * @param extent
 * 
*/
template <typename size_type>
auto transform_span(sp::basic_slice<size_type> const &s, size_t const extent)
{
    using slice_type = sp::basic_slice<size_type>;
    auto const extent0 = extent - 1;
    size_type first = sp::detail::noramlize_value(extent, s.first());
    size_type last = sp::detail::noramlize_value(extent, s.last());
    size_type step = s.step();

    if (s.empty())
        return slice_type(0, extent0, 1);
    else if (first == sp::detail::end)
        return slice_type(extent0, extent0, step);
    else if (last >= extent)
        return slice_type(first, extent0, step);
    else
        return slice_type(first, last, step);
}

/** @brief normalizes the static slices within the bounds of extents
 * 
 * @tparam extent of type size_t 
 * @tparam size_type 
 * @tparam Args of type parameter pack
 * @param s of type basic_slice<size_type, Args...>
 * 
*/
template <size_t extent, typename size_type, ptrdiff_t... Args>
auto transform_span(sp::basic_slice<size_type, Args...> const &s)
{
    using slice_type = sp::basic_slice<size_type, Args...>;
    auto constexpr extent0 = extent - 1;

    if constexpr (std::is_same_v<slice_type, sp::basic_slice<size_type>>)
        return sp::basic_slice<size_type, 0, extent0, 1>{};
    else
    {
        auto constexpr first = sp::detail::noramlize_value<extent, slice_type::first()>();
        auto constexpr last = sp::detail::noramlize_value<extent, slice_type::last()>();
        auto constexpr step = slice_type::step_;

        if constexpr (first == sp::detail::end)
            return sp::basic_slice<size_type, extent0, extent0, step>{};
        else if constexpr (last >= extent0)
            return sp::basic_slice<size_type, first, extent0, step>{};
        else
            return sp::basic_slice<size_type, first, last, step>{};
    }
}

/** @brief helper class or proxy class for normalizing slices */
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
            using slice_type = typename span_array::value_type;
            using value_type = typename slice_type::value_type;
            if constexpr (is_slice<span_type>::value)
            {
                spans_arr.at(r) = transform_span(s, static_cast<value_type>(extents.at(r)));
            }
            else
            {
                spans_arr.at(r) = transform_span(slice_type{static_cast<value_type>(s)}, static_cast<value_type>(extents.at(r)));
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

/** @brief generates a array or type list according to extents type with normalized slices
 * 
 * @tparam extents_type of type either static extents or dynamic extents
 * @tparam span_types of type parameter pack of slices
 * @param e of template type extents_type
 * @param spans of paramter pack type span_types
 * 
*/
template <class extents_type, class... span_types,
          std::enable_if_t<boost::numeric::ublas::detail::is_extents<extents_type>::value, int> = 0>
auto generate_span_array(extents_type const &e, span_types &&... spans)
{
    constexpr auto n = sizeof...(spans);
    if (e.size() != n)
        throw std::runtime_error("Error in boost::numeric::ublas::generate_span_vector() when creating subtensor: the number of spans does not match with the tensor rank.");
    transform_spans_impl tr;

    if constexpr (boost::numeric::ublas::detail::is_static<extents_type>::value && is_slice<span_types...>::value)
    {
        auto l = list{};
        if constexpr (n > 0)
            return tr.template operator()<0>(e, l, std::forward<span_types>(spans)...);
    }
    else
    {
        using value_type = typename slice_common_type<span_types...>::type;
        std::vector<sp::basic_slice<value_type>> span_vector(n);
        if constexpr (n > 0)
            tr.template operator()<0>(e, span_vector, std::forward<span_types>(spans)...);
        return span_vector;
    }
}

/** @brief helper function for generating new extents with type list
 * 
 * @tparam S type of data which is at first position of type list
 * @tparam Ss remaing types of type list
 * @tparam E of type parameter pack of extent
 * @param spans of type list<S,Ss...>
 * @param b of type basic_shape<E...>
 * 
*/
template <typename S, typename... Ss, ptrdiff_t... E>
auto extents_helper(list<S, Ss...> const &spans, boost::numeric::ublas::detail::basic_shape<E...> const &b)
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

/** @brief generating new extents according to slices
 * 
 * @tparam span_array type of either std::vector or type list
 * @param spans of template type span_array
 * 
*/
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
        using span_type = typename span_array::value_type;
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

/** @brief specialization for gets the span array of type list<...> containg normalized slices */
template <typename E, typename... S>
struct default_span_array_impl<true, E, S...>
{
    using type = decltype(generate_span_array(E{}, S{}...));
};

/** @brief specialization for gets the span array of type std::vector<...> */
template <typename E, typename... S>
struct default_span_array_impl<false, E, S...>
{
    using type = std::vector<boost::numeric::ublas::span::slice<>>;
};

/** @brief gets the span array type either std::vector or type list 
 *  if extents is static and slices are provided using template args
 *  then type will be type list otherwise std::vector 
 */
template <typename E, typename... S>
struct default_span_arary
{
    using type = typename default_span_array_impl<(::boost::numeric::ublas::detail::is_static<E>::value && sizeof...(S) > 0), E, S...>::type;
};

template <typename E, typename... S>
using default_span_arary_t = typename default_span_arary<E, S...>::type;

/** @brief gets the extents type either static extents or dynamic extents
 *  if span array is type list then static extents otherwise dynamic extents
 */
template <typename T>
struct sub_extents
{
    using type = std::conditional_t<boost::numeric::ublas::span::detail::is_list<T>::value, decltype(extents(T{})), dynamic_extents<>>;
};

/** @brief gets the strides type either static strides or dynamic strides
 *  if span array is type list then static strides otherwise dynamic strides
 */
template <typename T, typename L>
struct sub_strides
{
    using type = strides_t<typename sub_extents<T>::type, L>;
};

/** @brief gets the span strides type either static strides or dynamic strides
 *  if span array is type list then static strides otherwise dynamic strides
 */
template <typename T, typename E, typename L>
struct sub_span_stride
{
    using type = std::conditional_t<boost::numeric::ublas::span::detail::is_list<T>::value, decltype(span_strides(strides_t<E, L>{}, T{})), strides_t<E, L>>;
};
} // namespace boost::numeric::ublas::detail

#endif