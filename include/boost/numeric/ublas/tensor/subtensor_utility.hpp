//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer and Google in producing this work
//  which firsted as a Google Summer of Code project.
//


/// \file subtensor_utility.hpp Definition for the tensor template class

#ifndef _BOOST_NUMERIC_UBLAS_TENSOR_SUBTENSOR_UTILITY_HPP_
#define _BOOST_NUMERIC_UBLAS_TENSOR_SUBTENSOR_UTILITY_HPP_

#include <algorithm>
#include <numeric>
#include <iostream>

#include "span.hpp"
#include "extents.hpp"
#include "tags.hpp"


namespace boost::numeric::ublas::detail {


/*! @brief Computes span strides for a subtensor
 *
 * span stride v is computed according to: v[i] = w[i]*s[i], where
 * w[i] is the i-th stride of the tensor
 * s[i] is the step size of the i-th span
 *
 * @param[in] strides strides of the tensor, the subtensor refers to
 * @param[in] spans vector of spans of the subtensor
*/
template<integral size_type, class Spans>
auto to_span_strides(std::vector<size_type> const& strides, Spans const& spans)
{
  if(strides.size() != spans.size())
    throw std::runtime_error("Error in boost::numeric::ublas::subtensor::to_span_strides(): tensor strides.size() != spans.size()");

  auto span_strides = std::vector<size_type>(spans.size());

  std::transform(strides.begin(), strides.end(), spans.begin(), span_strides.begin(),
                 [](auto w, auto const& s) { return w * s.step(); } );

  return std::vector<size_type>( span_strides );
}

/*! @brief Computes the data pointer offset for a subtensor
 *
 * offset is computed according to: sum ( f[i]*w[i] ), where
 * f[i] is the first element of the i-th span
 * w[i] is the i-th stride of the tensor
 *
 * @param[in] strides strides of the tensor, the subtensor refers to
 * @param[in] spans vector of spans of the subtensor
*/
template<integral Size, class Spans>
auto to_offset(std::vector<Size> const& strides, Spans const& spans)
{
  if(strides.size() != spans.size())
    throw std::runtime_error("Error in boost::numeric::ublas::subtensor::offset(): tensor strides.size() != spans.size()");

  return std::inner_product(spans.begin(), spans.end(), strides.begin(), Size(0),
                            std::plus<Size>(), [](auto const& s, Size w) {return s.first() * w; } );
}


/*! @brief Computes the extents of the subtensor.
 *
 * i-th extent is given by span[i].size()
 *
 * @param[in] spans vector of spans of the subtensor
 */
template<class spans_type>
auto to_extents(spans_type const& spans)
{
  using extents_t  = extents<>;
  using base_type  = typename extents_t::base_type;
  if(spans.empty())
    return extents_t{};
  auto extents = base_type(spans.size());
  std::transform(spans.begin(), spans.end(), extents.begin(), [](auto const& s) { return s.size(); } );
  return extents_t( extents );
}


/*! @brief Auxiliary function for subtensor which possibly transforms a span instance
 *
 * transform_span(span()     ,4) -> span(0,3)
 * transform_span(span(1,1)  ,4) -> span(1,1)
 * transform_span(span(1,3)  ,4) -> span(1,3)
 * transform_span(span(2,end),4) -> span(2,3)
 * transform_span(span(end)  ,4) -> span(3,3)
 *
 * @note span is zero-based indexed.
 *
 * @param[in] s      span that is going to be transformed
 * @param[in] extent extent that is maybe used for the tranformation
 */
template<class size_type, class span_tag>
auto transform_span(span<span_tag, size_type> const& s, std::size_t const extent)
{
  using span_type = span<span_tag, size_type>;

  std::size_t first = s.first();
  std::size_t last  = s.last ();
  std::size_t size  = s.size ();

  auto const extent0 = extent-1;

  auto constexpr is_sliced = std::is_same<span_tag,boost::numeric::ublas::tag::sliced>::value;


  if constexpr ( is_sliced ){
    if(size == 0)        return span_type(0       , extent0);
    else if(first== max) return span_type(extent0 , extent0);
    else if(last == max) return span_type(first   , extent0);
    else                 return span_type(first   , last  );
  }
  else {
    size_type step  = s.step ();
    if(size == 0)        return span_type(0       , size_type(1), extent0);
    else if(first== max) return span_type(extent0 , step, extent0);
    else if(last == max) return span_type(first   , step, extent0);
    else                 return span_type(first   , step, last  );
  }
  return span_type{};
}


template<std::size_t r, std::size_t n, class Span, class ... Spans>
void transform_spans_impl (extents<> const& extents, std::array<Span,n>& span_array, std::size_t arg, Spans&& ... spans );

template<std::size_t r, std::size_t n, class size_type, class span_tag, class Span, class ... Spans>
void transform_spans_impl(extents<> const& extents, std::array<Span, n>& span_array, span<span_tag,size_type> const& s, Spans&& ... spans)
{
  std::get<r>(span_array) = transform_span(s, extents[r]);
  static constexpr auto nspans = sizeof...(spans);
  static_assert (n==(nspans+r+1),"Static error in boost::numeric::ublas::detail::transform_spans_impl: size mismatch");
  if constexpr (nspans>0)
    transform_spans_impl<r+1>(extents, span_array, std::forward<Spans>(spans)...);
}

template<std::size_t r, std::size_t n, class Span, class ... Spans>
void transform_spans_impl (extents<> const& extents, std::array<Span,n>& span_array, std::size_t arg, Spans&& ... spans )
{
  static constexpr auto nspans = sizeof...(Spans);
  static_assert (n==(nspans+r+1),"Static error in boost::numeric::ublas::detail::transform_spans_impl: size mismatch");
  std::get<r>(span_array) = transform_span(Span(arg), extents[r]);
  if constexpr (nspans>0)
    transform_spans_impl<r+1>(extents, span_array, std::forward<Spans>(spans) ... );

}


/*! @brief Auxiliary function for subtensor that generates array of spans
 *
 * generate_span_array<span>(shape(4,3,5,2), span(), 1, span(2,end), end  )
 * -> std::array (span(0,3), span(1,1), span(2,4),span(1,1))
 *
 * @note span is zero-based indexed.
 *
 * @param[in] extents of the tensor
 * @param[in] spans spans with which the subtensor is created
 */
template<class span_type, class ... Spans>
auto generate_span_array(extents<> const& extents, Spans&& ... spans)
{
  constexpr static auto n = sizeof...(Spans);
  if(extents.size() != n)
    throw std::runtime_error("Error in boost::numeric::ublas::generate_span_vector() when creating subtensor: the number of spans does not match with the tensor rank.");
  std::array<span_type,n> span_array;
  if constexpr (n>0)
      transform_spans_impl<0>( extents, span_array, std::forward<Spans>(spans)... );
  return span_array;
}


/*! @brief Auxiliary function for subtensor that generates array of spans
 *
 * generate_span_array<span>(shape(4,3,5,2), span(), 1, span(2,end), end  )
 * -> std::array (span(0,3), span(1,1), span(2,4),span(1,1))
 *
 * @note span is zero-based indexed.
 *
 * @param[in] extents of the tensor
 * @param[in] spans spans with which the subtensor is created
 */
template<class span_type, class ... Spans>
auto generate_span_vector(extents<> const& extents, Spans&& ... spans)
{
  auto span_array = generate_span_array<span_type>(extents,std::forward<Spans>(spans)...);
  return std::vector<span_type>(span_array.begin(), span_array.end());
}


} // namespace boost::numeric::ublas::detail





#endif
