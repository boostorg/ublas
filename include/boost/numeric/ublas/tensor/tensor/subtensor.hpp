//
//  Copyright (c) 2021, Kannav Mehta, kmkannavkmehta@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//:


/// \file subtensor.hpp Definition for the subtensor template class

#ifndef BOOST_UBLAS_SUBTENSOR_HPP
#define BOOST_UBLAS_SUBTENSOR_HPP

#include "../algorithms.hpp"
#include "../concepts.hpp"
#include "../expression.hpp"
#include "../expression_evaluation.hpp"
#include "../extents.hpp"
#include "../index.hpp"
#include "../index_functions.hpp"
#include "../layout.hpp"
#include "../span.hpp"
#include "../tags.hpp"
#include "../traits/read_write_traits.hpp"
#include "../type_traits.hpp"
#include "subtensor_engine.hpp"
#include "tensor_engine.hpp"

#include <initializer_list>

namespace boost::numeric::ublas {

template <class T>
class tensor_core<subtensor_engine<T>>
  : public detail::tensor_expression<tensor_core<subtensor_engine<T>>,
                                     tensor_core<subtensor_engine<T>>> {
public:
	using engine_type = subtensor_engine<T>;
	using self_type   = tensor_core<engine_type>;

	template <class derived_type>
	using tensor_expression_type =
		detail::tensor_expression<self_type, derived_type>;

	template <typename container> struct subtensor_iterator {
	};

	static constexpr bool is_const =
		std::is_const<std::remove_reference_t<T>>::value;

	using container_type = typename engine_type::container_type;
	using layout_type    = typename engine_type::layout_type;
	using extents_type   = typename engine_type::extents_type;
	using strides_type   = typename extents_type::base_type;

	using container_traits_type = container_traits<container_type>;

	using size_type       = typename container_traits_type::size_type;
	using difference_type = typename container_traits_type::difference_type;
	using value_type      = typename container_traits_type::value_type;

	using reference =
		std::conditional_t<is_const,
						typename container_traits_type::const_reference,
						typename container_traits_type::reference>;
	using const_reference = typename container_traits_type::const_reference;

	using pointer =
		std::conditional_t<is_const,
						typename container_traits_type::const_pointer,
						typename container_traits_type::pointer>;
	using const_pointer = typename container_traits_type::const_pointer;

	using iterator = typename self_type::subtensor_iterator<container_type>;
	using const_iterator =
		typename self_type::subtensor_iterator<container_type> const;

	using reverse_iterator = typename container_traits_type::reverse_iterator;
	using const_reverse_iterator =
		typename container_traits_type::const_reverse_iterator;

	using container_tag = typename container_traits_type::container_tag;
	using resizable_tag = typename container_traits_type::resizable_tag;

	using span_type = span<std::size_t>;

	tensor_core(const tensor_core& tens) {}

	template <typename eng> tensor_core(const tensor_core<eng>& tens) {}
};

template <typename T> using subtensor = tensor_core<subtensor_engine<T>>;

template <typename T> auto make_subtensor(const T& tens)
{
  return subtensor<T>(tens);
}

}   // namespace boost::numeric::ublas

#endif
