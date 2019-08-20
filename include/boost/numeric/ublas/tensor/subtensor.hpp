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

#ifndef _BOOST_UBLAS_TENSOR_SUBTENSOR_HPP
#define _BOOST_UBLAS_TENSOR_SUBTENSOR_HPP

#include "subtensor_helper.hpp"
#include "fwd.hpp"

namespace boost::numeric::ublas
{

/** @brief contains a view to the tensor */
template <typename T, typename E, typename F, typename A, typename... S>
struct subtensor<tensor<T, E, F, A>, S...> : public detail::tensor_expression<
												 subtensor<tensor<T, E, F, A>, S...>,
												 subtensor<tensor<T, E, F, A>, S...>>
{

	using tensor_type = tensor<T, E, F, A>;
	using self_type = subtensor<tensor_type, S...>;

	using span_type = span::slice<>;
	using span_arr = detail::default_span_arary_t<E, S...>;

	template <class derived_type>
	using tensor_expression_type = detail::tensor_expression<self_type, derived_type>;

	template <class derived_type>
	using matrix_expression_type = matrix_expression<derived_type>;

	template <class derived_type>
	using vector_expression_type = vector_expression<derived_type>;

	using super_type = tensor_expression_type<self_type>;

	using array_type = typename tensor_type::array_type;
	using layout_type = typename tensor_type::layout_type;

	using size_type = typename tensor_type::size_type;
	using difference_type = typename tensor_type::difference_type;
	using value_type = typename tensor_type::value_type;

	using reference = typename tensor_type::reference;
	using const_reference = typename tensor_type::const_reference;

	using pointer = typename tensor_type::pointer;
	using const_pointer = typename tensor_type::const_pointer;

	using tensor_temporary_type = self_type;

	using extents_type = typename detail::sub_extents<span_arr>::type;
	using strides_type = typename detail::sub_strides<span_arr, F>::type;
	using span_strides_type = typename detail::sub_span_stride<span_arr,E,layout_type>::type;

	using matrix_type = matrix<value_type, layout_type, array_type>;
	using vector_type = vector<value_type, array_type>;

	subtensor() = delete;

	/** @brief Constructor for constructing subtensor using tensor 
	 * 
	 * @param t of type tensor
	 * 
	*/
	subtensor(tensor_type &t)
		: super_type(), spans_(), data_(t.data())
	{
		if constexpr (detail::is_dynamic<extents_type>::value)
		{	
			using ex_type = std::decay_t< decltype(t.extents()) >;
			if constexpr (detail::is_static<ex_type>::value){
				extents_ = extents_type(t.extents().to_dynamic_extents());
				strides_ = strides_type(extents_);
			}else{
				extents_ = extents_type(t.extents());
				strides_ = strides_type(t.strides());
			}
			span_strides_ = span_strides_type(t.strides());
		}
		else
		{
			span_strides_ = detail::span_strides(t.strides(), spans_);
			data_ = data_ + detail::offset(t.strides(), spans_);
		}
	}

	/** @brief Constructor for constructing subtensor using tensor and slice for dynamic slices 
	 * 
	 * @param t of type tensor
	 * @param span of basis_slice
	 * @param spans parameter pack of basic_slice
	 * 
	*/
	template <typename U, ptrdiff_t... Args, typename... span_types>
	subtensor(tensor_type &t, span::basic_slice<U,Args...> const &span, span_types &&... spans)
		:   super_type()
		  , spans_(detail::generate_span_array(t.extents(), span, std::forward<span_types>(spans)...))
		  , extents_(detail::extents(spans_))
		  , strides_(extents_)
		  , span_strides_(detail::span_strides(t.strides(), spans_))
		  , data_(t.data() + detail::offset(t.strides(), spans_))
	{
		static_assert( !span::detail::is_list<span_arr>::value, "boost::numeric::ublas::subtensor : cannot slice a static sliced subtensor " );
	}
	/** @brief returns the slices */
	TENSOR_AUTO_CONSTEXPR_RETURN const &spans() const noexcept
	{
		return spans_;
	}

	/** @brief returns the extents */
	TENSOR_AUTO_CONSTEXPR_RETURN const &extents() const noexcept
	{
		return extents_;
	}

	/** @brief returns the strides */
	TENSOR_AUTO_CONSTEXPR_RETURN const &strides() const noexcept
	{
		return strides_;
	}

	/** @brief returns the span strides */
	TENSOR_AUTO_CONSTEXPR_RETURN const &span_strides() const noexcept
	{
		return span_strides_;
	}

	/** @brief returns the element at given indeces */
	template <class... size_types>
	TENSOR_CONSTEXPR_RETURN(const_reference)
	at(size_type i, size_types... is) const
	{	
		if constexpr (sizeof...(is) == 0)
			return this->data_[i];
		else
			return this->data_[detail::access<0ul>(size_type(0), this->span_strides_, i , std::forward<size_types>(is)...)];
	}

	/** @brief returns the element at given indeces */
	template <class... size_types>
	TENSOR_CONSTEXPR_RETURN(reference)
	at(size_type i, size_types... is)
	{
		if constexpr (sizeof...(is) == 0)
			return this->data_[i];
		else
		{
			auto temp = detail::access<0ul>(size_type(0), this->span_strides_, i, std::forward<size_types>(is)...);
			return this->data_[temp];
		}
	}

	/** @brief returns the size of subtensor */
	TENSOR_CONSTEXPR_RETURN(size_type) size() const noexcept{
		size_type s = 1;
		for(auto i = 0ul; i < extents_.size(); i++){
			s *= this->extents_[i];
		}
		return s;
	}

	/** @brief returns the extents of subtensor at given index 
	 * @param i for index
	 * @return extent of indexed subtensor 
	*/
	TENSOR_CONSTEXPR_RETURN(size_type) size(size_type i) const noexcept{
		return this->extents_[i];
	}

private:
	span_arr spans_;
	extents_type extents_;
	strides_type strides_;
	span_strides_type span_strides_;
	pointer data_;
};

template <typename tensor_type>
subtensor(tensor_type &)->subtensor<tensor_type>;

template <typename tensor_type, typename U, typename... Us>
subtensor(tensor_type &, U const &, Us &&...)->subtensor<tensor_type>;

} // namespace boost::numeric::ublas

#endif