//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB, Ettlingen Germany
//


#ifndef _BOOST_UBLAS_TENSOR_EXTENTS_
#define _BOOST_UBLAS_TENSOR_EXTENTS_

#include <vector>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <initializer_list>
#include <algorithm>

#include <cassert>

namespace boost { namespace numeric { namespace ublas {

template<class __int_type>
class basic_extents
{
	using base_type = std::vector<__int_type>;
	static_assert( std::numeric_limits<typename base_type::value_type>::is_integer, "Static error in basic_layout: type must be of type integer.");
	static_assert(!std::numeric_limits<typename base_type::value_type>::is_signed,  "Static error in basic_layout: type must be of type unsigned integer.");

public:
	using value_type = typename base_type::value_type;
	using const_reference = typename base_type::const_reference;
	using reference = typename base_type::reference;
	using size_type = typename base_type::size_type;
	using const_pointer = typename base_type::const_pointer;
	using const_iterator = typename base_type::const_iterator;

	constexpr explicit basic_extents()
		: _base{}
	{
	}

	explicit basic_extents(base_type const& b)
		: _base(b)
		//: _base(eliminate_last_ones(b))
	{
		if (!this->valid())
			throw std::length_error("Error in basic_extents::basic_extents() : shape tuple is not a valid permutation: has zero elements.");
	}

	explicit basic_extents(base_type && b)
		: _base(std::move(b)) //_base(std::move(eliminate_last_ones(std::move(b))))
	{
		if (!this->valid())
			throw std::length_error("Error in basic_extents::basic_extents() : shape tuple is not a valid permutation: has zero elements.");
	}


	explicit basic_extents(std::initializer_list<value_type> l)
	    : basic_extents( base_type(std::move(l)) )
	{
	}

	template<class InputIt>
	basic_extents(InputIt first, InputIt last)
	    : basic_extents ( base_type( first,last ) )
	{		
	}

	basic_extents(basic_extents const& l )
	    : _base(l._base)
	{
	}

	basic_extents(basic_extents && l )
	    : _base(std::move(l._base))
	{
	}

	~basic_extents() = default;

	basic_extents& operator=(basic_extents other)
	{
		swap (*this, other);
		return *this;
	}

	friend void swap(basic_extents& lhs, basic_extents& rhs) {
		std::swap(lhs._base   , rhs._base   );
	}



	bool is_scalar() const
	{
		return _base.size() != 0 &&
		       std::all_of(_base.begin(), _base.end(),
		                   [](const_reference a){ return a == 1;});
	}

	bool is_vector() const
	{
		if(_base.size() == 0)
			return false;

		if(_base.size() == 1)
			return _base.at(0) > 1;

		auto greater_one = [](const_reference a){ return a >  1;};
		auto equal_one   = [](const_reference a){ return a == 1;};

		return  std::any_of(_base.begin(),   _base.begin()+2, greater_one) &&
		        std::any_of(_base.begin(),   _base.begin()+2, equal_one  ) &&
		        std::all_of(_base.begin()+2, _base.end(),     equal_one);
	}

	bool is_matrix() const
	{
		if(_base.size() < 2)
			return false;

		auto greater_one = [](const_reference a){ return a >  1;};
		auto equal_one   = [](const_reference a){ return a == 1;};

		return  std::all_of(_base.begin(),   _base.begin()+2, greater_one) &&
		        std::all_of(_base.begin()+2, _base.end(),     equal_one  );
	}

	bool is_tensor() const
	{
		if(_base.size() < 3)
			return false;

		auto greater_one = [](const_reference a){ return a > 1;};

		return std::any_of(_base.begin()+2, _base.end(), greater_one);
	}


	const_reference back() const
	{
		return _base.back();
	}
	const_reference front() const
	{
		return _base.front();
	}

	const_pointer data() const
	{
		return &_base[0];
	}

	const_reference operator[] (size_type p) const
	{
		return _base[p];
	}

	const_reference at (size_type p) const
	{
		return _base.at(p);
	}

	reference operator[] (size_type p)
	{
		return _base[p];
	}

	reference at (size_type p)
	{
		return _base.at(p);
	}


	bool empty() const
	{
		return _base.empty();
	}

	size_type size() const
	{
		return _base.size();
	}


	bool valid() const
	{
		return this->size() > 1 && std::none_of(_base.begin(), _base.end(),
																						[](const_reference a){ return a == value_type(0); });
	}

	size_type product() const
	{
		if(_base.empty())
			return 0;
		else
			return std::accumulate(_base.begin(), _base.end(), 1ul, std::multiplies<size_t>());
	}


	basic_extents squeeze() const
	{
		if(this->empty() || this->size() == 2)
			return *this;

		basic_extents newb;
		auto not_equal_one = [](const_reference a){ return a != 1;};
		std::remove_copy_if(this->_base.begin(), this->_base.end(), std::insert_iterator<base_type>(newb._base,newb._base.begin()),not_equal_one);
		return newb;

	}

	void clear()
	{
		this->_base.clear();
	}

	bool operator == (basic_extents const& b) const
	{
		return _base == b._base;
	}

	bool operator != (basic_extents const& b) const
	{
		return _base != b._base;
	}

	bool operator == (basic_extents && b)
	{
		return _base == b._base;
	}

	bool operator != (basic_extents && b)
	{
		return _base != b._base;
	}

	bool operator == (base_type const& b) const
	{
		return _base == b;
	}

	bool operator != (base_type const& b) const
	{
		return _base != b;
	}

	const_iterator
	begin() const
	{
		return _base.begin();
	}

	const_iterator
	end() const
	{
		return _base.end();
	}

	base_type const& base() const { return _base; }

//	template<class __other_allocator_type>
//	friend basic_extents<value_type,__other_allocator_type>
//	basic_layout::operator()(basic_extents<value_type, __other_allocator_type> const&);

private:

	static base_type
	eliminate_last_ones(base_type const& bb)
	{
		if(bb.size() == 2 || bb.size() == 0)
			return bb;

		if(bb.size() == 1)
			return base_type{bb[0],bb[0]};

		auto not_equal_one   = [](const_reference a){ return a != 1;};
		auto rit             = std::find_if(bb.rbegin(), bb.rend()-2, not_equal_one);
		auto  it             = typename base_type::const_iterator((rit).base());
		assert((it - bb.begin()) >= 2);
		return base_type(bb.begin(),it);
	}

	static base_type
	eliminate_last_ones(base_type && bb)
	{
		if(bb.size() == 2 || bb.size() == 0)
			return base_type(bb);

		if(bb.size() == 1)
			return base_type{bb[0],bb[0]};

		auto not_equal_one   = [](const_reference a){ return a != 1;};
		auto rit             = std::find_if(bb.rbegin(), bb.rend()-2, not_equal_one);
		auto  it             = typename base_type::iterator((rit).base());
		assert((it - bb.begin()) >= 2);
		return base_type(bb.begin(),it);
	}

	base_type _base;

};

using shape = basic_extents<std::size_t>;

}}}


#endif
