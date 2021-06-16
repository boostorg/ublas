//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer and Google in producing this work
//  which started as a Google Summer of Code project.
//


#ifndef BOOST_UBLAS_TENSOR_SPAN
#define BOOST_UBLAS_TENSOR_SPAN

#include <typeinfo>
#include <limits>
#include <stdexcept>
#include <ostream>


#include "concepts.hpp"


namespace boost::numeric::ublas {

/** \class span
 * \ingroup Core_Module
 *
 * \brief Selection operator class to initialize stl::multi_subarray
 *
 * This class is used to generate stl::multi_subarray from stl::multi_array and to
 * work on views.
 * \note zero based indexing is used.
 *
 */

static constexpr inline std::size_t max = std::numeric_limits<std::size_t>::max();

template<typename unsigned_type = std::size_t>
class span
{
public:
	using value_type = std::size_t;

	// covers the complete range of one dimension
	// e.g. a(:)
	constexpr explicit span()
		: first_{}
		, last_ {}
		, step_ {}
		, size_ {}
	{}


	// covers a linear range of one dimension
	// e.g. a(1:3:n)
	span(value_type f, value_type s, value_type l)
		: first_(f)
		, last_ (l)
		, step_ (s)
	{
		if(f == l){
			last_ = l;
			size_ = value_type(1);
		}
		else {
			if(s == 0 && f != l)
				throw std::runtime_error("Error in span::span : cannot have a step_ equal to zero.");

			if(f > l)
				throw std::runtime_error("Error in span::span: last_ is smaller than first");

			last_ = l - ((l-f)%s);
			size_ = (last_-first_)/s+value_type(1);
		}
	}

	// covers only one index of one dimension
	// e.g. a(1) or a(end)
	span(value_type f, value_type l)
		: span(f,1,l)
	{
	}

	span(span const& other)
		: first_(other.first_)
		, last_ (other.last_ )
		, step_ (other.step_ )
		, size_ (other.size_ )
	{
	}

	span& operator=(span const& other)
	{
		first_ = other.first_;
		last_  = other.last_ ;
		step_  = other.step_ ;
		size_  = other.size_ ;
		return *this;
	}

  inline auto first() const {return first_; }
  inline auto last () const {return last_ ; }
  inline auto step () const {return step_ ; }
  inline auto size () const {return size_ ; }

	~span() = default;

  inline value_type operator[] (std::size_t idx) const
	{
		return first_ + idx * step_;
	}

  inline span operator()(const span &rhs) const
	{
		auto const& lhs = *this;
		return span(
					rhs.first_*lhs.step_ + lhs.first_,
					lhs.step_ *rhs.step_,
					rhs.last_ *lhs.step_ + lhs.first_ );
	}

protected:

	value_type first_, last_ , step_, size_;
};



template<integral unsigned_type_lhs, integral unsigned_type_rhs>
inline auto ran(unsigned_type_lhs f, unsigned_type_rhs l)
{
	return span(f,l);
}

template<integral unsigned_type_left, integral unsigned_type_middle, integral unsigned_type_right>
inline auto ran(unsigned_type_left f, unsigned_type_middle s, unsigned_type_right l)
{
	return span(f,s,l);
}

} // namespace


template <class unsigned_type>
std::ostream& operator<< (std::ostream& out,  boost::numeric::ublas::span<unsigned_type> const& s)
{
  return out << "[" << s.first() << ":" << s.step() << ":" << s.last() << "]" << std::endl;
}

template<class unsigned_type_lhs, class unsigned_type_rhs>
inline bool operator==(
		boost::numeric::ublas::span<unsigned_type_lhs> const& lhs,
		boost::numeric::ublas::span<unsigned_type_rhs> const& rhs)
{
	return lhs.first() == rhs.first() && lhs.last() == rhs.last() && lhs.step() == rhs.step();
}


template<class unsigned_type_lhs, class unsigned_type_rhs>
inline bool operator!=(
		boost::numeric::ublas::span<unsigned_type_lhs> const& lhs,
		boost::numeric::ublas::span<unsigned_type_rhs> const& rhs)
{
	return !(lhs == rhs);
}

#endif // FHG_range_H
