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


#ifndef BOOST_UBLAS_TENSOR_SPAN_HPP
#define BOOST_UBLAS_TENSOR_SPAN_HPP

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

static constexpr inline std::size_t max = std::numeric_limits<std::ptrdiff_t>::max();

template<typename unsigned_type = std::size_t>
class span
{
public:
	using value_type = std::size_t;

	// covers the complete range of one dimension
	// e.g. a(:)
	constexpr explicit span()
		: first_{}
		, step_ {1}
		, last_ {max}
	{
    }

    // covers only one index of one dimension
	// e.g. a(1) or a(0)
    // TODO: case where stop < 0 then stop += length
	explicit span(value_type l)
		: span(l,1,l)
	{
	}

    // covers only one index of one dimension
	// e.g. a(1:3) or a(0:end)
	span(value_type f, value_type l)
		: span(f,1,l)
	{
	}

	// covers a linear range of one dimension
	// e.g. a(1:3:n)
	span(value_type f, value_type s, value_type l)
		: first_(f)
		, step_ (s)
        , last_ (l)
	{
        if(s == 0 && f != l)
            throw std::runtime_error("Error in span::span : cannot have a step_ equal to zero.");
        if(f > l)
				throw std::runtime_error("Error in span::span: last_ is smaller than first");
        last_ = l - ((l-f)%s);
	}

    span(span const& other)
		: first_(other.first_)
		, step_ (other.step_ )
		, last_ (other.last_ )
	{
	}

	span& operator=(span const& other)
	{
		first_ = other.first_;
		last_  = other.last_ ;
		step_  = other.step_ ;
		return *this;
	}

    inline auto first() const {return first_; }
    inline auto last () const {return last_ ; }
    inline auto step () const {return step_ ; }

	~span() = default;

    [[ nodiscard ]] inline value_type size() const {
        if (first_ == last_) {
            return value_type(1);
        }
        return (last_-first_) / step_ + value_type(1);
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

	value_type first_, step_, last_ ;
};

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

#endif // BOOST_UBLAS_TENSOR_SPAN_HPP
