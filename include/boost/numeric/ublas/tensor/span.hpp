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


#ifndef _BOOST_UBLAS_TENSOR_SPAN_
#define _BOOST_UBLAS_TENSOR_SPAN_

#include <typeinfo>
#include <limits>
#include <stdexcept>
#include <ostream>

namespace boost   {
namespace numeric {
namespace ublas   {
namespace tag     {

struct sliced  {};
struct strided {};

}
}
}
}


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



//using offsets = std::vector<std::ptrdiff_t>;

template<class span_tag, class unsigned_type>
class span;


static constexpr inline std::size_t end = std::numeric_limits<std::size_t>::max();

template<>
class span<tag::strided, std::size_t>
{
public:
	using span_tag = tag::strided;
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
	span(value_type n)
		: span(n,1,n)
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

using strided_span = span<tag::strided, std::size_t>;

} // namespace


/////////////

namespace boost::numeric::ublas {

template<>
class span<tag::sliced, std::size_t> :
		private span<tag::strided, std::size_t>
{
	using super_type = span<tag::strided,std::size_t>;
public:
	using span_tag = tag::sliced;
	using value_type = typename super_type::value_type;
	constexpr explicit span()
		: super_type()
	{
	}

	span(value_type f, value_type l)
		: super_type(f, value_type(1), l )
	{
	}

	span(value_type n)
		: super_type(n)
	{
	}

	span(span const& other)
		: super_type(other)
	{
	}

  inline span& operator=(const span &other)
	{
		super_type::operator=(other);
		return *this;
	}

	~span() = default;

  inline value_type operator[] (std::size_t idx) const
	{
		return super_type::operator [](idx);
	}

  inline auto first() const {return super_type::first(); }
  inline auto last () const {return super_type::last (); }
  inline auto step () const {return super_type::step (); }
  inline auto size () const {return super_type::size (); }

  inline span operator()(const span &rhs) const
	{
		auto const& lhs = *this;
		return span( rhs.first_ + lhs.first_, rhs.last_  + lhs.first_ );
	}
};

using sliced_span = span<tag::sliced, std::size_t>;


template<class unsigned_type>
inline auto ran(unsigned_type f, unsigned_type l)
{
	return sliced_span(f,l);
}

template<class unsigned_type>
inline auto ran(unsigned_type f, unsigned_type s, unsigned_type l)
{
	return strided_span(f,s,l);
}

} // namespace


template <class span_tag, class unsigned_type>
std::ostream& operator<< (std::ostream& out,  boost::numeric::ublas::span<span_tag,unsigned_type> const& s)
{
  return out << "[" << s.first() << ":" << s.step() << ":" << s.last() << "]" << std::endl;
}

template<class span_tag_lhs, class span_tag_rhs, class unsigned_type>
inline bool operator==(
		boost::numeric::ublas::span<span_tag_lhs,unsigned_type> const& lhs,
		boost::numeric::ublas::span<span_tag_rhs,unsigned_type> const& rhs)
{
	return lhs.first() == rhs.first() && lhs.last() == rhs.last() && lhs.step() == rhs.step();
}


template<class span_tag_lhs, class span_tag_rhs, class unsigned_type>
inline bool operator!=(
		boost::numeric::ublas::span<span_tag_lhs,unsigned_type> const& lhs,
		boost::numeric::ublas::span<span_tag_rhs,unsigned_type> const& rhs)
{
	return lhs.first() != rhs.first() || lhs.last() != rhs.last() || lhs.step() != rhs.step();
}

#endif // FHG_range_H
