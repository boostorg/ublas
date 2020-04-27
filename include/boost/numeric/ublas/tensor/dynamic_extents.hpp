//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//


#ifndef _BOOST_NUMERIC_UBLAS_TENSOR_DYNAMIC_EXTENTS_HPP_
#define _BOOST_NUMERIC_UBLAS_TENSOR_DYNAMIC_EXTENTS_HPP_

#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>
#include <boost/numeric/ublas/tensor/detail/type_traits.hpp>
#include <boost/numeric/ublas/tensor/detail/extents_functions.hpp>

namespace boost {
namespace numeric {
namespace ublas {

/** @brief Template class for storing tensor extents with runtime variable size.
 *
 * Proxy template class of std::vector<int_type>.
 *
 */
template<class int_type>
class basic_extents
{
    static_assert( std::numeric_limits<typename std::vector<int_type>::value_type>::is_integer, "Static error in basic_layout: type must be of type integer.");
    static_assert(!std::numeric_limits<typename std::vector<int_type>::value_type>::is_signed,  "Static error in basic_layout: type must be of type unsigned integer.");

public:
    using base_type = std::vector<int_type>;
    using value_type = typename base_type::value_type;
    using const_reference = typename base_type::const_reference;
    using reference = typename base_type::reference;
    using size_type = typename base_type::size_type;
    using const_pointer = typename base_type::const_pointer;
    using const_iterator = typename base_type::const_iterator;


    /** @brief Default constructs basic_extents
     *
     * @code auto ex = basic_extents<unsigned>{};
     */
    basic_extents()
      : _base{}
    {
    }

    /** @brief Copy constructs basic_extents from a one-dimensional container
     *
     * @code auto ex = basic_extents<unsigned>(  std::vector<unsigned>(3u,3u) );
     *
     * @note checks if size > 1 and all elements > 0
     *
     * @param b one-dimensional std::vector<int_type> container
     */
    explicit basic_extents(base_type const& b)
      : _base(b)
    {
        if ( !valid(*this) ){
            throw std::length_error("Error in basic_extents::basic_extents() : shape tuple is not a valid permutation: has zero elements.");
        }
    }

    /** @brief Move constructs basic_extents from a one-dimensional container
     *
     * @code auto ex = basic_extents<unsigned>(  std::vector<unsigned>(3u,3u) );
     *
     * @note checks if size > 1 and all elements > 0
     *
     * @param b one-dimensional container of type std::vector<int_type>
     */
    explicit basic_extents(base_type && b)
      : _base(std::move(b))
    {
        if (!valid(*this)){
            throw std::length_error("Error in basic_extents::basic_extents() : shape tuple is not a valid permutation: has zero elements.");
        }
    }

    /** @brief Constructs basic_extents from an initializer list
     *
     * @code auto ex = basic_extents<unsigned>{3,2,4};
     *
     * @note checks if size > 1 and all elements > 0
     *
     * @param l one-dimensional list of type std::initializer<int_type>
     */
    basic_extents(std::initializer_list<value_type> l)
      : basic_extents( base_type(std::move(l)) )
    {
    }

    /** @brief Constructs basic_extents from a range specified by two iterators
     *
     * @code auto ex = basic_extents<unsigned>(a.begin(), a.end());
     *
     * @note checks if size > 1 and all elements > 0
     *
     * @param first iterator pointing to the first element
     * @param last iterator pointing to the next position after the last element
     */
    basic_extents(const_iterator first, const_iterator last)
      : basic_extents ( base_type( first,last ) )
    {
    }

    /** @brief Copy constructs basic_extents */
    basic_extents(basic_extents const& l )
      : _base(l._base)
    {
    }

    /** @brief Move constructs basic_extents */
    basic_extents(basic_extents && l ) noexcept
      : _base(std::move(l._base))
    {
    }


    template<typename OtherExtentsType,
        std::enable_if_t< 
            is_extents<OtherExtentsType>::value
            ,int > = 0
    >
    basic_extents(OtherExtentsType const& e)
        : _base(e.size())
    {
        std::copy(e.begin(),e.end(), _base.begin());
    }

    ~basic_extents() = default;

    basic_extents& operator=(basic_extents other) noexcept
    {
        swap (*this, other);
        return *this;
    }

    friend void swap(basic_extents& lhs, basic_extents& rhs) {
        std::swap(lhs._base   , rhs._base   );
    }

    [[nodiscard]] inline
    constexpr const_pointer data() const
    {
        return this->_base.data();
    }

    [[nodiscard]] inline
    constexpr const_reference operator[] (size_type p) const noexcept
    {
        return this->_base[p];
    }

    [[nodiscard]] inline
    constexpr const_reference at (size_type p) const
    {
        return this->_base.at(p);
    }

    [[nodiscard]] inline
    constexpr reference operator[] (size_type p)
    {
        return this->_base[p];
    }

    [[nodiscard]] inline
    constexpr reference at (size_type p)
    {
        return this->_base.at(p);
    }

    [[nodiscard]] inline
    constexpr const_reference back () const
    {
        return this->_base.back();
    }


    [[nodiscard]] inline
    constexpr bool empty() const
    {
        return this->_base.empty();
    }

    [[nodiscard]] inline
    constexpr size_type size() const
    {
        return this->_base.size();
    }

    inline
    constexpr void clear()
    {
        this->_base.clear();
    }

    template <class Extents, std::enable_if_t<is_extents_v<Extents>, int> = 0 >
    [[nodiscard]] inline
    constexpr bool operator==(Extents const& rhs) const noexcept{
        static_assert(is_extents_v<Extents>,
            "boost::numeric::ublas::operator==() : invalid type, type should be an extents");
        if( this->size() != rhs.size() ){
            return false;
        }else{
            return std::equal(this->begin(), this->end(), rhs.begin());
        }
    }

    template <class Extents, std::enable_if_t<is_extents_v<Extents>, int> = 0 >
    [[nodiscard]] inline
    constexpr bool operator!=(Extents const& rhs) const noexcept{
       return !( *this == rhs );
    }

    [[nodiscard]] inline
    constexpr const_iterator
    begin() const
    {
        return _base.begin();
    }

    [[nodiscard]] inline
    constexpr const_iterator
    end() const
    {
        return _base.end();
    }

    [[nodiscard]] inline
    constexpr base_type const& base() const { return _base; }

private:

    base_type _base;

};

} // namespace ublas
} // namespace numeric
} // namespace boost

namespace boost::numeric::ublas{
    
    template <class T> 
    struct is_extents< basic_extents<T> > : std::true_type {};

    template <class T>
    struct is_dynamic< basic_extents<T> > : std::true_type {};

    template <class T>
    struct is_dynamic_rank< basic_extents<T> > : std::true_type {};


    namespace detail{
        
        template <> struct dynamic_extents_impl<> {
            using type = basic_extents<std::size_t>;
        };

    } // namespace detail

} // namespace boost::numeric::ublas


#endif
