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
/// \file strides.hpp Definition for the basic_strides template class


#ifndef BOOST_UBLAS_TENSOR_DYNAMIC_STRIDES_HPP
#define BOOST_UBLAS_TENSOR_DYNAMIC_STRIDES_HPP

#include <boost/numeric/ublas/functional.hpp>

#include "detail/extents_functions.hpp"
#include "detail/strides_functions.hpp"
#include "dynamic_extents.hpp"
#include "layout.hpp"
#include "strides_base.hpp"
#include "type_traits.hpp"


namespace boost::numeric::ublas
{

template<class T>
class basic_extents;


/** @brief Template class for storing tensor strides for iteration with runtime variable size.
 *
 * Proxy template class of std::vector<int_type>.
 *
 */
template<class T, class L>
class basic_strides
  : public strides_base<basic_strides<T, L>>
{
public:

    using base_type = std::vector<T>;
    using layout_type = L;
    using value_type = typename base_type::value_type;
    using reference = typename base_type::reference;
    using const_reference = typename base_type::const_reference;
    using size_type = typename base_type::size_type;
    using const_pointer = typename base_type::const_pointer;
    using const_iterator = typename base_type::const_iterator;
    using const_reverse_iterator = typename base_type::const_reverse_iterator;

    static_assert( std::numeric_limits<value_type>::is_integer,
                  "Static error in boost::numeric::ublas::basic_strides: type must be of type integer.");
    static_assert(!std::numeric_limits<value_type>::is_signed,
                  "Static error in boost::numeric::ublas::basic_strides: type must be of type unsigned integer.");
    static_assert(std::is_same<layout_type,layout::first_order>::value || std::is_same<layout_type,layout::last_order>::value,
                  "Static error in boost::numeric::ublas::basic_strides: layout type must either first or last order");


    /** @brief Default constructs basic_strides
     *
     * @code auto ex = basic_strides<unsigned>{};
     */
    constexpr explicit basic_strides() = default;

    /** @brief Constructs basic_strides from basic_extents for the first- and last-order storage formats
     *
     * @code auto strides = basic_strides<unsigned>( basic_extents<std::size_t>{2,3,4} );
     *
     */
    constexpr explicit basic_strides(basic_extents<value_type> const& extents)
            : _base(extents.size(),1U)
    {
      if(extents.empty() || extents.size() != this->size()){
        return;
      }

      std::fill(_base.begin(), _base.end(), 1U);

      if( is_vector(extents) || is_scalar(extents) ){
        return;
      }

      //using layout_type = typename derived_type_strides::layout_type;
      if constexpr (std::is_same<layout_type,layout::first_order>::value ) {
        std::transform(extents().begin(), extents().end() - 1, _base.begin(), _base.begin() + 1, std::multiplies<>{});
      } else {
        std::transform(extents().rbegin(), extents().rend() - 1, _base.rbegin(), _base.rbegin() + 1, std::multiplies<>{});
      }
    }

    constexpr basic_strides(basic_strides const& l)
        : _base(l._base)
    {}

    constexpr basic_strides(basic_strides && l ) noexcept
        : _base(std::move(l._base))
    {}

    constexpr explicit basic_strides(base_type const& l )
        : _base(l)
    {}

    constexpr explicit basic_strides(base_type && l ) noexcept
        : _base(std::move(l))
    {}

    ~basic_strides() = default;

    constexpr basic_strides& operator=(basic_strides&& other) 
        noexcept(std::is_nothrow_swappable_v<base_type>)
    {
        swap (*this, other);
        return *this;
    }

    constexpr basic_strides& operator=(basic_strides const& other) 
        noexcept(std::is_nothrow_swappable_v<base_type>)
    {
        basic_strides temp(other);
        swap (*this, temp);
        return *this;
    }

    friend void swap(basic_strides& lhs, basic_strides& rhs) 
        noexcept(std::is_nothrow_swappable_v<base_type>) 
    {
        std::swap(lhs._base   , rhs._base);
    }

    [[nodiscard]] inline
    constexpr const_reference operator[] (size_type p) const{
        return _base[p];
    }

    [[nodiscard]] inline
    constexpr const_pointer data() const{
        return _base.data();
    }

    [[nodiscard]] inline
    constexpr const_reference at (size_type p) const{
        return _base.at(p);
    }

    [[nodiscard]] inline
    constexpr const_reference back () const{
        return _base.back();
    }

    [[nodiscard]] inline
    constexpr reference back (){
        return _base.back();
    }

    [[nodiscard]] inline
    constexpr bool empty() const noexcept{
        return _base.empty();
    }

    [[nodiscard]] inline
    constexpr size_type size() const noexcept{
        return _base.size();
    }

    [[nodiscard]] inline
    constexpr const_iterator begin() const noexcept{
        return _base.begin();
    }

    [[nodiscard]] inline
    constexpr const_iterator end() const noexcept{
        return _base.end();
    }

    inline
    constexpr void clear() noexcept{
        this->_base.clear();
    }

    [[nodiscard]] inline
    constexpr base_type const& base() const noexcept{
        return this->_base;
    }

    [[nodiscard]] inline
    constexpr const_reverse_iterator
    rbegin() const noexcept
    {
        return _base.rbegin();
    }

    [[nodiscard]] inline
    constexpr const_reverse_iterator
    rend() const noexcept
    {
        return _base.rend();
    }
    
private:
    base_type _base{};
};


template <class L, class T> struct is_strides<basic_strides<T,L>> : std::true_type {};
template <class T, class L> struct is_dynamic< basic_strides<T,L> > : std::true_type {};
template <class T, class L> struct is_dynamic_rank< basic_strides<T, L> > : std::true_type {};

template <class T>
struct strides<basic_extents<T>>
{
  template<typename Layout>
  using type = basic_strides<T, Layout>;
};

} // namespace boost::numeric::ublas

#endif
