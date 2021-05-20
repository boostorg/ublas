//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//
/// \file strides.hpp Definition for the basic_strides template class



#ifndef BOOST_UBLAS_TENSOR_FIXED_RANK_STRIDES_HPP
#define BOOST_UBLAS_TENSOR_FIXED_RANK_STRIDES_HPP

#if 0

#include "detail/strides_functions.hpp"
#include "extents/extents_static_size.hpp"
#include "layout.hpp"
#include "strides_base.hpp"


namespace boost::numeric::ublas {

/** @brief Template class for storing tensor strides for iteration with runtime variable size.
 *
 * Proxy template class of std::array<int_type,N>.
 *
 */
template<class L,std::size_t N>
class strides<extents<N>,L> : public strides_base<strides<extents<N>,L>>
{
public:

  using extents_type           = extents<N>;
  using layout_type            = L;
  using base_type              = typename extents_type::base_type;
  using value_type             = typename base_type::value_type;
  using reference              = typename base_type::reference;
  using const_reference        = typename base_type::const_reference;
  using size_type              = typename base_type::size_type;
  using const_pointer          = typename base_type::const_pointer;
  using const_iterator         = typename base_type::const_iterator;
  using const_reverse_iterator = typename base_type::const_reverse_iterator;


  static_assert(std::is_same<layout_type,layout::first_order>::value ||
                std::is_same<layout_type,layout::last_order >::value);
    /** @brief Default constructs strides with static size
     *
     * @code auto s = strides<extents<3>>{};
     */
    constexpr strides() noexcept = default;

    /** @brief Constructs strides from extents with static size for the first- and last-order storage formats
     *
     * @code auto s = strides<extents<3>>({2,3,4},layout::first_order{});
     *
     */
    constexpr explicit strides(extents_type const& e)
      : _base(compute_strides(e))
    {
    }

    constexpr strides(strides const& l)
        : _base(l._base)
    {}

    constexpr strides(strides && l ) noexcept
        : _base(std::move(l._base))
    {}

    ~strides() = default;


    strides& operator=(strides other)
        noexcept(std::is_nothrow_swappable_v<base_type>)
    {
        swap (*this, other);
        return *this;
    }

    friend void swap(strides& lhs, strides& rhs)
        noexcept(std::is_nothrow_swappable_v<base_type>)
    {
        std::swap(lhs._base,rhs._base);
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
  base_type _base;
  static constexpr std::size_t const _size = N;


  [[nodiscard]] inline auto compute_strides( extents_type const& e)
  {
    using base_type    = typename extents_type::base_type;
    namespace ub       = boost::numeric::ublas;
    auto init = []<std::size_t ... is>(std::index_sequence<is...>){ return base_type{is...}; };

    auto s = init(std::make_index_sequence<N>{});

    if (std::tuple_size_v<extents_type> == 0UL)
      return s;
    if (ub::is_vector(e) || ub::is_scalar(e))
      return s;

    if constexpr(std::is_same_v<layout_type,layout::first_order>){
      std::transform(ub::begin(e),  ub::end(e) - 1,  s.begin(),  s.begin()  + 1, std::multiplies<>{});
    }
    else {
      std::transform(ub::rbegin(e), ub::rbegin(e) - 1, s.rbegin(),  s.rbegin()  + 1, std::multiplies<>{});
    }

    return s;
  }
};


template <class L, std::size_t E> struct is_strides     <strides<extents<E>,L>> : std::true_type {};
template <class L, std::size_t E> struct is_dynamic     <strides<extents<E>,L>> : std::true_type {};
template <class L, std::size_t E> struct is_static_rank <strides<extents<E>,L>> : std::true_type {};

} // namespace boost::numeric::ublas

#endif

#endif
