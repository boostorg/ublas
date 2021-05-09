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

#ifndef BOOST_UBLAS_TENSOR_STATIC_STRIDES_HPP
#define BOOST_UBLAS_TENSOR_STATIC_STRIDES_HPP

#include "detail/extents_functions.hpp"
#include "detail/strides_functions.hpp"
#include "layout.hpp"
#include "static_extents.hpp"
#include "strides_base.hpp"
#include "traits/type_traits_strides.hpp"

#include <cstdlib>

namespace boost::numeric::ublas{

  template <class E, class L> class basic_static_strides;

} // boost::numeric::ublas

namespace boost::numeric::ublas::detail{

  namespace impl {
 
    // concat two static_stride_list togather
    // @code using type = typename concat< static_stride_list<int, 1,2,3>, static_stride_list<int, 4,5,6> >::type @endcode
    template<typename L1, typename L2>
    struct concat;

    template<typename T, T... N1, T... N2>
    struct concat< basic_static_extents<T, N1...>, basic_static_extents<T, N2...> > {
      using type = basic_static_extents<T, N1..., N2...>;
    };

    template<typename L1, typename L2>
    using concat_t = typename concat<L1,L2>::type;

  } // namespace impl

  template<typename Layout, typename ExtentsType>
  constexpr auto make_static_strides() noexcept{
    using ext_type = typename ExtentsType::value_type;
    
    constexpr auto ext_arr = ExtentsType{}.base();
    std::array<ext_type,ext_arr.size()> strides{};

    if constexpr(ext_arr.empty()){
      return strides;
    }else{
      std::fill(strides.begin(), strides.end(), 1);
      if constexpr( !( is_vector(ExtentsType{}) || is_scalar(ExtentsType{}) ) )
        detail::compute_strides_helper(ext_arr,strides,Layout{});

      return strides;
    }
    
  }
  // It is use for first order to
  // get std::array containing strides
  template<typename Layout, typename ExtentsType>
  inline static constexpr auto strides_helper_v = make_static_strides<Layout,ExtentsType>();


} // namespace boost::numeric::ublas::detail

namespace boost::numeric::ublas
{
/** @brief Partial Specialization for layout::first_order or column_major
 *
 * @code basic_static_strides<basic_static_extents<4,1,2,3,4>, layout::first_order> s @endcode
 *
 * @tparam R rank of basic_static_extents
 * @tparam Extents paramerter pack of extents
 *
 */
template <class Layout, class T, T... Extents>
class basic_static_strides<basic_static_extents<T,Extents...>, Layout>
  : public strides_base<basic_static_strides<basic_static_extents<T,Extents...>, Layout>>
{

public:

  static constexpr std::size_t const _size = sizeof...(Extents);

  using layout_type     = Layout;
  using extents_type    = basic_static_extents<T,Extents...>;
  using base_type       = std::array<T, _size>;
  using value_type      = typename base_type::value_type;
  using reference       = typename base_type::reference;
  using const_reference = typename base_type::const_reference;
  using size_type       = typename base_type::size_type;
  using const_pointer   = typename base_type::const_pointer;
  using const_iterator  = typename base_type::const_iterator;
  using const_reverse_iterator = typename base_type::const_reverse_iterator;

  /**
   * @param k pos of extent
   * @returns the element at given pos
   */
  [[nodiscard]] inline 
  constexpr const_reference at(size_type k) const 
  {
    return m_data.at(k);
  }

  [[nodiscard]] inline
    constexpr const_reference operator[](size_type k) const { return m_data[k]; }

  //@returns the rank of basic_static_extents
  [[nodiscard]] inline
    constexpr size_type size() const noexcept { return m_data.size(); }

  [[nodiscard]] inline
  constexpr const_reference back () const{
      return m_data.back();
  }

  // default constructor
  constexpr basic_static_strides() noexcept{
    static_assert(
      m_data.empty() ||
      ( is_valid(extents_type{}) &&
        ( is_vector(extents_type{}) ||
          is_scalar(extents_type{}) ||
          m_data.size() >= 2
        )
      )
      , 
      "Error in boost::numeric::ublas::basic_static_strides() : "
      "Size cannot be 0 or Shape should be valid and shape can be vector or shape can be scalar or size should be greater than"
      " or equal to 2"
    ); 	
    
    
  }

  constexpr explicit basic_static_strides(extents_type const& e) noexcept{ (void)e; };

  // default copy constructor
  constexpr basic_static_strides(basic_static_strides const &other) noexcept = default;
  constexpr basic_static_strides(basic_static_strides &&other) noexcept = default;
  
  // default assign constructor
  constexpr basic_static_strides &
  operator=(basic_static_strides const &other) noexcept = default;
  
  constexpr basic_static_strides &
  operator=(basic_static_strides &&other) noexcept = default;

  ~basic_static_strides() = default;

   /** @brief Returns ref to the std::array containing extents */
  [[nodiscard]] inline
  constexpr auto const& base() const noexcept{
    return m_data;
  }

  /** @brief Returns pointer to the std::array containing extents */
  [[nodiscard]] inline
  constexpr const_pointer data() const noexcept{
    return m_data.data();
  }

  [[nodiscard]] inline
  constexpr const_iterator begin() const noexcept{
    return m_data.begin();
  }

  [[nodiscard]] inline
  constexpr const_iterator end() const noexcept{
    return m_data.end();
  }

  [[nodiscard]] inline
  constexpr bool empty() const noexcept{
    return m_data.empty();
  }

  [[nodiscard]] inline
  constexpr const_reverse_iterator
  rbegin() const noexcept
  {
      return m_data.rbegin();
  }

  [[nodiscard]] inline
  constexpr const_reverse_iterator
  rend() const noexcept
  {
      return m_data.rend();
  }

private:
  static constexpr base_type const m_data{ detail::strides_helper_v<layout_type,extents_type> };
};

template <class L, class T, T... E> struct is_strides     < basic_static_strides< basic_static_extents<T, E...>, L > > : std::true_type {};
template <class L, class T, T... E> struct is_static      < basic_static_strides< basic_static_extents<T, E...>, L > > : std::true_type {};
template <class L, class T, T... E> struct is_static_rank < basic_static_strides< basic_static_extents<T, E...>, L > > : std::true_type {};

template <class T, T... Extents>
struct strides<basic_static_extents<T, Extents...>>
{
  template<typename Layout>
  using type = basic_static_strides<basic_static_extents<T, Extents...>, Layout>;
};

} // namespace boost::numeric::ublas

#endif
