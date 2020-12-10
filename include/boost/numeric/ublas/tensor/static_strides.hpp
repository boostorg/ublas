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

#ifndef BOOST_UBLAS_TENSOR_STATIC_STRIDES_HPP
#define BOOST_UBLAS_TENSOR_STATIC_STRIDES_HPP

#include <boost/numeric/ublas/tensor/static_extents.hpp>
#include <boost/numeric/ublas/tensor/layout.hpp>
#include <boost/numeric/ublas/tensor/extents_functions.hpp>

namespace boost::numeric::ublas{

  template <class E, class L> class basic_static_strides;

} // boost::numeric::ublas

namespace boost::numeric::ublas::detail{

  namespace impl{
 
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

    // generates static_stride_list containing ones with specific size
    template<typename T, std::size_t N> 
    struct make_sequence_of_ones;

    template<typename T, std::size_t N> 
    using make_sequence_of_ones_t = typename make_sequence_of_ones<T, N>::type;

    template<typename T, std::size_t N>
    struct make_sequence_of_ones {
      using type = concat_t<make_sequence_of_ones_t<T, N/2>, make_sequence_of_ones_t<T, N - N/2>>;
    };

    template<typename T> 
    struct make_sequence_of_ones<T, 0ul> {
      using type = basic_static_extents<T>;
    };
    template<typename T> 
    struct make_sequence_of_ones<T, 1ul>{ 
      using type = basic_static_extents<T, T(1)>;
    };

    template<typename T>
    struct extents_to_array;
    
    template<typename T>
    inline static constexpr auto extents_to_array_v = extents_to_array<T>::value;
    
    template<typename T, T... Es>
    struct extents_to_array< basic_static_extents<T,Es...> >
    {
      static constexpr std::array<T,sizeof...(Es)> const value = {Es...};
    };

  } // impl


  template<typename T, std::size_t N> 
  using make_sequence_of_ones_t = impl::make_sequence_of_ones_t<T,N>;

  template<typename E, std::size_t I = 0ul, typename T, T... Es>
  constexpr auto make_static_strides_first_order( [[maybe_unused]] E const& e, [[maybe_unused]] basic_static_extents<T,Es...> const& res ){
    if constexpr( I >=  E::_size - 1ul ){
      return impl::extents_to_array_v< basic_static_extents<T,Es...>  >;
    }else{
      using res_type = basic_static_extents<T,Es...>;

      constexpr auto prod = E::template get<I>().value * res_type::template get<I>().value;
      using nextents = basic_static_extents<T, Es..., prod>;
      return make_static_strides_first_order<E,I + 1>(e, nextents{});
    }
  }

  template<typename E, std::size_t I = 0ul, typename T, T... Es>
  constexpr auto make_static_strides_last_order( [[maybe_unused]] E const& e, [[maybe_unused]] basic_static_extents<T,Es...> const& res ){
    if constexpr( I >=  E::_size - 1ul ){
      return impl::extents_to_array_v< basic_static_extents<T,Es...>  >;
    }else{
      using res_type = basic_static_extents<T,Es...>;

      constexpr auto J = E::_size - I - 1ul;
      constexpr auto K = res_type::_size - I - 1ul;
      constexpr auto prod = E::template get<J>().value * res_type::template get<K>().value;
      using nextents = basic_static_extents<T, prod, Es...>;
      return make_static_strides_last_order<E,I + 1>(e, nextents{});
    }
  }

  template<typename L, typename E>
  constexpr auto make_static_strides( [[maybe_unused]] E const& e ){
    using value_type = typename E::value_type;
    if constexpr( E::_size == 0 ){
      return impl::extents_to_array_v<E>;
    }else if constexpr( is_scalar(E{}) || is_vector(E{}) ){
      using extents_with_ones = make_sequence_of_ones_t<value_type, E::_size>;
      return impl::extents_to_array_v<extents_with_ones>;
    }else{
      if constexpr( std::is_same_v<L, layout::first_order> ){
        return make_static_strides_first_order(e, basic_static_extents<value_type,1>{});
      }else{
        return make_static_strides_last_order(e, basic_static_extents<value_type,1>{});
      }
    }
  }

  // It is use for first order to
  // get std::array containing strides
  template<typename Layout, typename ExtentsType>
  inline static constexpr auto strides_helper_v = make_static_strides<Layout>(ExtentsType{});

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
  constexpr size_type size() const noexcept { return static_cast<size_type>(_size); }

  [[nodiscard]] inline
  constexpr const_reference back () const{
      return m_data.back();
  }

  // default constructor
  constexpr basic_static_strides() noexcept{
    static_assert( 
      _size == 0 || 
      ( is_valid(extents_type{}) &&
        ( is_vector(extents_type{}) ||
          is_scalar(extents_type{}) ||
          _size >= 2 
        )
      )
      , 
      "Error in boost::numeric::ublas::basic_static_strides() : "
      "Size cannot be 0 or Shape should be valid and shape can be vector or shape can be scalar or size should be greater than"
      " or equal to 2"
    ); 	
    
    
  }

  constexpr basic_static_strides(extents_type const& e) noexcept{ (void)e; };

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

} // namespace boost::numeric::ublas

#endif
