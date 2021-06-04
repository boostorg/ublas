//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_FUNCTIONS_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_FUNCTIONS_HPP


#include <algorithm>
#include <numeric>
//#include <concepts>

#include "../layout.hpp"
#include "../concepts.hpp"

namespace boost::numeric::ublas
{
template<integral T, T...>
class extents_core;


template<class D>
struct extents_base;

template <class D> [[nodiscard]] constexpr inline auto front (extents_base<D> const& e) noexcept -> typename D::const_reference         { return e().base().front(); }
template <class D> [[nodiscard]] constexpr inline auto back  (extents_base<D> const& e) noexcept -> typename D::const_reference         { return e().base().back();  }
template <class D> [[nodiscard]] constexpr inline auto begin (extents_base<D> const& e) noexcept -> typename D::const_iterator          { return e().base().begin(); }
template <class D> [[nodiscard]] constexpr inline auto end   (extents_base<D> const& e) noexcept -> typename D::const_iterator          { return e().base().end(); }
template <class D> [[nodiscard]] constexpr inline auto cbegin(extents_base<D> const& e) noexcept -> typename D::const_iterator          { return e().base().cbegin(); }
template <class D> [[nodiscard]] constexpr inline auto cend  (extents_base<D> const& e) noexcept -> typename D::const_iterator          { return e().base().cend(); }
template <class D> [[nodiscard]] constexpr inline auto rbegin(extents_base<D> const& e) noexcept -> typename D::const_reverse_iterator  { return e().base().rbegin(); }
template <class D> [[nodiscard]] constexpr inline auto rend  (extents_base<D> const& e) noexcept -> typename D::const_reverse_iterator  { return e().base().rend(); }
template <class D> [[nodiscard]] constexpr inline auto empty (extents_base<D> const& e) noexcept -> bool                                { return e().base().empty(); }
template <class D> [[nodiscard]] constexpr inline auto size  (extents_base<D> const& e) noexcept -> typename D::size_type               { return e().base().size(); }

} //namespace boost::numeric::ublas


namespace boost::numeric::ublas
{

/** @brief Returns true if extents equals ([m,n,...,l]) with m>0,n>0,...,l>0  */
template <class D>
[[nodiscard]] inline constexpr bool is_valid(extents_base<D> const& e)
{
  return std::all_of(begin(e),end(e), [](auto a){ return a>0UL; } );
}

/** @brief Returns true if extents equals (m,[n,...,l]) with m=1,n=1,...,l=1  */
template <class D>
[[nodiscard]] inline constexpr bool is_scalar(extents_base<D> const& e)
{
  return (size(e)>0) && std::all_of  (cbegin(e),cend(e),[](auto a){return a==1ul;});
}

/** @brief Returns true if extents equals (m,[n,1,...,1]) with m>=1||n>=1 && m==1||n==1*/
template <class D>
[[nodiscard]] inline constexpr bool is_vector(extents_base<D> const& e)
{
  if (empty(e)     ) {return false;}
  if (size (e) == 1) {return front(e)>=1ul;}

  return
    std::any_of(cbegin(e)    ,cbegin(e)+2ul, [](auto a){return a>=1ul;}) &&
    std::any_of(cbegin(e)    ,cbegin(e)+2ul, [](auto a){return a==1ul;}) &&
    std::all_of(cbegin(e)+2ul,cend  (e)    , [](auto a){return a==1ul;});

//    std::any_of(cbegin(e)  ,cbegin(e)+2, [](auto a){return a>=1UL;}) && // a>1UL
//         std::any_of(cbegin(e)  ,cbegin(e)+2, [](auto a){return a==1UL;}) &&
//         std::all_of(cbegin(e)+2,cend(e)    , [](auto a){return a==1UL;});
}

/** @brief Returns true if extents equals (m,[1,1,...,1]) with m>=1 */
template <class D>
[[nodiscard]] inline constexpr bool is_row_vector(extents_base<D> const& e)
{
  if (empty(e) || size(e) == 1  ) {return false;}

  if(cbegin(e)[0] ==  1ul &&
      cbegin(e)[1] >  1ul &&
      std::all_of(cbegin(e)+2ul,cend  (e) , [](auto a){return a==1ul;})){
    return true;
  }

  return false;
}


/** @brief Returns true if extents equals (m,[1,1,...,1]) with m>=1 */
template <class D>
[[nodiscard]] inline constexpr bool is_col_vector(extents_base<D> const& e)
{
  if (empty(e) || size(e) == 1  ) {return false;}

  if(cbegin(e)[0] >  1ul &&
      cbegin(e)[1] == 1ul &&
      std::all_of(cbegin(e)+2ul,cend  (e) , [](auto a){return a==1ul;})){
    return true;
  }

  return false;
}

/** @brief Returns true if (m,[n,1,...,1]) with m>=1 or n>=1 */
template <class D>
[[nodiscard]] inline constexpr bool is_matrix(extents_base<D> const& e)
{
  if (empty(e)     ) {return false;}
  if (size (e) == 1) {return front(e)>=1ul;}

  return std::any_of  (cbegin(e)  ,cbegin(e)+2, [](auto a){return a>=1ul;}) && // all_of > 1UL
         std::all_of  (cbegin(e)+2,cend(e)    , [](auto a){return a==1ul;});
}

/** @brief Returns true if shape is has a tensor shape
 *
 * @returns true if is_valid & not (is_scalar&is_vector&is_matrix)
 */
template <class D>
[[nodiscard]] inline constexpr bool is_tensor(extents_base<D> const& e)
{
  return size(e) > 2 &&
         std::all_of  (cbegin(e)  ,cbegin(e)+2, [](auto a){return a>=1ul;}) && // all_of > 1UL
         std::any_of  (cbegin(e)+2,cend(e)    , [](auto a){return a> 1ul;});
}





/** @brief Computes the number of elements */
template <class D>
[[nodiscard]] inline constexpr auto product( extents_base<D> const& e )
{
  if( empty(e) ){
    return std::size_t{0UL};
  }

  return std::accumulate(begin(e), end(e), std::size_t{1UL}, std::multiplies<>{});
}


//template<class InputIt, class OutputIt> // std::inserter(out,out.begin())
//inline constexpr
//  void squeeze(
//    extents_base<D> const& in,
//    extents_base<D> const& out)
//{
//  if(e().size() < 2){ return; }

//  if(is_vector(e) || is_scalar(e)) {
//    std::copy   (in, in+2,   out );
//  }
//  else{
//    std::copy_if(in, in_end, out, [](auto a){return a!=1u;});
//  }
//}

//template<class D>
//[[nodiscard]] inline bool operator==(
//  extents_base<D> const& lhs,
//  extents_base<D> const& rhs )
//{
//  return size(lhs) == size(rhs) && std::equal( begin(lhs), end  (lhs), begin(rhs) );
//}

//template<class D>
//[[nodiscard]] inline bool operator!=(
//  extents_base<D> const& lhs,
//  extents_base<D> const& rhs )
//{
//  return !( lhs == rhs) ;
//}

template<integral T, class L>
[[nodiscard]] inline auto to_strides(extents_core<T> const& e, L /*unused*/)
{
  auto s = typename extents_core<T>::base_type(e.size(),1ul);

  if(empty(e) || is_scalar(e)){
    return s;
  }

  // || is_vector(e)

  if constexpr(std::is_same_v<L,layout::first_order>){
    std::transform(begin (e), end (e) - 1,  s.begin (),  s.begin ()+1, std::multiplies<>{});
  } else {
    std::transform(rbegin(e), rend(e) - 1,  s.rbegin(),  s.rbegin()+1, std::multiplies<>{});
  }
  return s;
}

template<integral T, T n, class L>
[[nodiscard]] inline auto to_strides(extents_core<T,n> const& e, L /*unused*/)
{
  auto s = typename extents_core<T,n>::base_type{};
  std::fill(s.begin(),s.end(),1ul);

  if(empty(e) || is_scalar(e)){
    return s;
  }


  // || is_vector(e)

  if constexpr(std::is_same_v<L,layout::first_order>){
    std::transform(begin (e), end (e) - 1,  s.begin (),  s.begin ()+1, std::multiplies<>{});
  } else {
    std::transform(rbegin(e), rend(e) - 1,  s.rbegin(),  s.rbegin()+1, std::multiplies<>{});
  }
  return s;
}



} // namespace boost::numeric::ublas


template<boost::numeric::ublas::integral T, T n, T m>
[[nodiscard]] inline constexpr bool operator==(
  boost::numeric::ublas::extents_core<T,n> const& lhs,
  boost::numeric::ublas::extents_core<T,m> const& rhs )
{
  if constexpr(m != n)
    return false;
  return std::equal( begin(lhs), end  (lhs), begin(rhs) );
}

template<boost::numeric::ublas::integral T, T n, T m>
[[nodiscard]] inline constexpr bool operator!=(
  boost::numeric::ublas::extents_core<T,n> const& lhs,
  boost::numeric::ublas::extents_core<T,m> const& rhs )
{
  if constexpr(m == n)
    return false;
  return !(lhs == rhs) ;
}

template<class D, class F>
[[nodiscard]] inline constexpr bool operator==(
  boost::numeric::ublas::extents_base<D> const& lhs,
  boost::numeric::ublas::extents_base<F> const& rhs )
{
  return size(lhs) == size(rhs) && std::equal( begin(lhs), end  (lhs), begin(rhs) );
}

template<class D, class F>
[[nodiscard]] inline constexpr bool operator!=(
  boost::numeric::ublas::extents_base<D> const& lhs,
  boost::numeric::ublas::extents_base<F> const& rhs )
{
  return !( lhs == rhs) ;
}


namespace std
{

template<boost::numeric::ublas::integral T, T e, T... es>
struct tuple_size< boost::numeric::ublas::extents_core<T,e,es...> >
  : integral_constant<std::size_t,
                           std::tuple_size_v<typename boost::numeric::ublas::extents_core<T,e,es...>::base_type>>
{};

template<size_t i, boost::numeric::ublas::integral T, T e1, T ... es>
[[nodiscard]] constexpr inline
  auto get(boost::numeric::ublas::extents_core<T,e1,es...> const& e) noexcept
{
  return std::get<i>(e.base());
}

} // namespace std


#endif // _BOOST_NUMERIC_UBLAS_TENSOR_EXTENTS_FUNCTIONS_HPP_


