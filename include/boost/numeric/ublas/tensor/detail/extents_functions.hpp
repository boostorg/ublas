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


#ifndef BOOST_NUMERIC_UBLAS_TENSOR_DETAIL_EXTENTS_FUNCTIONS_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_DETAIL_EXTENTS_FUNCTIONS_HPP


#include <algorithm>
#include <functional>
#include <numeric>


namespace boost::numeric::ublas
{

template<class derived_type>
struct extents_base;


/** @brief Returns true if this has a matrix shape
 *
 * @returns true if (1,1,[1,...,1])
 *
 * @param first input iterator pointing to the start of a shape object
 * @param last  input iterator pointing to the end   of a shape object
 */
template <class D>
[[nodiscard]] inline constexpr
  bool is_scalar(extents_base<D> const& e)
{
  return std::distance(e().begin(),e().end())>0 &&
         std::all_of  (e().begin(),e().end(),[](auto a){return a==1UL;});
}


/** @brief Returns true if this has a matrix shape
 *
 * @returns true if (m,n,[1,...,1]) with m > 1 or n > 1 but not m > 1 and n > 1
 *
 * @param first input iterator pointing to the start of a shape object
 * @param last  input iterator pointing to the end   of a shape object
 */
template <class D>
[[nodiscard]] inline constexpr
  bool is_vector(extents_base<D> const& e)
{
  if (std::distance(e().begin(),e().end()) <  1) {return false;}
  if (std::distance(e().begin(),e().end()) == 1) {return *e().begin() > 1UL;}

  return std::any_of(e().begin()  ,e().begin()+2, [](auto a){return a >1UL;}) &&
         std::any_of(e().begin()  ,e().begin()+2, [](auto a){return a==1UL;}) &&
         std::all_of(e().begin()+2,e().end()    , [](auto a){return a==1UL;});
}


/** @brief Returns true if this has a matrix shape
 *
 * @returns true if (m,n,[1,...,1]) with m > 1 and n > 1
 */
template <class D>
[[nodiscard]] inline constexpr
  bool is_matrix(extents_base<D> const& e)
{
  return std::distance(e().begin()  ,e().end()) > 1 &&
         std::all_of  (e().begin()  ,e().begin()+2, [](auto a){return a> 1UL;}) &&
         std::all_of  (e().begin()+2,e().end()    , [](auto a){return a==1UL;});
}


/** @brief Returns true if shape is has a tensor shape
 *
 * @returns true if is_valid & not (is_scalar&is_vector&is_matrix)
 */
template <class D>
[[nodiscard]] inline constexpr
  bool is_tensor(extents_base<D> const& e)
{
  return std::distance(e().begin()   ,e().end())>2 &&
         std::any_of  (e().begin()+2 ,e().end(), [](auto a){return a>1U;});
}


/** @brief Returns true if size > 1 and all elements > 0 or size == 1 && e[0] == 1 */
template <class D>
[[nodiscard]] inline constexpr
  bool is_valid(extents_base<D> const& e)
{
  if (std::distance(e().begin(),e().end()) < 1){
    return false;
  }

  if (std::distance(e().begin(),e().end()) == 1){
    return *e().begin() > 0;
  }

  return std::all_of(e().begin(),e().end(), [](auto a){ return a>0UL; } );
}

/** @brief Computes the product of all extents */
template <class D>
[[nodiscard]] inline constexpr
  auto product( extents_base<D> const& e )
{
  using value_type = typename D::value_type;
  if( std::distance(e().begin(),e().end()) <= 0 ){
    return value_type{0U};
  }

  auto acc = value_type{1U};
  for(auto const& ee : e()){
    acc *= ee;
  }

  return acc;


//  return std::accumulate(e().begin(),e().end(),1U,std::multiplies<>()) ;
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

template <class L, class R>
[[nodiscard]] inline constexpr
  bool operator==( extents_base<L> const& lhs,  extents_base<R> const& rhs )
{
  return lhs().size() == rhs().size() && std::equal( lhs().begin(), lhs().end(), rhs().begin() );
}

template <class L, class R>
[[nodiscard]] inline constexpr
  bool operator!=( extents_base<L> const& lhs, extents_base<R> const& rhs )
{
  return !( lhs == rhs) ;
}


} // namespace boost::numeric::ublas

#endif // _BOOST_NUMERIC_UBLAS_TENSOR_DETAIL_EXTENTS_FUNCTIONS_HPP_


