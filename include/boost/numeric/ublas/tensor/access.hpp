//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB, Ettlingen, Germany
//


#ifndef _BOOST_UBLAS_TENSOR_ACCESS_HPP
#define _BOOST_UBLAS_TENSOR_ACCESS_HPP


#include <algorithm>
#include <numeric>
#include <functional>

namespace boost::numeric::ublas::detail{


/** \brief Computes a single index from multi-index of a tensor or subtensor
 *
 * \param i iterator to a multi-index vector of length std::distance(begin,end)
 * \param ip iterator to a multi-index vector of length std::distance(begin,end)
 * \param w iterator to a stride vector of length std::distance(begin,end) or greater
*/
template<typename InputIt1, typename InputIt2>
constexpr inline auto compute_single_index(InputIt1 i, InputIt1 ip, InputIt2 w)
{
  return std::inner_product(i,ip,w,0ul,std::plus<>{},std::multiplies<>{});
}


/** \brief Computes a single index from a multi-index of a tensor or subtensor
 *
 * \param i iterator to a multi-index vector of length std::distance(begin,end)
 * \param ip iterator to a multi-index vector of length std::distance(begin,end)
 * \param w iterator to a stride vector of length std::distance(begin,end) or greater
*/
template<unsigned p, typename InputIt1, typename InputIt2>
constexpr inline auto compute_single_index(InputIt1 i, InputIt1 /*ip*/, InputIt2 w)
{
       if constexpr(p==0u) return 0ul;
  else if constexpr(p >1u) return compute_single_index<p-1>(i,i,w)+i[p-1]*w[p-1];
  else                     return i[p-1]*w[p-1];
}

/** @brief Computes a multi-index from single index of a tensor or subtensor
 *
 * j = compute_single_index (i, ip, w)
 * compute_multi_index  (j, w, wp, k) with k == i
 *
 * @param w begin input iterator to a container with tensor or subtensor strides of length std::distance(begin,end)
 * @param wp end input iterator to a container with tensor or subtensor strides of length std::distance(begin,end)
 * @param i begin output iterator to a container with tensor or subtensor indices length std::distance(begin,end) or greater
*/
template<typename InputIt1, typename OutputIt>
constexpr inline void compute_multi_index(std::size_t j, InputIt1 w, InputIt1 wp, OutputIt i)
{
    if(w == wp)
        return;
    auto p = std::distance(w,wp);
    auto kq = j;
    //auto q = 0ul;



    for(int r = p-1; r >= 0; --r)
    {
        //q  = l[r]-1;
        i[r] = kq/w[r];
        kq -= w[r]*i[r];
    }

  //std::transform(w,wp,i, [&j](auto v) { auto k=j/v; j-=v*k; return k; } );
}


template<typename InputIt1, typename OutputIt>
constexpr inline void compute_multi_index_first(std::size_t j, InputIt1 w, InputIt1 wp, OutputIt i)
{
    if(w == wp)
        return;
    auto p = std::distance(w,wp);
    auto kq = j;
    //auto q = 0ul;



//    for(int r = p-1; r >= 0; --r)
//    {
//        //q  = l[r]-1;
//        i[r] = kq/w[r];
//        kq -= w[r]*i[r];
//    }

    std::transform(w,wp,i, [&j](auto v) { auto k=j/v; j-=v*k; return k; } );
}

template<typename InputIt1, typename OutputIt>
constexpr inline void compute_multi_index_last(std::size_t j, InputIt1 w, InputIt1 wp, OutputIt i)
{
    if(w == wp)
        return;
    auto p = std::distance(w,wp);
    auto kq = j;

    for(unsigned r = 0ul; r < p; ++r) {
        i[r] = kq/w[r];
        kq -= w[r]*i[r];
    }

    //std::transform(w,wp,i, [&j](auto v) { auto k=j/v; j-=v*k; return k; } );
}



/** @brief Computes a single index from a multi-index of a dense tensor or subtensor
 *
 * @param j single index that is transformed into a multi-index
 * @param w begin input iterator to a container with strides of length p
 * @param i begin input iterator to a container with indices of length p or greater
*/
template<unsigned p, typename InputIt1, typename OutputIt>
constexpr inline void compute_multi_index(std::size_t j, InputIt1 w, InputIt1 /*wp*/, OutputIt i)
{
       if constexpr (p==0u) return;
  else if constexpr (p >1u) {i[p-1]=j/w[p-1]; compute_multi_index<p-1>(j-w[p-1]*i[p-1],w,w,i); }
  else                      {i[p-1]=j/w[p-1]; }
}


/** @brief Computes a single (relative memory) index of a dense tensor from a single index of one of its subtensor
 *
 * @param jv single index of a subtensor that is transformed into a single index of a dense tensor
 * @param w begin input iterator of a container with tensor strides of length std::distance(w,wp)
 * @param wp end input iterator of a container with tensor strides of length std::distance(w,wp)
 * @param v begin input iterator of a container with subtensor strides of length std::distance(w,wp) or greater
*/
template<typename InputIt1, typename InputIt2>
constexpr inline auto compute_single_index(std::size_t jv, InputIt1 w, InputIt1 wp, InputIt2 v)
{
  return std::inner_product(w,wp,v,0ul,
    std::plus<>{},
    [&jv](auto ww, auto vv) { auto k=jv/vv; jv-=vv*k; return ww*k; }
  );
}

} // namespace

#endif
