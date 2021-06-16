//
//  Copyright (c) 2020, Amit Singh, amitsingh19975@gmail.com
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef BOOST_UBLAS_TENSOR_ENGINE_HPP
#define BOOST_UBLAS_TENSOR_ENGINE_HPP

namespace boost::numeric::ublas{

template<typename E, typename L, typename C>
struct tensor_engine
{
  using extents_type 	 = E;
  using layout_type 	 = L;
  using container_type   = C;
};

} // namespace boost::numeric::ublas


#endif
