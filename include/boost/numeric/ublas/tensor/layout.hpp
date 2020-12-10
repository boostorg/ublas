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

#ifndef BOOST_UBLAS_TENSOR_LAYOUT_HPP
#define BOOST_UBLAS_TENSOR_LAYOUT_HPP

#include <boost/numeric/ublas/functional.hpp>

namespace boost::numeric::ublas::layout{
    
    using first_order = column_major;
    using last_order = row_major;

} // namespace boost::numeric::ublas::layout


#endif // BOOST_UBLAS_TENSOR_LAYOUT_HPP
