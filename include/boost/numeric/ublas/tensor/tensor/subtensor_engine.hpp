//
//  Copyright (c) 2021, Kannav Mehta, kmkannavkmehta@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef BOOST_UBLAS_SUBTENSOR_ENGINE_HPP
#define BOOST_UBLAS_SUBTENSOR_ENGINE_HPP


namespace boost::numeric::ublas
{

template <class T>
struct subtensor_engine
{
	using tensor_type    = std::decay_t<T>;
	using engine_type    = typename tensor_type::engine_type; // reference to the parent engine
	using extents_type   = typename tensor_type::extents_type; // reference to the parent extents
	using layout_type    = typename tensor_type::layout_type; // reference to the parent layout
	using container_type = typename tensor_type::container_type; // reference to the parent container
};


template<typename T>
using subtensor = tensor_core<subtensor_engine<T>>;

} // namespace boost::numeric::ublas

#endif
