//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2021, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//


#ifndef BOOST_NUMERIC_UBLAS_TENSOR_PROD_COMMON_HPP
#define BOOST_NUMERIC_UBLAS_TENSOR_PROD_COMMON_HPP

#include "../tensor_core.hpp"

namespace boost::numeric::ublas{


    /** @brief Reshapes the basic_tensor
     *
     *
     * (1) @code A.reshape(extents{m,n,o});     @endcode or
     * (2) @code A.reshape(extents{m,n,o},4);   @endcode
     *
     * If the size of this smaller than the specified extents than
     * default constructed (1) or specified (2) value is appended.
     *
     * @note rank of the basic_tensor might also change.
     *
     * @param e extents with which the basic_tensor is reshaped.
     * @param v value which is appended if the basic_tensor is enlarged.
     */
    template<typename E, typename D>
    constexpr
    void reshape (tensor_core<E>& in, extents_base<D> const& e, typename tensor_core<E>::value_type v = {})
    {
      using extents_type = typename tensor_core<E>::extents_type;
      using strides_type = typename tensor_core<E>::strides_type;
      using resizable_tag = typename tensor_core<E>::resizable_tag;

      static_assert(is_dynamic_v<extents_type> && is_dynamic_v<strides_type>,
          "Error in boost::numeric::ublas::basic_tensor::reshape(extents_type const&,value_type) : "
          "static extents or static strides cannot used inside reshape function"
      );

      auto& extents = in.extents();
      extents = e();
      in.strides() = strides_type(extents);

      auto p = product(extents);
      
      if( p != in.size() ){
        if constexpr( !std::is_same_v< resizable_tag, storage_resizable_container_tag > ){
            throw std::runtime_error(
                    "boost::numeric::ublas::basic_tensor::reshape(extents_type const&,value_type) : "
                    "cannot resize the non-resizable container, change the extents such a way that the product does not change"
                );
            }
        else in.base().resize (p, v);
      }
    }

} // namespace boost::numeric::ublas

#endif   // BOOST_NUMERIC_UBLAS_TENSOR_PROD_COMMON_HPP
