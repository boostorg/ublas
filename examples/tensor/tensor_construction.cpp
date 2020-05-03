//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google
//


#include <boost/numeric/ublas/tensor.hpp>
#include <iostream>
#include <type_traits>

namespace ub = boost::numeric::ublas;

int main() {
    
    using value_type = float;

    /** tensor initialization using type deduction 
     * Storage Type: std::vector
     * Layout: first_order
     * Initial Value: 0
    **/
    auto tensor_dynamic_extents_and_dynamic_rank_0  = ub::dynamic_tensor<value_type>{1,2,3,4};
    
    /** Dynamic extents but static rank
     * Storage Type: std::array of static rank
     * Layout: first_order
     * Initial Value: 0
    **/
    auto tensor_dynamic_extents_and_static_rank_1   = ub::fixed_rank_tensor<value_type,4>{ub::dynamic_extents<4>{1,2,3,4}};

    /** Dynamic extents but dynamic rank
     * Storage Type: std::vector
     * Layout: first_order
     * Initial Value: 0
    **/
    auto tensor_dynamic_extents_and_dynamic_rank_2  = ub::dynamic_tensor<value_type>{ub::dynamic_extents<>{1,2,3,4}};

    /** Dynamic extents but dynamic rank
     * Storage Type: std::vector
     * Layout: first_order
     * Initial Value: 0.1f
    **/
    auto tensor_dynamic_extents_and_dynamic_rank_3  = ub::dynamic_tensor<value_type>{ub::dynamic_extents<>{1,2,3,4},0.1};

    /** Static extents but static rank
     * Storage Type: std::array
     * Layout: first_order
     * Initial Value: 0
    **/
    auto tensor_static_extents_and_static_rank_4    = ub::static_tensor<value_type,ub::static_extents<1,2,3,4>>();

    /** Static extents but static rank
     * Storage Type: std::array
     * Layout: first_order
     * Initial Value: 0
    **/
    auto tensor_static_extents_and_static_rank_5    = ub::static_tensor<value_type,ub::static_extents<1,2,3,4>,ub::first_order>();

    /** Dynamic extents but dynamic rank
     * Storage Type: std::array
     * Layout: first_order
     * Initial Value: 0
    **/
    auto tensor_dynamic_extents_and_dynamic_rank_6  = ub::dynamic_tensor<value_type,ub::first_order>{1,2,3,4};

    auto t1_e = tensor_dynamic_extents_and_dynamic_rank_0.extents();
    auto t2_e = tensor_dynamic_extents_and_static_rank_1.extents();
    auto t3_e = tensor_dynamic_extents_and_dynamic_rank_2.extents();
    auto t4_e = tensor_static_extents_and_static_rank_4.extents();
    auto t5_e = tensor_dynamic_extents_and_dynamic_rank_3.extents();
    auto t6_e = tensor_static_extents_and_static_rank_5.extents();
    auto t7_e = tensor_dynamic_extents_and_dynamic_rank_6.extents();

    auto t1_s = tensor_dynamic_extents_and_dynamic_rank_0.strides();
    auto t2_s = tensor_dynamic_extents_and_static_rank_1.strides();
    auto t3_s = tensor_dynamic_extents_and_dynamic_rank_2.strides();
    auto t4_s = tensor_static_extents_and_static_rank_4.strides();
    auto t5_s = tensor_dynamic_extents_and_dynamic_rank_3.strides();
    auto t6_s = tensor_static_extents_and_static_rank_5.strides();
    auto t7_s = tensor_dynamic_extents_and_dynamic_rank_6.strides();

    std::cout<<"\nExtents: "<<(t1_e)<<std::endl;
    std::cout<<"Strides: "<<(t1_s)<<std::endl;
    
    std::cout<<"\nExtents: "<<(t2_e)<<std::endl;
    std::cout<<"Strides: "<<(t2_s)<<std::endl;

    std::cout<<"\nExtents: "<<(t3_e)<<std::endl;
    std::cout<<"Strides: "<<(t3_s)<<std::endl;

    std::cout<<"\nExtents: "<<(t4_e)<<std::endl;
    std::cout<<"Strides: "<<(t4_s)<<std::endl;

    std::cout<<"\nExtents: "<<(t5_e)<<std::endl;
    std::cout<<"Strides: "<<(t5_s)<<std::endl;

    std::cout<<"\nExtents: "<<(t6_e)<<std::endl;
    std::cout<<"Strides: "<<(t6_s)<<std::endl;

    std::cout<<"\nExtents: "<<(t7_e)<<std::endl;
    std::cout<<"Strides: "<<(t7_s)<<std::endl;

    return 0;
}
