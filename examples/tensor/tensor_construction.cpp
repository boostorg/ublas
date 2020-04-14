//
// 	Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
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
    auto tensor_dynamic_extents_and_dynamic_rank_0  = ub::tensor<value_type>{1,2,3,4};
    
    /** Dynamic extents but static rank
     * Storage Type: std::array of static rank
     * Layout: first_order
     * Initial Value: 0
    **/
    auto tensor_dynamic_extents_and_static_rank_1   = ub::tensor<value_type,ub::dynamic_extents<4>>{ub::dynamic_extents<4>{1,2,3,4}};

    /** Dynamic extents but dynamic rank
     * Storage Type: std::vector
     * Layout: first_order
     * Initial Value: 0
    **/
    auto tensor_dynamic_extents_and_dynamic_rank_2  = ub::tensor<value_type,ub::dynamic_extents<>>{ub::dynamic_extents<>{1,2,3,4}};

    /** Dynamic extents but dynamic rank
     * Storage Type: std::vector
     * Layout: first_order
     * Initial Value: 0.1f
    **/
    auto tensor_dynamic_extents_and_dynamic_rank_3  = ub::tensor<value_type,ub::dynamic_extents<>>{ub::dynamic_extents<>{1,2,3,4},0.1};

    /** Static extents but static rank
     * Storage Type: std::array
     * Layout: first_order
     * Initial Value: 0
    **/
    auto tensor_static_extents_and_static_rank_4    = ub::tensor<value_type,ub::static_extents<1,2,3,4>>();

    /** Static extents but static rank
     * Storage Type: std::array
     * Layout: first_order
     * Initial Value: 0
    **/
    auto tensor_static_extents_and_static_rank_5    = ub::tensor<value_type,ub::static_extents<1,2,3,4>,ub::first_order>();

    /** Dynamic extents but dynamic rank
     * Storage Type: std::array
     * Layout: first_order
     * Initial Value: 0
    **/
    auto tensor_dynamic_extents_and_dynamic_rank_6  = ub::tensor<value_type,ub::dynamic_extents<>,ub::first_order>{1,2,3,4};

    /** Dynamic extents but dynamic rank
     * Storage Type: std::vector
     * Layout: first_order
     * Initial Value: 0
    **/
    auto tensor_dynamic_extents_and_dynamic_rank_7  = ub::tensor<value_type,ub::dynamic_extents<>,ub::first_order,std::vector<value_type>>{1,2,3,4};

    /** Static extents but static rank
     * Storage Type: std::array of size 24
     * Layout: first_order
     * Initial Value: 0
    **/
    auto tensor_static_extents_and_static_rank_8    = ub::tensor<value_type,ub::static_extents<1,2,3,4>,ub::first_order,std::array<value_type,24>>();

    auto t1_e = tensor_dynamic_extents_and_dynamic_rank_0.extents();
    auto t2_e = tensor_dynamic_extents_and_static_rank_1.extents();
    auto t3_e = tensor_dynamic_extents_and_dynamic_rank_2.extents();
    auto t4_e = tensor_static_extents_and_static_rank_4.extents();
    auto t5_e = tensor_dynamic_extents_and_dynamic_rank_3.extents();
    auto t6_e = tensor_static_extents_and_static_rank_5.extents();
    auto t7_e = tensor_dynamic_extents_and_dynamic_rank_6.extents();
    auto t8_e = tensor_dynamic_extents_and_dynamic_rank_7.extents();
    auto t9_e = tensor_static_extents_and_static_rank_8.extents();

    auto t1_s = tensor_dynamic_extents_and_dynamic_rank_0.strides();
    auto t2_s = tensor_dynamic_extents_and_static_rank_1.strides();
    auto t3_s = tensor_dynamic_extents_and_dynamic_rank_2.strides();
    auto t4_s = tensor_static_extents_and_static_rank_4.strides();
    auto t5_s = tensor_dynamic_extents_and_dynamic_rank_3.strides();
    auto t6_s = tensor_static_extents_and_static_rank_5.strides();
    auto t7_s = tensor_dynamic_extents_and_dynamic_rank_6.strides();
    auto t8_s = tensor_dynamic_extents_and_dynamic_rank_7.strides();
    auto t9_s = tensor_static_extents_and_static_rank_8.strides();

    std::cout<<"\nExtents: "<<(t1_e)<<'\n';
    std::cout<<"Strides: "<<(t1_s)<<'\n';
    
    std::cout<<"\nExtents: "<<(t2_e)<<'\n';
    std::cout<<"Strides: "<<(t2_s)<<'\n';

    std::cout<<"\nExtents: "<<(t3_e)<<'\n';
    std::cout<<"Strides: "<<(t3_s)<<'\n';

    std::cout<<"\nExtents: "<<(t4_e)<<'\n';
    std::cout<<"Strides: "<<(t4_s)<<'\n';

    std::cout<<"\nExtents: "<<(t5_e)<<'\n';
    std::cout<<"Strides: "<<(t5_s)<<'\n';

    std::cout<<"\nExtents: "<<(t6_e)<<'\n';
    std::cout<<"Strides: "<<(t6_s)<<'\n';

    std::cout<<"\nExtents: "<<(t7_e)<<'\n';
    std::cout<<"Strides: "<<(t7_s)<<'\n';

    std::cout<<"\nExtents: "<<(t8_e)<<'\n';
    std::cout<<"Strides: "<<(t8_s)<<'\n';

    std::cout<<"\nExtents: "<<(t9_e)<<'\n';
    std::cout<<"Strides: "<<(t9_s)<<'\n';

    /**
     *  Checking if layout is first order or not
     *  Output: True
     **/
    std::cout<<std::is_same<typename decltype(tensor_dynamic_extents_and_dynamic_rank_7)::layout_type, ub::first_order>::value<<'\n';
    
    /**
     *  Checking if layout is first order or not
     *  Output: True
     **/
    std::cout<<std::is_same<typename decltype(tensor_static_extents_and_static_rank_8)::layout_type, ub::first_order>::value<<'\n';
    
    /**
     *  Checking if container is std::vector
     *  Output: True
     **/
    std::cout<<std::is_same<typename decltype(tensor_dynamic_extents_and_dynamic_rank_7)::array_type, std::vector<float> >::value<<'\n';
    
    /**
     *  Checking if container is std::array
     *  Output: True
     **/
    std::cout<<std::is_same<typename decltype(tensor_static_extents_and_static_rank_8)::array_type, std::array<float,24> >::value<<'\n';
    
    /**
     *  Checking if extents is static
     *  Output: True
     **/
    std::cout<<ub::detail::is_static<typename decltype(tensor_static_extents_and_static_rank_8)::extents_type >::value<<'\n';
    
    /**
     *  Checking if extents is dynamic
     *  Output: True
     **/
    std::cout<<ub::detail::is_dynamic<typename decltype(tensor_dynamic_extents_and_dynamic_rank_7)::extents_type >::value<<'\n';

    return 0;
}
