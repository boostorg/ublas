//
//  Copyright (c) 2018-2019, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB, Ettlingen, Germany
//


#include <boost/numeric/ublas/tensor.hpp>
#include <iostream>
#include <type_traits>

int main() {
    
    using namespace boost::numeric::ublas;

    auto t1 = tensor<float>{1,2,3,4};
    auto t2 = tensor<float,dynamic_extents<4>>{dynamic_extents<4>{1,2,3,4}};
    auto t3 = tensor<float,dynamic_extents<>>{dynamic_extents<>{1,2,3,4}};
    auto t5 = tensor<float,dynamic_extents<>>{dynamic_extents<>{1,2,3,4},0.1}; 
    auto t4 = tensor<float,static_extents<1,2,3,4>>();

    auto t6 = tensor<float,static_extents<1,2,3,4>,first_order>();
    auto t7 = tensor<float,dynamic_extents<>,first_order>{1,2,3,4};

    auto t8 = tensor<float,dynamic_extents<>,first_order,std::vector<float>>{1,2,3,4};
    auto t9 = tensor<float,static_extents<1,2,3,4>,first_order,std::array<float,24>>();

    auto t1_e = t1.extents();
    auto t2_e = t2.extents();
    auto t3_e = t3.extents();
    auto t4_e = t4.extents();
    auto t5_e = t5.extents();
    auto t6_e = t6.extents();
    auto t7_e = t7.extents();
    auto t8_e = t8.extents();
    auto t9_e = t9.extents();

    auto t1_s = t1.strides();
    auto t2_s = t2.strides();
    auto t3_s = t3.strides();
    auto t4_s = t4.strides();
    auto t5_s = t5.strides();
    auto t6_s = t6.strides();
    auto t7_s = t7.strides();
    auto t8_s = t8.strides();
    auto t9_s = t9.strides();

    puts("\nt1: ");
    for(auto i = 0u; i < t1_e.size(); i++){
        std::cout<<t1_e[i]<<' ';
    }
    std::cout<<'\n';
    for(auto i = 0u; i < t1_s.size(); i++){
        std::cout<<t1_s[i]<<' ';
    }
    std::cout<<'\n';

    puts("\nt2: ");
    for(auto i = 0u; i < t2_e.size(); i++){
        std::cout<<t2_e[i]<<' ';
    }
    std::cout<<'\n';
    for(auto i = 0u; i < t2_s.size(); i++){
        std::cout<<t2_s.extent(i)<<' ';
    }
    std::cout<<'\n';
    
    puts("\nt3: ");
    for(auto i = 0u; i < t3_e.size(); i++){
        std::cout<<t3_e[i]<<' ';
    }
    std::cout<<'\n';
    for(auto i = 0u; i < t3_s.size(); i++){
        std::cout<<t3_s[i]<<' ';
    }
    std::cout<<'\n';
    
    puts("\nt4: ");
    for(auto i = 0u; i < t4_e.size(); i++){
        std::cout<<t4_e[i]<<' ';
    }
    std::cout<<'\n';
    for(auto i = 0u; i < t4_s.size(); i++){
        std::cout<<t4_s[i]<<' ';
    }
    std::cout<<'\n';
    
    puts("\nt5: ");
    for(auto i = 0u; i < t5_e.size(); i++){
        std::cout<<t5_e[i]<<' ';
    }
    std::cout<<'\n';
    for(auto i = 0u; i < t5_s.size(); i++){
        std::cout<<t5_s[i]<<' ';
    }
    std::cout<<'\n';
    
    puts("\nt6: ");
    for(auto i = 0u; i < t6_e.size(); i++){
        std::cout<<t6_e[i]<<' ';
    }
    std::cout<<'\n';
    for(auto i = 0u; i < t6_s.size(); i++){
        std::cout<<t6_s[i]<<' ';
    }
    std::cout<<'\n';
    
    puts("\nt7: ");
    for(auto i = 0u; i < t7_e.size(); i++){
        std::cout<<t7_e[i]<<' ';
    }
    std::cout<<'\n';
    for(auto i = 0u; i < t7_s.size(); i++){
        std::cout<<t7_s[i]<<' ';
    }
    std::cout<<'\n';
    
    puts("\nt8: ");
    for(auto i = 0u; i < t8_e.size(); i++){
        std::cout<<t8_e[i]<<' ';
    }
    std::cout<<'\n';
    for(auto i = 0u; i < t8_s.size(); i++){
        std::cout<<t8_s[i]<<' ';
    }
    std::cout<<'\n';
    
    puts("\nt9: ");
    for(auto i = 0u; i < t9_e.size(); i++){
        std::cout<<t9_e[i]<<' ';
    }
    std::cout<<'\n';
    for(auto i = 0u; i < t9_s.size(); i++){
        std::cout<<t9_s[i]<<' ';
    }
    std::cout<<'\n';
    
    static_assert(std::is_same<typename decltype(t8)::layout_type, first_order>::value,"");
    static_assert(std::is_same<typename decltype(t9)::layout_type, first_order>::value,"");
    static_assert(std::is_same<typename decltype(t8)::array_type, std::vector<float> >::value,"");
    static_assert(std::is_same<typename decltype(t9)::array_type, std::array<float,24> >::value,"");
    static_assert(detail::is_static_extents<typename decltype(t9)::extents_type >::value,"");
    static_assert(detail::is_dynamic<typename decltype(t8)::extents_type >::value,"");

    return 0;
}
