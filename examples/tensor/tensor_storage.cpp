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
    using namespace storage::sparse_tensor;
    using namespace storage::dense_tensor;

    auto t1 = tensor<float,dynamic_extents<>,first_order,std::vector<float>>{1,2,3,4};
    auto t2 = tensor<float,dynamic_extents<>,last_order, std::array<float,24> >(dynamic_extents<>{1,2,3,4},4.f);
    
    //Sparse Tensor with compressed_map as storage type
    auto t3 = tensor<float,dynamic_extents<>,first_order, compressed_map<float> >{1,2,3,4};// container size is 0
    auto t4 = tensor<float,dynamic_extents<>,first_order, compressed_map<float> >(dynamic_extents<>{1,2,3,4},4.f);// container size is 24
    auto t5 = tensor<float,dynamic_extents<4>,first_order, compressed_map<float> >(dynamic_extents<4>{1,2,3,4},4.f);// container size is 24
    auto t6 = tensor<float,static_extents<1,2,3,4>,first_order, compressed_map<float> >(static_extents<1,2,3,4>{},4.f);// container size is 24

    std::cout<<"t3: ";
    for(auto i = 0u; i < t3.size(); i++){
        std::cout<<t3[i]<<' ';
    }
    std::cout<<'\n'<<"t4: ";

    for(auto i = 0u; i < t4.size(); i++){
        std::cout<<t4[i]<<' ';
    }
    std::cout<<'\n'<<"t5: ";

    for(auto i = 0u; i < t5.size(); i++){
        std::cout<<t5[i]<<' ';
    }
    std::cout<<'\n'<<"t6: ";

    for(auto i = 0u; i < t6.size(); i++){
        std::cout<<t6[i]<<' ';
    }
    std::cout<<'\n'<<"Modified t6: ";
    t6[0] = 0;
    t6[1] = 0;
    t6[2] = 0;
    t6.prune(); // It will prune the 0 and reduce the size of container so new container size will be 21
    for(auto i = 0u; i < t6.size(); i++){
        std::cout<<t6[i]<<' ';
    }
    std::cout<<'\n';

    return 0;
}
