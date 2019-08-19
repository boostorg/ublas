#include <boost/numeric/ublas/tensor.hpp>
#include <iostream>

namespace ub = boost::numeric::ublas;
namespace sp = ub::span;

int main(){

    auto t1 = ub::tensor{ ub::static_extents<10,10>{}, 1.f };
    auto t2 = ub::tensor{ ub::dynamic_extents<>{10,10}, 1.f};

    for(auto i = 0; i < 10; i ++){
        for(auto j = 0; j < 10; j ++){
            t1.at(i,j) = i * 10 + j;
            t2.at(i,j) = i * 10 + j;
        }
    }

    // statically sliced subtensor and can only be used with static tensor
    auto s1 = ub::subtensor<decltype(t1),sp::slice<2,6>,sp::slice<>>(t1);

    auto s2 = ub::subtensor<decltype(t1),sp::slice<2>,sp::slice<2,6,3>>(t1);

    auto s3 = ub::subtensor<decltype(t1),sp::slice<>,sp::slice<>>(t1);

    //-----------------------------------------------------------------//

    // dynamically sliced subtensor
    auto d1 = ub::subtensor { t2 };
    
    auto d2 = ub::subtensor { t2, sp::slice<>{2,6}, sp::slice<>{} };

    auto d3 = ub::subtensor { t2, sp::slice<>{2}, sp::slice<>{2,6,3} };

    //---------------------------From Tensor----------------------------//
    
    // --> static slice <--
    
    auto ts1 = t1(sp::slice<2,6>{},sp::slice<>{}); 
    
    auto ts2 = t1(sp::slice<2>{},sp::slice<2,6,3>{});
    
    /**
     * @brief sp::slice<2,-1> == sp::slice<2,extents[0] - 1> 
     */
    auto ts3 = t1(sp::slice<2,-1>{},sp::slice<2,6,3>{});

    /**
     * @brief sp::slice<-6,-1, 2> == sp::slice<extents[0] - 6 ,extents[0] - 1, 2> 
     */
    auto ts4 = t1(sp::slice<-6,-1, 2>{},sp::slice<2,6,3>{});

  
    // --> dynamic slice <--

    auto td1 = t1(sp::slice<>{2,6},sp::slice<>{}); 
    
    auto td2 = t1(sp::slice<>{2},sp::slice<>{2,6,3});
    
    /**
     * @brief sp::slice<>{2,-1} == sp::slice<>{2,extents[0] - 1} 
     */
    auto td3 = t1(sp::slice<>{2,-1},sp::slice<>{2,6,3});

    /**
     * @brief sp::slice<>{-6,-1, 2} == sp::slice<>{extents[0] - 6 ,extents[0] - 1, 2} 
     */
    auto td4 = t1(sp::slice<>{-6,-1, 2},sp::slice<>{2,6,3});
    
    auto td5 = t1(sp::slice<>{},sp::slice<>{});

    return 0;
}