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

    /** @brief How to create static_tensor
     *  
     * static_tensor< / Value Type, / Extents /, / Layout Type / >( ... );
     * 
    **/

    /**
     * static rank tensor with 
     * Layout: first_order by default if not 
     * provided
     * Value Type: float
     * Extents Type: dynamic_extents<5>{1,2,3,4,5}
    **/
    auto t1 = ub::static_tensor<float,ub::static_extents<1,2,3,4,5> >{5.f};

    /**
     * static rank tensor with 
     * Layout: first_order by default if not 
     * provided
     * Value Type: float
     * Extents Type: dynamic_extents<5>{1,2,3,4,5}
    **/
    auto t2 = ub::static_tensor<float,ub::static_extents<1,2,3,4,5>, ub::last_order >{5.f};

    /*--------------------------Expressions------------------------------*/
    
    using s_tensor = ub::static_tensor<float, ub::static_extents<1,2,3,4,5> >;
    
    auto t3 = ub::fixed_rank_tensor{ ub::dynamic_extents<5>{1,2,3,4,5}, 5.f };
    auto st1 = s_tensor{ 5.f };
    auto dt1 = ub::dynamic_tensor{ ub::dynamic_extents<>{1,2,3,4,5}, 5.f };

    s_tensor exp1 = st1 + st1 * 2 + st1*st1;
    
    // type of the expression will be whichever
    // tensor type is present at the start of the
    // expression
    s_tensor exp2 = st1 + t3 * 2 + t3 * st1;

    s_tensor exp3 = st1 + dt1 * 2 + dt1 * t3;

    std::cout<<"( exp1 == exp2 ) => "<<( exp1 == exp2 ? "True" : "False")<<'\n';
    
    std::cout<<"( exp2 == exp3 ) => "<<( exp2 == exp3 ? "True" : "False")<<'\n';



    return 0;
}
