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

    /** @brief How to create fixed_rank_tensor
     *  
     * fixed_rank_tensor< / Value Type, / Extents Size /, / Layout Type / >( ... );
     * 
    **/

    /**
     * fixed rank tensor with 
     * Layout: first_order by default if not 
     * provided
     * Value Type: float
     * Extents Type: dynamic_extents<5>{1,2,3,4,5}
    **/
    auto t11 = ub::fixed_rank_tensor<float,5>{1,2,3,4,5};

    /**
     * fixed rank tensor using deduction guide
     * Layout: last_order 
     * provided
     * Value Type: float
     * Extents Type: dynamic_extents<5>{1,2,2,3,4}
    **/
    auto t12 = ub::fixed_rank_tensor<float,5,ub::last_order>{ ub::dynamic_extents<5>{1,2,2,3,4}, 5.f };

    /**
     * fixed rank tensor using deduction guide
     * Layout: first_order by default if not 
     * provided
     * Value Type: float
     * Extents Type: dynamic_extents<5>{1,2,2,3,4}
    **/
    auto t2 = ub::fixed_rank_tensor<float,5>( ub::dynamic_extents<5>{1,2,2,3,4}, 5.f );

    /*--------------------------Expressions------------------------------*/
    
    using f_tensor = ub::fixed_rank_tensor<float,5>;
    using s_tensor = ub::static_tensor<float, ub::static_extents<1,2,3,4,5> >;
    
    auto t3 = ub::fixed_rank_tensor<float,5>( ub::dynamic_extents<5>{1,2,3,4,5}, 5.f );
    auto st1 = s_tensor{ ub::static_extents<1,2,3,4,5>{}, 5.f };
    auto dt1 = ub::dynamic_tensor<float>( ub::dynamic_extents<>{1,2,3,4,5}, 5.f );

    f_tensor exp1 = t3 + t3 * 2 + t3*t3;
    
    // type of the expression will be whichever
    // tensor type is present at the start of the
    // expression
    f_tensor exp2 = t3 + st1 * 2 + st1 * t3;

    f_tensor exp3 = t3 + dt1 * 2 + dt1 * t3;

    std::cout<<std::boolalpha;

    std::cout<<"( exp1 == exp2 ) => "<<( exp1 == exp2 )<<std::endl;
    
    std::cout<<"( exp2 == exp3 ) => "<<( exp2 == exp3 )<<std::endl;

    /*--------------------------Assignment------------------------------*/

    // Assigning static_tensor into fixed_rank_tensor
    ub::fixed_rank_tensor<float,5> t4 = st1;

    // Assigning dynamic_tensor into fixed_rank_tensor
    ub::dynamic_tensor<float> const& t5 = dt1;

    std::cout<<"( t4 == t3 ) => "<<( t4 == t3 )<<std::endl;
    
    std::cout<<"( t5 == dt1 ) => "<<( t5 == dt1 )<<std::endl;

    return 0;
}
