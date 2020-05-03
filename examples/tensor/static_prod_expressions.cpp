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
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <iostream>

int main()
{
    using namespace boost::numeric::ublas;

    using format_t  = column_major;
    using value_t   = float; // std::complex<double>;
    using matrix_t = matrix<value_t,format_t>;
    using vector_t = vector<value_t>;

    // Tensor-Vector-Multiplications - Including Transposition
    // static_extents
    {
        auto n = static_extents<3,4,2>{};
        auto A = static_tensor<value_t,static_extents<3,4,2>>(value_t(2));
        auto q = 0u; // contraction mode

        // C1(j,k) = T2(j,k) + A(i,j,k)*T1(i);
        q = 1u;
        static_tensor< value_t, static_extents<4,2> > C1 = matrix_t(n[1],n[2],2) + prod<1ul /* q = 1 */ >(A,vector_t(n[q-1],1));

        // C2(i,k) = A(i,j,k)*T1(j) + 4;
        q = 2u;
        static_tensor< value_t, static_extents<3,2> > C2 = prod<2ul /* q = 2 */ >(A,vector_t(n[q-1],1)) + 4;

        // C3() = A(i,j,k)*T1(i)*T2(j)*T2(k);  
        auto C3 = prod(prod(prod(A,vector_t(n[0],1),1),vector_t(n[1],1),1),vector_t(n[2],1),1);



        // formatted output
        std::cout << "% --------------------------- " << std::endl;
        std::cout << "% --------------------------- " << std::endl << std::endl;
        std::cout << "% C1(j,k) = T2(j,k) + A(i,j,k)*T1(i);" << std::endl << std::endl;
        std::cout << "C1=" << C1 << ";" << std::endl << std::endl;

        // formatted output
        std::cout << "% --------------------------- " << std::endl;
        std::cout << "% --------------------------- " << std::endl << std::endl;
        std::cout << "% C2(i,k) = A(i,j,k)*T1(j) + 4;" << std::endl << std::endl;
        std::cout << "C2=" << C2 << ";" << std::endl << std::endl;

        // formatted output
        std::cout << "% --------------------------- " << std::endl;
        std::cout << "% --------------------------- " << std::endl << std::endl;
        std::cout << "% C3() = A(i,j,k)*T1(i)*T2(j)*T2(k);" << std::endl << std::endl;
        std::cout << "C3()=" << C3(0) << ";" << std::endl << std::endl;

    }

    // Tensor-Matrix-Multiplications - Including Transposition
    // static_extents
    {

        auto n = static_extents<3,4,2>{};
        auto A = static_tensor<value_t,static_extents<3,4,2>>(value_t(2));
        static constexpr auto m = 5u;
        auto q = 0u; // contraction mode

        // C1(l,j,k) = T2(l,j,k) + A(i,j,k)*T1(l,i);
        q = 1ul;
        static_tensor< value_t, static_extents<5,4,2> > C1 = static_tensor<value_t,static_extents<m,4,2>>( value_t(2) )
            + prod < 1ul /* q */,  m /*n[q-1] => n[1-1] => n[0]*/ > (A, matrix_t(m,n[q-1],value_t(1) )) ;

        // C2(i,l,k) = A(i,j,k)*T1(l,j) + 4;
        q = 2u;
        static_tensor< value_t, static_extents<3,5,2> > C2 = prod<2ul/* q = 2 */, m>(A,matrix_t(m,n[q-1],1)) + 4 ;

        // C3(i,l1,l2) = A(i,j,k)*T1(l1,j)*T2(l2,k);
        q = 3u;
        auto C3 = prod<3ul, m + 2>( prod<2ul /* q(3) - 1 = 2 */, m + 1 >(A,matrix_t(m+1,n[q-2],1)), matrix_t(m+2,n[q-1],1) ) ;

        // C4(i,l1,l2) = A(i,j,k)*T2(l2,k)*T1(l1,j);
        auto C4 = prod<2ul, m + 1>( prod<3ul, m + 2>(A,matrix_t(m+2,n[q-1],1)), matrix_t(m+1,n[q-2],1) ) ;

        // formatted output
        std::cout << "% --------------------------- " << std::endl;
        std::cout << "% --------------------------- " << std::endl << std::endl;
        std::cout << "% C1(l,j,k) = T2(l,j,k) + A(i,j,k)*T1(l,i);" << std::endl << std::endl;
        std::cout << "C1=" << C1 << ";" << std::endl << std::endl;

        // formatted output
        std::cout << "% --------------------------- " << std::endl;
        std::cout << "% --------------------------- " << std::endl << std::endl;
        std::cout << "% C2(i,l,k) = A(i,j,k)*T1(l,j) + 4;" << std::endl << std::endl;
        std::cout << "C2=" << C2 << ";" << std::endl << std::endl;

        // formatted output
        std::cout << "% --------------------------- " << std::endl;
        std::cout << "% --------------------------- " << std::endl << std::endl;
        std::cout << "% C3(i,l1,l2) = A(i,j,k)*T1(l1,j)*T2(l2,k);" << std::endl << std::endl;
        std::cout << "C3=" << C3 << ";" << std::endl << std::endl;

        // formatted output
        std::cout << "% --------------------------- " << std::endl;
        std::cout << "% --------------------------- " << std::endl << std::endl;
        std::cout << "% C4(i,l1,l2) = A(i,j,k)*T2(l2,k)*T1(l1,j);" << std::endl << std::endl;
        std::cout << "C4=" << C4 << ";" << std::endl << std::endl;
        std::cout << "% C3 and C4 should have the same values, true? " << std::boolalpha << (C3 == C4) << "!" << std::endl;

    }

}
