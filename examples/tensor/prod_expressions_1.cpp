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
	// dynamic_extents with dynamic rank
	{

		auto n = dynamic_extents<>{3,4,2};
		auto A = tensor(n,value_t(2));
		using tensor_t = decltype(A);
		auto q = 0u; // contraction mode

		// C1(j,k) = T2(j,k) + A(i,j,k)*T1(i);
		q = 1u;
		tensor_t C1 = matrix_t(n[1],n[2],2) + prod(A,vector_t(n[q-1],1),q) ;

		// C2(i,k) = A(i,j,k)*T1(j) + 4;
		q = 2u;
		tensor_t C2 = prod(A,vector_t(n[q-1],1),q) + 4 ;

		// C3() = A(i,j,k)*T1(i)*T2(j)*T2(k);		
		tensor_t C3 = prod(prod(prod(A,vector_t(n[0],1),1),vector_t(n[1],1),1),vector_t(n[2],1),1);

		// C4(i,j) = A(k,i,j)*T1(k) + 4;
		q = 1u;
		tensor_t C4 = prod(trans(A,{2,3,1}),vector_t(n[2],1),q) + 4 ;


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

		// formatted output
		std::cout << "% --------------------------- " << std::endl;
		std::cout << "% --------------------------- " << std::endl << std::endl;
		std::cout << "% C4(i,j) = A(k,i,j)*T1(k) + 4;" << std::endl << std::endl;
		std::cout << "C4=" << C4 << ";" << std::endl << std::endl;

	}

	// Tensor-Vector-Multiplications - Including Transposition
	// dynamic_extents with static rank
	{

		auto n = dynamic_extents<3>{3,4,2};
		auto A = tensor(n,value_t(2));
		using tensor_t = tensor< value_t, dynamic_extents<2> >;
		auto q = 0u; // contraction mode

		// C1(j,k) = T2(j,k) + A(i,j,k)*T1(i);
		q = 1u;
		tensor_t C1 = matrix_t(n[1],n[2],2) + prod(A,vector_t(n[q-1],1),q);

		// C2(i,k) = A(i,j,k)*T1(j) + 4;
		q = 2u;
		tensor_t C2 = prod(A,vector_t(n[q-1],1),q) + 4;

		// C3() = A(i,j,k)*T1(i)*T2(j)*T2(k);		
		tensor_t C3 = prod(prod(prod(A,vector_t(n[0],1),1),vector_t(n[1],1),1),vector_t(n[2],1),1);


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

	// Tensor-Vector-Multiplications - Including Transposition
	// static_extents
	{

		auto n = static_extents<3,4,2>{};
		auto A = tensor(n,value_t(2));
		auto q = 0u; // contraction mode

		// C1(j,k) = T2(j,k) + A(i,j,k)*T1(i);
		q = 1u;
		tensor< value_t, static_extents<4,2> > C1 = matrix_t(n[1],n[2],2) + prod<1ul /* q = 1 */ >(A,vector_t(n[q-1],1));

		// C2(i,k) = A(i,j,k)*T1(j) + 4;
		q = 2u;
		tensor< value_t, static_extents<3,2> > C2 = prod<2ul /* q = 2 */ >(A,vector_t(n[q-1],1)) + 4;

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
	// dynamic_extents with dynamic rank
	{

		auto n = dynamic_extents<>{3,4,2};
		auto A = tensor(n,value_t(2));
		using tensor_t = decltype(A);
		auto m = 5u;
		auto q = 0u; // contraction mode

		// C1(l,j,k) = T2(l,j,k) + A(i,j,k)*T1(l,i);
		q = 1u;
		tensor_t C1 = tensor( dynamic_extents<>{m,n[1],n[2]},value_t(2) ) + prod(A,matrix_t(m,n[q-1],1),q);

		// C2(i,l,k) = A(i,j,k)*T1(l,j) + 4;
		q = 2u;
		tensor_t C2 = prod(A,matrix_t(m,n[q-1],1),q) + 4 ;

		// C3(i,l1,l2) = A(i,j,k)*T1(l1,j)*T2(l2,k);
		q = 3u;
		tensor_t C3 = prod(prod(A,matrix_t(m+1,n[q-2],1),q-1),matrix_t(m+2,n[q-1],1),q) ;

		// C4(i,l1,l2) = A(i,j,k)*T2(l2,k)*T1(l1,j);
		tensor_t C4 = prod(prod(A,matrix_t(m+2,n[q-1],1),q),matrix_t(m+1,n[q-2],1),q-1) ;

		// C5(i,k,l) = A(i,k,j)*T1(l,j) + 4;
		q = 3u;
		tensor_t C5 = prod(trans(A,{1,3,2}),matrix_t(m,n[1],1),q) + 4 ;

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


		// formatted output
		std::cout << "% --------------------------- " << std::endl;
		std::cout << "% --------------------------- " << std::endl << std::endl;
		std::cout << "% C5(i,k,l) = A(i,k,j)*T1(l,j) + 4;" << std::endl << std::endl;
		std::cout << "C5=" << C5 << ";" << std::endl << std::endl;
	}

	// Tensor-Matrix-Multiplications - Including Transposition
	// dynamic_extents with static rank
	{

		auto n = dynamic_extents<3>{3,4,2};
		auto A = tensor(n,value_t(2));
		using tensor_t = decltype(A);
		auto m = 5u;
		auto q = 0u; // contraction mode

		// C1(l,j,k) = T2(l,j,k) + A(i,j,k)*T1(l,i);
		q = 1u;
		tensor_t C1 = tensor( dynamic_extents<3>{m,n[1],n[2]},value_t(2) ) + prod(A,matrix_t(m,n[q-1],1),q);

		// C2(i,l,k) = A(i,j,k)*T1(l,j) + 4;
		q = 2u;
		tensor_t C2 = prod(A,matrix_t(m,n[q-1],1),q) + 4 ;

		// C3(i,l1,l2) = A(i,j,k)*T1(l1,j)*T2(l2,k);
		q = 3u;
		tensor_t C3 = prod(prod(A,matrix_t(m+1,n[q-2],1),q-1),matrix_t(m+2,n[q-1],1),q) ;

		// C4(i,l1,l2) = A(i,j,k)*T2(l2,k)*T1(l1,j);
		tensor_t C4 = prod(prod(A,matrix_t(m+2,n[q-1],1),q),matrix_t(m+1,n[q-2],1),q-1) ;

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

	// Tensor-Matrix-Multiplications - Including Transposition
	// static_extents
	{

		auto n = static_extents<3,4,2>{};
		auto A = tensor(n,value_t(2));
		static constexpr auto m = 5u;
		auto q = 0u; // contraction mode

		// C1(l,j,k) = T2(l,j,k) + A(i,j,k)*T1(l,i);
		q = 1ul;
		tensor< value_t, static_extents<5,4,2> > C1 = tensor( static_extents<m,4,2>{},value_t(2) )
			+ prod < 1ul /* q */,  m /*n[q-1] => n[1-1] => n[0]*/ > (A, matrix_t(m,n[q-1],value_t(1) )) ;

		// C2(i,l,k) = A(i,j,k)*T1(l,j) + 4;
		q = 2u;
		tensor< value_t, static_extents<3,5,2> > C2 = prod<2ul/* q = 2 */, m>(A,matrix_t(m,n[q-1],1)) + 4 ;

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





	// Tensor-Tensor-Multiplications Including Transposition
	// dynamic_extents with dynamic rank
	{

		using perm_t = std::vector<std::size_t>;

		auto na = dynamic_extents<>{3,4,5};
		auto nb = dynamic_extents<>{4,6,3,2};
		auto A = tensor(na,value_t(2));
		auto B = tensor(nb,value_t(3));
		using tensor_t = decltype(A);


		// C1(j,l) = T(j,l) + A(i,j,k)*A(i,j,l) + 5;
		tensor_t C1 = tensor(dynamic_extents<>{na[2],na[2]},value_t(2)) + prod(A,A,perm_t{1,2}) + 5;

		// formatted output
		std::cout << "% --------------------------- " << std::endl;
		std::cout << "% --------------------------- " << std::endl << std::endl;
		std::cout << "% C1(k,l) = T(k,l) + A(i,j,k)*A(i,j,l) + 5;" << std::endl << std::endl;
		std::cout << "C1=" << C1 << ";" << std::endl << std::endl;


		// C2(k,l,m) = T(k,l,m) + A(i,j,k)*B(j,l,i,m) + 5;
		tensor_t C2 = tensor(dynamic_extents<>{na[2],nb[1],nb[3]},value_t(2)) + prod(A,B,perm_t{1,2},perm_t{3,1}) + 5;

		// formatted output
		std::cout << "% --------------------------- " << std::endl;
		std::cout << "% --------------------------- " << std::endl << std::endl;
		std::cout << "%  C2(k,l,m) = T(k,l,m) + A(i,j,k)*B(j,l,i,m) + 5;" << std::endl << std::endl;
		std::cout << "C2=" << C2 << ";" << std::endl << std::endl;


		// C3(k,l,m) = T(k,l,m) + A(i,j,k)*trans(B(j,l,i,m),{2,3,1,4})+ 5;
		tensor_t C3 = tensor(dynamic_extents<>{na[2],nb[1],nb[3]},value_t(2)) + prod(A,trans(B,{2,3,1,4}),perm_t{1,2}) + 5;

		// formatted output
		std::cout << "% --------------------------- " << std::endl;
		std::cout << "% --------------------------- " << std::endl << std::endl;
		std::cout << "%  C3(k,l,m) = T(k,l,m) + A(i,j,k)*trans(B(j,l,i,m),{2,3,1,4})+ 5;" << std::endl << std::endl;
		std::cout << "C3=" << C3 << ";" << std::endl << std::endl;

	}

	// Tensor-Tensor-Multiplications Including Transposition
	// dynamic_extents with static rank
	{

		using perm_t = std::vector<std::size_t>;

		auto na = dynamic_extents<3>{3,4,5};
		auto nb = dynamic_extents<4>{4,6,3,2};
		auto A = tensor(na,value_t(2));
		auto B = tensor(nb,value_t(3));
		using tensor_t = tensor< value_t, dynamic_extents<> >;


		// C1(j,l) = T(j,l) + A(i,j,k)*A(i,j,l) + 5;
		// Right now there exist no tensor other than dynamic_extents with 
		// dynamic rank so every tensor times tensor operator automatically
		// to dynamic tensor
		tensor_t C1 = tensor(dynamic_extents<>{na[2],na[2]},value_t(2)) + prod(A,A,perm_t{1,2})  + 5;

		// formatted output
		std::cout << "% --------------------------- " << std::endl;
		std::cout << "% --------------------------- " << std::endl << std::endl;
		std::cout << "% C1(k,l) = T(k,l) + A(i,j,k)*A(i,j,l) + 5;" << std::endl << std::endl;
		std::cout << "C1=" << C1 << ";" << std::endl << std::endl;


		// C2(k,l,m) = T(k,l,m) + A(i,j,k)*B(j,l,i,m) + 5;
		// Similar Problem as above
		tensor_t C2 = tensor(dynamic_extents<>{na[2],nb[1],nb[3]},value_t(2)) + prod(A,B,perm_t{1,2},perm_t{3,1}) + 5;

		// formatted output
		std::cout << "% --------------------------- " << std::endl;
		std::cout << "% --------------------------- " << std::endl << std::endl;
		std::cout << "%  C2(k,l,m) = T(k,l,m) + A(i,j,k)*B(j,l,i,m) + 5;" << std::endl << std::endl;
		std::cout << "C2=" << C2 << ";" << std::endl << std::endl;


		// C3(k,l,m) = T(k,l,m) + A(i,j,k)*trans(B(j,l,i,m),{2,3,1,4})+ 5;
		// Similar Problem as above
		tensor_t C3 = tensor(dynamic_extents<>{na[2],nb[1],nb[3]},value_t(2)) + prod(A,trans(B,{2,3,1,4}),perm_t{1,2}) + 5;

		// formatted output
		std::cout << "% --------------------------- " << std::endl;
		std::cout << "% --------------------------- " << std::endl << std::endl;
		std::cout << "%  C3(k,l,m) = T(k,l,m) + A(i,j,k)*trans(B(j,l,i,m),{2,3,1,4})+ 5;" << std::endl << std::endl;
		std::cout << "C3=" << C3 << ";" << std::endl << std::endl;

	}
}
