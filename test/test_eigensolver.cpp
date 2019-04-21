// Copyright 2008 Gunter Winkler <guwi17@gmx.de>
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

// switch automatic singular check off
#define BOOST_UBLAS_TYPE_CHECK 0

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/eigen_solver.hpp>
#include <boost/cstdlib.hpp>

#include "common/testhelper.hpp"

#include <iostream>
#include <sstream>

using namespace boost::numeric::ublas;
using std::string;

static const string matrix_IN[] = { "[3,3]((-149.0,-50.0,-154.0),(537.0,180.0,546.0),(-27.0,-9.0,-25.0))\0", 
																		"[3,3]((2.0,3.0,5.0),(2.0,-3.0,7.0),(4.0,1.0,1.0))\0",
																		"[6,6]((7.0, 3.0, 4.0, -11.0, -9.0, -2.0),(-6.0, 4.0, -5.0, 7.0, 1.0, 12.0),(-1.0, -9.0, 2.0, 2.0, 9.0, 1.0),(-8.0, 0.0, -1.0, 5.0, 0.0, 8.0),(-4.0, 3.0, -5.0, 7.0, 2.0, 10.0),(6.0, 1.0, 4.0, -11.0, -7.0, -1.0))\0" 
																		"[4,4]((0.421761282626275,0.655740699156587,0.678735154857774,0.655477890177557),(0.915735525189067,0.0357116785741896,0.757740130578333,0.171186687811562),(0.792207329559554,0.849129305868777,0.743132468124916,0.706046088019609),(0.959492426392903, 0.933993247757551, 0.392227019534168, 0.0318328463774207))\0"};
static const string matrix_EVALR[] = { "[3,3]((1,0.0,0.0),(0.0,2,0.0),(0.0,0.0,3))\0", 
																			 "[3,3]((7.547183,0.0,0.0),(0.0,-3.773591,0.0),(0.0,0.0,-3.773591))\0",
																			 "[6,6]((5.0,0.0,0.0,0.0,0.0,0.0),(0.0,5.0,0.0,0.0,0.0,0.0),(0.0,0.0,1.0,0.0,0.0,0.0),(0.0,0.0,0.0,1.0,0.0,0.0),(0.0,0.0,0.0,0.0,4.0,0.0),(0.0,0.0,0.0,0.0,0.0,3.0))\0" 
																			 "[4,4]((2.44784,0.00000,0.00000,0.00000),(0.00000,-0.56040,0.00000,0.00000),(0.00000,0.00000,-0.56040,0.00000),(0.00000,0.00000,0.00000,-0.09461))\0" };
static const string matrix_EVALC[] = { "[3,3]((0.0,0.0,0.0),(0.0,0.0,0.0),(0.0,0.0,0.0))\0",
																			 "[3,3]((0.0,0.0,0.0),(0.0,1.649236,0.0),(0.0,0.0,-1.649236))\0",
																			 "[6,6]((6.0,0.0,0.0,0.0,0.0,0.0),(0.0,-6.0,0.0,0.0,0.0,0.0),(0.0,0.0,2.0,0.0,0.0,0.0),(0.0,0.0,0.0,-2.0,0.0,0.0),(0.0,0.0,0.0,0.0,0.0,0.0),(0.0,0.0,0.0,0.0,0.0,0.0))\0" 
																			 "[4,4]((0.00000,0.00000,0.00000,0.00000),(0.00000,0.31770,0.00000,0.00000),(0.00000,0.00000,-0.31770,0.00000),(0.00000,0.00000,0.00000,0.00000))\0" };


int main() {

	typedef double TYPE;

	typedef matrix<TYPE> MATRIX;

	MATRIX A;
	MATRIX EVALR;
	MATRIX EVALC;
	MATRIX EVECR;
	MATRIX EVECC;
	MATRIX GT_EVALR;
	MATRIX GT_EVALC;
	MATRIX Zero_Matrix = zero_matrix<TYPE>(3,3);

	int numTestCases = 4;


	for (int i = 0; i < numTestCases; i++){

		std::cout << "Running test case# " << (i + 1) << "\n";

		{
			std::istringstream is(matrix_IN[i]);
			is >> A;
		}

	  {
			std::istringstream is(matrix_EVALR[i]);
			is >> GT_EVALR;
		}

		{
			std::istringstream is(matrix_EVALC[i]);
			is >> GT_EVALC;
		}

		eigen_solver<MATRIX> es(A, EIGVEC);
		matrix<double> evals_r = es.get_real_eigenvalues();
		matrix<double> evals_c = es.get_complex_eigenvalues();
		matrix<double> evecs_r = es.get_real_eigenvectors();
		matrix<double> evecs_c = es.get_complex_eigenvectors();

		assertTrue("Real portion of Eigen Values Match:", compare_on_tolerance(evals_r, GT_EVALR));
		assertTrue("Imag. portion of Eigen Values Match:", compare_on_tolerance(evals_c, GT_EVALC));

		matrix<std::complex<TYPE> > V(3, 3);
		matrix<std::complex<TYPE> > D(3, 3);
		matrix<std::complex<TYPE> > M(3, 3);
		matrix<std::complex<TYPE> > Lambda;

		MATRIX Lambda_Real(3, 3);
		MATRIX Lambda_Complex(3, 3);

		for (int i = 0; i < 3; i++){
			for (int j = 0; j < 3; j++){
				V(i, j) = std::complex<TYPE>(evecs_r(i, j), evecs_c(i, j));
				D(i, j) = std::complex<TYPE>(evals_r(i, j), evals_c(i, j));
				M(i, j) = std::complex<TYPE>(A(i, j), 0.0);
			}
		}
		Lambda = prod(M, V) - prod(V, D);
		for (int i = 0; i < 3; i++){
			for (int j = 0; j < 3; j++){
				Lambda_Real(i, j) = Lambda(i, j).real();
				Lambda_Complex(i, j) = Lambda(i, j).imag();
			}
		}

		assertTrue("Real portion of verifier is zero:", compare_on_tolerance(Lambda_Real, Zero_Matrix));
		assertTrue("Imag. portion of verifier is zero:", compare_on_tolerance(Lambda_Complex, Zero_Matrix));

	}

	if (!_fail_counter) {
		std::cout << "\nEigen Solver regression suite passed.\n";
		return boost::exit_failure;
	}
	else {
		std::cout << "\nEigen Solver regression suite failed.\n";
		return boost::exit_success;
	}
}
