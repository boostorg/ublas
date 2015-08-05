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

static const string matrix_IN = "[3,3]((-149.0,-50.0,-154.0),(537.0,180.0,546.0),(-27.0,-9.0,-25.0))\0";
static const string matrix_EVALR = "[3,3]((1,0.0,0.0),(0.0,2,0.0),(0.0,0.0,3))\0";
static const string matrix_EVALC = "[3,3]((0.0,0.0,0.0),(0.0,0.0,0.0),(0.0,0.0,0.0))\0";

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


	{
		std::istringstream is(matrix_IN);
		is >> A;
	}

	{
		std::istringstream is(matrix_EVALR);
		is >> GT_EVALR;
	}

	{
		std::istringstream is(matrix_EVALC);
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
	
	return (getResults().second > 0) ? boost::exit_failure : boost::exit_success;
}
