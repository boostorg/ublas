//
// Rajaditya Mukherjee
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/eigen_solver.hpp>
#include <complex>

int main () {
    using namespace boost::numeric::ublas;
    matrix<double> m(3, 3);
	m <<= 2.0, 3.0, 5.0,
		2.0, -3.0, 7.0,
		4.0, 1.0, 1.0;
	eigen_solver<matrix<double> > es(m,EIGVEC);
	
	matrix<double> evals_r = es.get_real_eigenvalues();
	matrix<double> evals_c = es.get_complex_eigenvalues();
	matrix<double> evecs_r = es.get_real_eigenvectors();
	matrix<double> evecs_c = es.get_complex_eigenvectors();
	
	std::cout << "Eigenvalues (Real Part)\n";
	std::cout << evals_r << std::endl;
	
	std::cout << "Eigenvalues (Imag. Part)\n";
	std::cout << evals_c << std::endl;
	
	std::cout << "Eigenvectors (Real Part)\n";
	std::cout << evecs_r << std::endl;
	
	std::cout << "Eigenvectors (Imag. Part)\n";
	std::cout << evecs_c << std::endl;
	
	std::cout << "Verification\n";
	matrix<std::complex<double> > V(3,3);
	matrix<std::complex<double> > D(3,3);
	matrix<std::complex<double> > A(3,3);
	matrix<std::complex<double> > Lambda;
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			V(i, j) = std::complex<double>(evecs_r(i, j), evecs_c(i, j));
			D(i, j) = std::complex<double>(evals_r(i, j), evals_c(i, j));
			A(i, j) = std::complex<double>(m(i, j), 0.0);
		}
	}
	Lambda = prod(A, V) - prod(V, D);
	std::cout << Lambda << std::endl;
}

