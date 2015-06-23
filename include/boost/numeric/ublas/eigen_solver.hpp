//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_UBLAS_EIGENSOLVER_
#define _BOOST_UBLAS_EIGENSOLVER_

#include <boost\numeric\ublas\hessenberg.hpp>
#include <boost\numeric\ublas\householder.hpp>
#include <boost\numeric\ublas\schur_decomposition.hpp>

namespace boost { namespace numeric { namespace ublas {

//Add some sloppily defined enums here : later on ask mentor how we can use this to 
//Ask mentor about how to "formally" add them to ublas 
typedef enum eig_solver_params {EIGVAL,EIGVEC};

template <class M> 
class eigen_solver {
private:
	M matrix;
	M hessenberg_form;
	M real_schur_form;
	M transform_accumulations;
	bool has_complex;
	bool has_eigenvalues;
	bool has_eigenvectors;
	M eigenvalues;
	M eigenvectors;
	eig_solver_params solver_params;
public:
	//This is the constructor
	BOOST_UBLAS_INLINE 
	explicit eigen_solver(M &m, eig_solver_params params = EIGVAL) :
	matrix(m),has_complex(false),solver_params(params)
	{
		has_eigenvalues = has_eigenvectors = false;
		hessenberg_form = M(m);
		compute(solver_params);
	}

	//The compute function 
	BOOST_UBLAS_INLINE 
	void compute(eig_solver_params params = EIGVAL) {
		if (params != solver_params) {
			solver_params = params;
		}

		//If only eigen values are desired
		if (params == EIGVAL) {
			to_hessenberg<M>(hessenberg_form);
			real_schur_form = M(hessenberg_form);
			schur_decomposition<M>(real_schur_form);
			extract_eigenvalues_from_schur();
			has_eigenvalues = true;
		}
		else {
			to_hessenberg<M>(hessenberg_form, transform_accumulations);
			real_schur_form = M(hessenberg_form);
			schur_decomposition<M>(real_schur_form, transform_accumulations);
			extract_eigenvalues_from_schur();
			eigenvectors = M(transform_accumulations);
			M temp_q_storage = M(real_schur_form);
			block_diag(temp_q_storage, eigenvectors);
			has_eigenvectors = true;
		}


	}

	//This is a tricky function because it will do many things...
	//Right now it doesn't
	BOOST_UBLAS_INLINE
	void extract_eigenvalues_from_schur() {
		eigenvalues = diagonal_adaptor<M>(real_schur_form);
		has_eigenvalues = true;
	}

	//Extraction function 
	BOOST_UBLAS_INLINE 
		M& get_real_eigenvalues() {
		return eigenvalues;
	}

	BOOST_UBLAS_INLINE 
		M& get_real_eigenvectors() {
		return eigenvectors;
	}


};


}}}


#endif
