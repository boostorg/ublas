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
	bool has_complex_part;
	bool has_eigenvalues;
	bool has_eigenvectors;
	M eigenvalues_real;
	M eigenvalues_complex;
	M eigenvectors;
	eig_solver_params solver_params;
public:
	//This is the constructor
	BOOST_UBLAS_INLINE 
	explicit eigen_solver(M &m, eig_solver_params params = EIGVAL) :
	matrix(m), has_complex_part(false), solver_params(params)
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
		typedef typename M::size_type size_type;
		typedef typename M::value_type value_type;

		size_type n = real_schur_form.size1();
		size_type i = size_type(0);

		eigenvalues_real = zero_matrix<value_type>(n,n);
		eigenvalues_complex = zero_matrix<value_type>(n, n);

		while (i < n) {
			if ((i == n - size_type(1)) || (real_schur_form(i + 1, i) == value_type(0))) {
				eigenvalues_real(i, i) = real_schur_form(i, i);
				i += size_type(1);
			}
			else {
				value_type p = value_type(0.5) * (real_schur_form(i, i) - real_schur_form(i + size_type(1), i + size_type(1)));
				value_type z = (std::sqrt)((std::abs)(p * p + real_schur_form(i + size_type(1), i) * real_schur_form(i, i + size_type(1))));
				eigenvalues_real(i, i) = real_schur_form(i + size_type(1), i + size_type(1)) + p;
				eigenvalues_complex(i, i) = z;
				eigenvalues_real(i + size_type(1), i + size_type(1)) = real_schur_form(i + size_type(1), i + size_type(1)) + p;
				eigenvalues_complex(i + size_type(1), i + size_type(1)) = -z;
				i += size_type(2);
				has_complex_part = true;
			}
		}

		has_eigenvalues = true;
	}

	//Extraction function 
	BOOST_UBLAS_INLINE 
		M& get_real_eigenvalues() {
		return eigenvalues_real;
	}

	BOOST_UBLAS_INLINE
		M& get_complex_eigenvalues() {
		return eigenvalues_complex;
	}

	BOOST_UBLAS_INLINE 
		M& get_real_eigenvectors() {
		return eigenvectors;
	}


		//THis is my functions - just kept for debugging will be deleted (or maybe not)
		BOOST_UBLAS_INLINE
			M& get_real_schur_form() {
			return real_schur_form;
		}

		BOOST_UBLAS_INLINE
			M& get_hessenberg_form() {
			return hessenberg_form;
		}


};


}}}


#endif
