//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_UBLAS_EIGENSOLVER_
#define _BOOST_UBLAS_EIGENSOLVER_

#include <boost\numeric\ublas\matrix.hpp>
#include <boost\numeric\ublas\matrix_proxy.hpp>
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
			has_eigenvectors = false;
		}
		else {
			to_hessenberg<M>(hessenberg_form, transform_accumulations);
			real_schur_form = M(hessenberg_form);
			schur_decomposition<M>(real_schur_form, transform_accumulations);
			extract_eigenvalues_from_schur();
			extract_eigenvectors();
			has_eigenvalues = true;
			has_eigenvectors = true;
		}


	}

	BOOST_UBLAS_INLINE
	void extract_eigenvalues_from_schur() {
		typedef typename M::size_type size_type;
		typedef typename M::value_type value_type;

		size_type n = real_schur_form.size1();
		size_type i = size_type(0);

		eigenvalues_real = zero_matrix<value_type>(n,n);
		eigenvalues_complex = zero_matrix<value_type>(n, n);

		while (i < n) {
			if ((i == n - size_type(1)) || (real_schur_form(i + 1, i) == value_type(0)) || ((std::abs)(real_schur_form(i + 1, i)) <= 1.0e-5)) {
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


	BOOST_UBLAS_INLINE 
		void extract_eigenvectors() {

		typedef typename M::size_type size_type;
		typedef typename M::value_type value_type;

		M T(real_schur_form);
		eigenvectors = M(transform_accumulations);

		size_type n = eigenvalues_real.size1();

		value_type norm = value_type(0);
		for (size_type j = size_type(0); j < n; j++) {
			size_type idx = (j==size_type(0))?size_type(0):(j - size_type(1));
			vector<value_type> v = project(row(T,j), range(idx, n));
			norm += norm_1(v);
		}


		for (size_type k = n; k--!=size_type(0); ) {
			
			value_type p = eigenvalues_real(k, k);
			value_type q = eigenvalues_complex(k, k);

			value_type lastr, lastw;
			lastr = lastw = value_type(0);
			size_type l = k;

			T(k, k) = value_type(1);

				for (size_type i = k; i--!=size_type(0);) {
					value_type w = T(i, i) - p;
					vector<value_type> r_left = project(row(T, i), range(l, k + size_type(1)));
					vector<value_type> r_right = project(column(T, k), range(l, k + size_type(1)));
					value_type r = inner_prod(r_left, r_right);
					if (eigenvalues_complex(i, i) < value_type(0)) {
						lastw = w;
						lastr = r;
					}
					else {
						l = i;
						if (eigenvalues_complex(i, i) == value_type(0)) {
							if (w != value_type(0)) {
								T(i, k) = -r / w;
							}
							else {
								T(i, k) = -r / (norm * std::numeric_limits<value_type>::epsilon());
							}
						}
						else {
							value_type x = T(i, i + size_type(1));
							value_type y = T(i + size_type(1), i);
							value_type denom = (eigenvalues_real(i, i) - p)*(eigenvalues_real(i, i) - p) + (eigenvalues_complex(i, i))*(eigenvalues_complex(i, i));
							value_type t = (x * lastr - lastw * r) / denom;
							T(i, k) = t;
							if ((std::abs)(x) > (std::abs)(lastw)) {
								T(i + size_type(1), k) = (-r - w * t) / x;
							}
							else {
								T(i + size_type(1), k) = (-lastr - y * t) / lastw;
							}
						}

						// Overflow control
						value_type t = (std::abs)(T(i, k));
						if ((std::numeric_limits<value_type>::epsilon() * t) * t > value_type(1)) {
							for (size_type j = n - k - i; j < n;j++)
								T(k,j) /= t;
						}
					}
				}
		}

		for (size_type j = n; j--!=size_type(0);)
		{
			M matrix_left = project(eigenvectors, range(0, n),range(0, j + size_type(1)));
			vector<value_type> vector_right = project(column(T,j), range(0, j + size_type(1)));
			vector<value_type> v  = prod(matrix_left,vector_right);
			column(eigenvectors,j) = v;
		}

		for (size_type i = 0; i < n; i++) {
			vector<value_type> vi = column(eigenvectors, i);
			value_type norm_vi = norm_2(vi);
			vector<value_type> normalized_vi = vi / norm_vi;
			column(eigenvectors, i) = normalized_vi;
		}
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

		BOOST_UBLAS_INLINE
			M& get_transform_accumulations() {
			return transform_accumulations;
		}


};


}}}


#endif
