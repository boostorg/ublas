//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_UBLAS_EIGENSOLVER_
#define _BOOST_UBLAS_EIGENSOLVER_

#include <complex>

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
	M eigenvectors_real;
	M eigenvectors_complex;
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
		return eigenvectors_real;
	}

	BOOST_UBLAS_INLINE 
		M& get_complex_eigenvectors() {
		return eigenvectors_complex;
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

		if (norm == value_type(0))
			return;

		for (size_type k = n; k-- != size_type(0);) {

			value_type p = eigenvalues_real(k, k);
			value_type q = eigenvalues_complex(k, k);

			if (q == value_type(0)) {

				value_type lastr, lastw;
				lastr = lastw = value_type(0);
				size_type l = k;

				T(k, k) = value_type(1);

				for (size_type i = k; i-- != size_type(0);) {
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
							for (size_type j = n - k - i; j < n; j++)
								T(k, j) /= t;
						}
					}
				}
			}
			else if (q < value_type(0) && k>0) {

				value_type lastra, lastsa, lastw;
				lastra = lastsa = lastw = value_type(0);
				size_type l = k - 1;

				if ((std::abs)(T(k, k - size_type(1))) > (std::abs)(T(k - size_type(1), k))) {
					T(k - size_type(1), k - size_type(1)) = q / T(k, k - size_type(1));
					T(k - size_type(1), k) = -(T(k, k) - p) / T(k, k - size_type(1));
				}
				else {
					std::complex<value_type> x(value_type(0), -T(k - size_type(1), k));
					std::complex<value_type> y(T(k - size_type(1), k - size_type(1)) - p, q);
					std::complex<value_type> cc = x / y;
					T(k - size_type(1), k - size_type(1)) = cc.real();
					T(k - size_type(1), k) = cc.imag();
				}
				T(k ,k - size_type(1)) = value_type(0);
				T(k, k) = value_type(1);

				for (size_type i = k - size_type(1); i-- != size_type(0);) {
					
					vector<value_type> ra_left = project(row(T, i), range(l, k + size_type(1))); 
					vector<value_type> ra_right = project(column(T, k - size_type(1)), range(l, k + size_type(1)));
					value_type ra = inner_prod(ra_left, ra_right);

					vector<value_type> sa_right = project(column(T, k), range(l, k + size_type(1)));
					value_type sa = inner_prod(ra_left, sa_right);

					value_type w = T(i, i) - p;

					if (eigenvalues_complex(i, i) < value_type(0)) {
						lastw = w;
						lastra = ra;
						lastsa = sa;
					}
					else {
						l = i;

						if (eigenvalues_complex(i, i) == value_type(0)) {
							std::complex<value_type> x(-ra, -sa);
							std::complex<value_type> y(w, q);
							std::complex<value_type> cc = x / y;
							T(i, k - size_type(1)) = cc.real();
							T(i, k) = cc.imag();
						}
						else {
							value_type x = T(i, i + size_type(1));
							value_type y = T(i + size_type(1), i);
							value_type vr = (eigenvalues_real(i, i) - p) * (eigenvalues_real(i, i) - p) + eigenvalues_complex(i, i) * eigenvalues_complex(i, i) - q * q;
							value_type vi = (eigenvalues_real(i, i) - p) * value_type(2) * q;

							if (vr == value_type(0) && vi == value_type(0)) {
								vr = (std::numeric_limits<value_type>::epsilon()) * norm * ((std::abs)(w)+(std::abs)(q)+(std::abs)(x)+(std::abs)(y)+(std::abs)(lastw));
							}

							std::complex<value_type> x1(x*lastra - lastw*ra + q*sa, x*lastsa - lastw*sa - q*ra);
							std::complex<value_type> y1(vr, vi);
							std::complex<value_type> cc = x1 / y1;
							T(i, k - size_type(1)) = cc.real();
							T(i, k) = cc.imag();

							if ((std::abs)(x) > ((std::abs)(lastw)+(std::abs)(q))) {
								T(i + size_type(1), k - size_type(1)) = (-ra - w * T(i, k - size_type(1)) + q * T(i, k)) / x;
								T(i + size_type(1), k) = (-sa - w * T(i, k) - q * T(i, n - size_type(1))) / x;
							}
							else {
								x1 = std::complex<value_type>(-lastra - y*T(i, k - size_type(1)), -lastsa - y*T(i, k));
								y1 = std::complex<value_type>(lastw, q);
								cc = x1 / y1;
								T(i + size_type(1), k - size_type(1)) = cc.real();
								T(i + size_type(1), k) = cc.imag();
							}

						}

					}

				}
				k--;
			}

		}

		for (size_type j = n; j--!=size_type(0);)
		{
			M matrix_left = project(eigenvectors, range(0, n),range(0, j + size_type(1)));
			vector<value_type> vector_right = project(column(T,j), range(0, j + size_type(1)));
			vector<value_type> v  = prod(matrix_left,vector_right);
			column(eigenvectors,j) = v;
		}

		eigenvectors_real = zero_matrix<value_type>(n, n);
		eigenvectors_complex = zero_matrix<value_type>(n, n);

		for (size_type j = 0; j < n; j++) {
			if (j + size_type(1) == n || (eigenvalues_complex(j, j) == value_type(0))){
				vector<value_type> vj = column(eigenvectors, j);
				value_type norm_vj = norm_2(vj);
				vector<value_type> normalized_vj = vj / norm_vj;
				column(eigenvectors_real, j) = normalized_vj;
			}
			else {
				vector<std::complex<value_type> > col_ij(n);
				vector<std::complex<value_type> > col_ijp1(n);
				for (size_type i = 0; i < n; ++i) {
					col_ij(i) = std::complex<value_type>(eigenvectors(i, j), eigenvectors(i, j + 1));
					col_ijp1(i) = std::complex<value_type>(eigenvectors(i, j), -eigenvectors(i, j + 1));
				}

				std::complex<value_type> norm_ij = norm_2(col_ij);
				vector<std::complex<value_type> > normalized_ij = col_ij / norm_ij;
				std::complex<value_type> norm_ijp1 = norm_2(col_ijp1);
				vector<std::complex<value_type> > normalized_ijp1 = col_ijp1 / norm_ijp1;

				for (size_type i = 0; i < n; ++i) {
					eigenvectors_real(i, j) = normalized_ij(i).real();
					eigenvectors_complex(i, j) = normalized_ij(i).imag();

					eigenvectors_real(i, j+1) = normalized_ijp1(i).real();
					eigenvectors_complex(i, j+1) = normalized_ijp1(i).imag();
				}
				j++;
			}
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
