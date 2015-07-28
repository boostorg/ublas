//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


#ifndef _BOOST_UBLAS_MATRIXBALANCING_
#define _BOOST_UBLAS_MATRIXBALANCING_

#include <iostream>
#include <boost\numeric\ublas\matrix.hpp>
#include <boost\numeric\ublas\matrix_proxy.hpp>
#include <boost\numeric\ublas\vector.hpp>
#include <boost\numeric\ublas\vector_proxy.hpp>

namespace boost {
	namespace numeric {
		namespace ublas {

			template<class T> 
			void matrix_balance(matrix<T> &m, vector<T> &d) {
				typedef typename matrix<T>::size_type size_type;

				size_type n = m.size1();
				
				T row_norm,col_norm;
				bool converged = false;
				d = scalar_vector<T>(n, T(1));

				while (!converged) {
					T g, f, s;

					converged = true;

					for (size_type i = size_type(0); i < n; i++) {
						row_norm = col_norm = T(0);
						for (size_type j = size_type(0); j < n; j++){
							if (j != i){
								col_norm += (std::abs)(m(j, i));
								row_norm += (std::abs)(m(i, j));
							}
						}

						if (col_norm == T(0) || row_norm == T(0)) {
							continue;
						}

						g = row_norm / T(2);
						f = T(1);
						s = col_norm + row_norm;

						while (col_norm < g) {
							f *= T(2);
							col_norm *= T(4);
						}

						g = row_norm * T(2);

						while (col_norm > g) {
							f /= T(2);
							col_norm /= T(4);
						}

						if ((row_norm + col_norm) < T(0.95)*s*f){
							converged = false;
							
							g = T(1) / f;

							vector<T> v = row(m, i);
							v *= g;
							row(m, i) = v;

							v = column(m, i);
							v *= f;
							column(m, i) = v;

							d(i) *= f;

						}
					}
				}
			}

			template<class T>
			void apply_transformation(matrix<T> &m, vector<T> &d) {
				typedef typename matrix<T>::size_type size_type;
				size_type n = m.size1();
				//Personal note: Ask Mentor is this is right???
				assert(n == d.size());
				T s;
				for (size_type i = size_type(0); i < n; i++){
					s = d(i);
					vector<T> r = row(m, i);
					r *= s;
					row(m, i) = r;
				}				
			}


}}}



#endif