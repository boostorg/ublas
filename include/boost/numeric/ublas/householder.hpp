// Rajaditya Mukherjee
// 
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

/// \file householder.hpp Definition for the methods performing Householder and Givens Rotation

#ifndef _BOOST_UBLAS_HOUSEHOLDER_
#define _BOOST_UBLAS_HOUSEHOLDER_

#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/triangular.hpp>

namespace boost {
	namespace numeric {
		namespace ublas {

			/// \brief Performs Householder reflections on the vector x.
		  /// Given a vector \c x, returns a vector \c v and a scalar \c beta 
			/// which is used to introduce zeros in the vector. 
			/// More specifically we have (I - beta * v & v^T)x = ||x||e_1 where e_1 is a
			/// unit vector. 
			/// \param x input vector type (like vector<double>)
			/// \param v output vector type (like vector<double>)
			/// \param beta scalar output (same as M::value_type) containing the multiplication factor
			template<class M>
			void householder(M &x, M &v, typename M::value_type &beta) {

				typedef typename M::size_type size_type;
				typedef typename M::value_type value_type;

				// x must be a vector 
				// output will be a vector in R^n 
				size_type n = x.size();
				value_type sigma =  inner_prod(project(x, range(1, n)), project(x, range(1, n)));
				
				v = M(x);
				v(size_type(0)) = value_type(1);
				

				if (sigma != value_type()) {
					value_type mu = ((x(size_type(0)))*(x(size_type(0)))) + sigma;
					mu = value_type(boost_numeric_ublas_sqrt(mu));

					if (x(size_type(0)) <= value_type(0)) {
						v(size_type(0)) = x(size_type(0)) - mu;
					}
					else {
						v(size_type(0)) = (-sigma) / (x(size_type(0)) + mu); 
					}
					beta = (value_type(2) * v(size_type(0)) * v(size_type(0))) / (v(size_type(0)) * v(size_type(0)) + sigma);
					value_type inv_v_0 = value_type(1) / v(size_type(0));
					v *= inv_v_0;
				}
				else {
					if (x(size_type(0)) >= value_type(0)) {
						beta = value_type(0);
					}
					else {
						beta = value_type(-2); 
					}
				}
			}


			/// \brief Determines the Givens Rotation Coefficient for two scalars [a,b]. 
			/// Given a vector \c [a,b], returns two scalars \c c and \c d such that the 2x2 rotation matrix formed by the said values of
			/// cosine and sine will rotate the vector [a,b] to form [r,0].
			/// \param a input scalar type (like double)
			/// \param b input scalar type (like double)	
			/// \param c output scalar type (like double) containing cosine for Givens Rotation
			/// \param s output scalar type (like double) containing sine for Givens Rotation

			template<class M>
			void givens_rotation(M &a, M &b, M&c, M&s) {
				if (b == M(0)) {
					c = M(1);
					s = M(0);
				}
				else {
					if ((std::abs)(b) > (std::abs)(a)) {
						M tau = -a / b;
						M s_temp = M(1.0) + (tau*tau);
						s_temp = (std::sqrt)(s_temp);

						s = M(1.0) / s_temp;
						c = s * tau;
					}
					else {
						M tau = -b / a;
						M c_temp = M(1.0) + (tau*tau);
						c_temp = (std::sqrt)(c_temp);

						c = M(1.0) / c_temp;
						s = c * tau;
					}
				}
			}


}}}

#endif