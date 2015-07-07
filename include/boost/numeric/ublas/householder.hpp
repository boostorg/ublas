//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

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

			template<class M>
			typename M::size_type householder(M &x, M &v, typename M::value_type &beta) {

				typedef typename M::size_type size_type;
				typedef typename M::value_type value_type;

				// x must be a vector 
				// output will be a vector in R^n 
				size_type n = x.size();
				value_type sigma =  inner_prod(project(x, range(1, n)), project(x, range(1, n)));
				
				v = M(x); //Ask Mentor how to evoke copy constructor 
				v(size_type(0)) = value_type(1);
				

				if (sigma != value_type()) {
					value_type mu = ((x(size_type(0)))*(x(size_type(0)))) + sigma;
					mu = value_type(boost_numeric_ublas_sqrt(mu));

					if (x(size_type(0)) <= value_type(0)) {
						v(size_type(0)) = x(size_type(0)) - mu;
					}
					else {
						v(size_type(0)) = (-sigma) / (x(size_type(0)) + mu); // Ask mentor is -sigma is valid
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
						beta = value_type(-2); // Ask mentor if this is Valid 
					}
				}
				

				return size_type(0);
			}


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