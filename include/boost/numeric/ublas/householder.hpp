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
				value_type sigma = outer_prod(project(x, range(1, n)), project(x, range(1, n)));

				v = vector<M>(x);
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

			}

}}}

#endif