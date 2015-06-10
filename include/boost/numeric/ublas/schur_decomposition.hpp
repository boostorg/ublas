//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_UBLAS_SCHURDECOMPOSITION_
#define _BOOST_UBLAS_SCHURDECOMPOSITION_

#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/triangular.hpp>

namespace boost {
	namespace numeric {
		namespace ublas {

			template<class M>
			void schur_decomposition(M &h) {

				typedef typename M::size_type size_type;
				typedef typename M::value_type value_type;

				size_type n = h.size1();
				size_type m = n - size_type(1);

				value_type s = h(m, m) + h(n, n);
				value_type t = h(m, m) * h(n, n) - h(m, n) * h(n, m);
				value_type x = h(size_type(0), size_type(0)) * h(size_type(0), size_type(0)) +
					h(size_type(0), size_type(1)) * h(size_type(1), size_type(0)) -
					s * h(size_type(0), size_type(0)) + t;
				value_type y = h(size_type(1), size_type(0)) * (h(size_type(0), size_type(0)) + h(size_type(1), size_type(1)) - s);
				value_type z = h(size_type(1), size_type(0)) * h(size_type(2), size_type(1));

				//Ask mentor if we can assume that size_type will support signed type (otherwise we need a different mechanism 
				// On second thought lets make no such assumptions 
				for (size_type k = size_type(0); k <= n - size_type(3); k++) {
					vector<value_type> tx(3);
					tx(0) = x; tx(1) = y; tx(2) = z;
					vector<value_type> v;
					value_type beta;
					householder<vector<value_type> >(tx, v, beta);

					size_type q = (std::max)(size_type(1), k);

					matrix<value_type> vvt = outer_prod(v, v);
					vvt *= beta;
					size_type n_vvt = vvt.size1();
					matrix<value_type> imvvt = identity_matrix<value_type>(n_vvt) - vvt;

					matrix<value_type> t1 = prod(imvvt, project(h, range(k,k + size_type(3)), range(q - size_type(1),n - size_type(1))));
					project(h, range(k,k + size_type(3)), range(q - size_type(1), n - size_type(1))).assign(t1);

					size_type r = (std::min)(k + size_type(4), n);

					matrix<value_type> t2 = prod(project(h, range(0,r), range(k, k + size_type(3))), imvvt);
					project(h, range(0,r), range(k, k + size_type(3))).assign(t2);

					x = h(k + size_type(1), k);
					y = h(k + size_type(2), k);

					if (k < (n - size_type(3))) {
						z = h(k + size_type(3), k);
					}

				}

				vector<value_type> tx(2);
				tx(0) = x; tx(1) = y;
				vector<value_type> v;
				value_type beta;
				householder<vector<value_type> >(tx, v, beta);

				matrix<value_type> vvt = outer_prod(v, v);
				vvt *= beta;
				size_type n_vvt = vvt.size1();
				matrix<value_type> imvvt = identity_matrix<value_type>(n_vvt) -vvt;

				matrix<value_type> t1 = prod(imvvt, project(h, range(n - size_type(2),n), range(n - size_type(3), n)));
				project(h, range(n - size_type(2) ,n), range(n - size_type(3), n)).assign(t1);
				
				matrix<value_type> t2 = prod(project(h, range(0,n), range(n - size_type(2), n)),imvvt);
				project(h, range(0,n), range(n - size_type(2), n)).assign(t2);


			}
		
		
		
		
}}}


#endif
