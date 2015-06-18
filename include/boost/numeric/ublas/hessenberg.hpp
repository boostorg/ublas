//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_UBLAS_HESSENBERG_
#define _BOOST_UBLAS_HESSENBERG_

#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <iostream>

namespace boost { namespace numeric { namespace ublas {

	template<class M>
	typename M::size_type to_hessenberg(M &m) {

		typedef typename M::size_type size_type;
		typedef typename M::value_type value_type;

		//At this point, we should probably check if the number of rows and columns of M are the same or not 
		size_type num_rows = m.size1();
		size_type num_cols = m.size2();
		BOOST_UBLAS_CHECK(num_rows != num_cols, singular()); //Throw some kind of assertion error saying that eigen solver works only for sqaure matrices

		size_type n = num_rows;
		for (size_type i = 0; i < n-2; ++i) {
			vector<value_type> x = project(column(m,i), range(i + 1, n));	
			vector<value_type> v;
			value_type beta;
			householder(x, v, beta);
			
			matrix<value_type> vvt = outer_prod(v, v);
			vvt *= beta;
			size_type n_vvt = vvt.size1();
			matrix<value_type> imvvt = identity_matrix<value_type>(n_vvt) - vvt;

			matrix<value_type> t1 = prod(imvvt, project(m, range(i + 1, n), range(i, n)));
			project(m, range(i + 1, n), range(i, n)).assign(t1);
			matrix<value_type> t2 = prod(project(m, range(0, n), range(i + 1, n)), imvvt);
			project(m, range(0, n), range(i + 1, n)).assign(t2);

			//@TODO: Ask mentor why uncommented lines are more accurate than commented lines... 
			//project(m, range(i + 1, n), range(i, n)).assign(prod(imvvt, project(m, range(i + 1, n), range(i, n))));
			//project(m, range(0, n), range(i+1, n)).assign(prod(project(m, range(0, n), range(i+1, n)),imvvt));

		}

	}

	template<class M>
	typename M::size_type to_hessenberg(M &m, M &u0) {

		typedef typename M::size_type size_type;
		typedef typename M::value_type value_type;

		M m_copy = M(m);

		//At this point, we should probably check if the number of rows and columns of M are the same or not 
		size_type num_rows = m.size1();
		size_type num_cols = m.size2();
		BOOST_UBLAS_CHECK(num_rows != num_cols, singular()); //Throw some kind of assertion error saying that eigen solver works only for sqaure matrices

		size_type n = num_rows;
		u0 = identity_matrix<value_type>(n);
		for (size_type i = 0; i < n - 2; ++i) {
			vector<value_type> x = project(column(m, i), range(i + 1, n));
			vector<value_type> v;
			value_type beta;
			householder(x, v, beta);

			//std::cout << i << " " << x << " " << v << "\n";

			matrix<value_type> vvt = outer_prod(v, v);
			vvt *= beta;
			size_type n_vvt = vvt.size1();
			matrix<value_type> imvvt = identity_matrix<value_type>(n_vvt) -vvt;

			matrix<value_type> pk = identity_matrix<value_type>(n);
			project(pk, range(i+1, n), range(i+1, n)).assign(imvvt);

			matrix<value_type> t0 = prod(u0, pk);
			u0 = t0;

			matrix<value_type> t1 = prod(imvvt, project(m, range(i + 1, n), range(i, n)));
			project(m, range(i + 1, n), range(i, n)).assign(t1);
			matrix<value_type> t2 = prod(project(m, range(0, n), range(i + 1, n)), imvvt);
			project(m, range(0, n), range(i + 1, n)).assign(t2);

			//@TODO: Ask mentor why uncommented lines are more accurate than commented lines... 
			//project(m, range(i + 1, n), range(i, n)).assign(prod(imvvt, project(m, range(i + 1, n), range(i, n))));
			//project(m, range(0, n), range(i+1, n)).assign(prod(project(m, range(0, n), range(i+1, n)),imvvt));

		}
	}

}}}


#endif