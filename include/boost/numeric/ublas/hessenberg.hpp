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
		vector<value_type> temp(n);
		for (size_type i = 0; i < n; ++i) {
			//Insert householder modules here
		}

	}


}}}


#endif