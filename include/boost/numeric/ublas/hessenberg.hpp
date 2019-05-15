// Rajaditya Mukherjee
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

/// \file hessenberg.hpp Definition for the methods performing Hessenberg Transformation

#ifndef _BOOST_UBLAS_HESSENBERG_
#define _BOOST_UBLAS_HESSENBERG_

#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/householder.hpp>

#include <iostream>

namespace boost { namespace numeric { namespace ublas {

    /// \brief Performs Hessenberg for Matrix \c m without accumulating transformations.
    /// Replaces the input matrix by its Hessenberg Form which is a quasi-upper triangular matrix
    /// preserved by QR Algorithms. It is easier and faster for the QR Algorithm to work on the Hessenberg form instead of the dense form.
    /// We use a series of householder reflections to generate the Hessenberg Forms.
    /// We use this form when we need only the eigen-values.
    /// \param m matrix type (like matrix<double>)
    template<class M>
    void to_hessenberg(M &m) {

        typedef typename M::size_type size_type;
        typedef typename M::value_type value_type;

        //At this point, we should probably check if the number of rows and columns of M are the same or not
        size_type num_rows = m.size1();
        size_type num_cols = m.size2();
        BOOST_UBLAS_CHECK(num_rows == num_cols, singular()); //Throw some kind of assertion error saying that eigen solver works only for sqaure matrices

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
        }

    }

    /// \brief Performs Hessenberg for Matrix \c m with accumulating transformations.
    /// Replaces the input matrix by its Hessenberg Form which is a quasi-upper triangular matrix
    /// preserved by QR Algorithms. It is easier and faster for the QR Algorithm to work on the Hessenberg form instead of the dense form.
    /// We use a series of householder reflections to generate the Hessenberg Forms. In this format, we also accumulate the transformations which is needed
    /// to calculate the eigen-vectors. We use this form when we need both the eigen-values as well as eigen-vectors.
    /// \param m matrix type (like matrix<double>)
    /// \param u0 output with the transformations accumulated. (same type as M)
    template<class M>
    void to_hessenberg(M &m, M &u0) {

        typedef typename M::size_type size_type;
        typedef typename M::value_type value_type;

        M m_copy = M(m);

        //At this point, we should probably check if the number of rows and columns of M are the same or not
        size_type num_rows = m.size1();
        size_type num_cols = m.size2();
        BOOST_UBLAS_CHECK(num_rows == num_cols, singular()); //Throw some kind of assertion error saying that eigen solver works only for sqaure matrices

        size_type n = num_rows;
        u0 = identity_matrix<value_type>(n);
        for (size_type i = 0; i < n - 2; ++i) {
            vector<value_type> x = project(column(m, i), range(i + 1, n));
            vector<value_type> v;
            value_type beta;
            householder(x, v, beta);

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

        }
    }

}}}


#endif
