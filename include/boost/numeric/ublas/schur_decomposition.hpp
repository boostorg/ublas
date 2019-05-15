// Rajaditya Mukherjee
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


/// \file schur_decomposition.hpp Contains methods for real schur decomposition

#ifndef _BOOST_UBLAS_SCHURDECOMPOSITION_
#define _BOOST_UBLAS_SCHURDECOMPOSITION_

#define EPSA 1.0e-20

#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <iostream>
#include <limits>


namespace boost {
    namespace numeric {
        namespace ublas {

            /*! @fn schur_decomposition(M &h)
            * @brief Performs Real Schur Decomposition for Matrix \c m (Version without EigenVectors).
            * Replaces the input matrix which is assumed to be in Hessenberg Form by the
            real schur form of the matrix. This uses the Francis Double Shift QR Algorithm to compute the real Schur Form.
            * The diagonals of the schur form are either 1x1 blocks (real eigenvalues) or 2x2 blocks (complex eigenvalues).
            * @tparam[in] m matrix type (like matrix<double>) - input hessenberg form output - schur form
            */
            template<class M>
            void schur_decomposition(M &h) {

                typedef typename M::size_type size_type;
                typedef typename M::value_type value_type;

                size_type n = h.size1();
                size_type p = n;
                while (p > size_type(2)) {
                    size_type q = p - size_type(1);

                    value_type s = h(q - size_type(1), q - size_type(1)) + h(p - size_type(1), p - size_type(1));
                    value_type t = h(q - size_type(1), q - size_type(1)) * h(p - size_type(1), p - size_type(1)) - h(q - size_type(1), p - size_type(1)) * h(p - size_type(1), q - size_type(1));
                    value_type x = h(size_type(0), size_type(0)) * h(size_type(0), size_type(0)) +
                        h(size_type(0), size_type(1)) * h(size_type(1), size_type(0)) -
                        s * h(size_type(0), size_type(0)) + t;
                    value_type y = h(size_type(1), size_type(0)) * (h(size_type(0), size_type(0)) + h(size_type(1), size_type(1)) - s);
                    value_type z = h(size_type(1), size_type(0)) * h(size_type(2), size_type(1));

                    //Ask mentor if we can assume that size_type will support signed type (otherwise we need a different mechanism
                    // On second thought lets make no such assumptions
                    for (size_type k = size_type(0); k <= p - size_type(3); k++) {
                        vector<value_type> tx(3);
                        tx(0) = x; tx(1) = y; tx(2) = z;
                        vector<value_type> v;
                        value_type beta;
                        householder<vector<value_type> >(tx, v, beta);

                        size_type r = (std::max)(size_type(1), k);

                        matrix<value_type> vvt = outer_prod(v, v);
                        vvt *= beta;
                        size_type n_vvt = vvt.size1();
                        matrix<value_type> imvvt = identity_matrix<value_type>(n_vvt) -vvt;

                        matrix<value_type> t1 = prod(imvvt, project(h, range(k, k + size_type(3)), range(r - size_type(1), n)));
                        project(h, range(k, k + size_type(3)), range(r - size_type(1), n)).assign(t1);

                        r = (std::min)(k + size_type(4), p);

                        matrix<value_type> t2 = prod(project(h, range(0, r), range(k, k + size_type(3))), imvvt);
                        project(h, range(0, r), range(k, k + size_type(3))).assign(t2);

                        x = h(k + size_type(1), k);
                        y = h(k + size_type(2), k);

                        if (k < (p - size_type(3))) {
                            z = h(k + size_type(3), k);
                        }

                    }

                    vector<value_type> tx(2);
                    tx(0) = x; tx(1) = y;

                    value_type scale = (std::abs)(tx(0)) + (std::abs)(tx(1));
                    if (scale != value_type(0)) {
                        tx(0) /= scale;
                        tx(1) /= scale;
                    }

                    value_type cg, sg;
                    givens_rotation<value_type>(tx(0), tx(1), cg, sg);

                    //Apply rotation to matrices as needed
                    matrix<value_type> t1 = project(h, range(q - size_type(1), p), range(p - size_type(3), n));
                    size_type t1_cols = t1.size2();
                    size_type t1_rows = t1.size1();
                    for (size_type j = size_type(0); j < t1_cols; j++) {
                        value_type tau1 = t1(0, j);
                        value_type tau2 = t1(t1_rows - 1, j);
                        t1(0, j) = cg*tau1 - sg*tau2;
                        t1(t1_rows - 1, j) = sg*tau1 + cg*tau2;
                    }
                    //std::cout << t1 << "\n";
                    project(h, range(q - size_type(1), p), range(p - size_type(3), n)).assign(t1);

                    matrix<value_type> t2 = project(h, range(0, p), range(p - size_type(2), p));
                    size_type t2_cols = t2.size2();
                    size_type t2_rows = t2.size1();
                    for (size_type j = size_type(0); j < t2_rows; j++) {
                        value_type tau1 = t2(j, 0);
                        value_type tau2 = t2(j, t2_cols - 1);
                        t2(j, 0) = cg*tau1 - sg*tau2;
                        t2(j, t2_cols - 1) = sg*tau1 + cg*tau2;
                    }
                    project(h, range(0, p), range(p - size_type(2), p)).assign(t2);

                    if ((std::abs)(h(p - size_type(1), q - size_type(1))) < (std::numeric_limits<value_type>::epsilon())*((std::abs)(h(p - size_type(1), p - size_type(1))) + (std::abs)(h(q - size_type(1), q - size_type(1))))) {
                        h(p - size_type(1), q - size_type(1)) = value_type(0);
                        p = p - size_type(1);
                        q = p - size_type(1);
                    }
                    else if ((std::abs)(h(p - size_type(2), q - size_type(2))) < (std::numeric_limits<value_type>::epsilon())*((std::abs)(h(q - size_type(2), q - size_type(2))) + (std::abs)(h(q - size_type(1), q - size_type(1))))) {
                        h(p - size_type(2), q - size_type(2)) = value_type(0);
                        p = p - size_type(2);
                        q = p - size_type(1);
                    }

                }
            }

            /// \brief Performs Real Schur Decomposition for Matrix \c m (Version with EigenVectors).
            /// Replaces the input matrix which is assumed to be in Hessenberg Form by the
            /// real schur form of the matrix. This uses the Francis Double Shift QR Algorithm to compute the real Schur Form.
            /// The diagonals of the schur form are either 1x1 blocks (real eigenvalues) or 2x2 blocks (complex eigenvalues).
            /// \param m matrix type (like matrix<double>) - input hessenberg form output - schur form
            /// \param qv matrix type (like matrix<double>) - input accumulated transforms
            template<class M>
            void schur_decomposition(M &h, M &qv) {

                typedef typename M::size_type size_type;
                typedef typename M::value_type value_type;

                size_type n = h.size1();
                size_type p = n;
                while (p > size_type(2)) {
                    size_type q = p - size_type(1);

                    value_type s = h(q - size_type(1), q - size_type(1)) + h(p - size_type(1), p - size_type(1));
                    value_type t = h(q - size_type(1), q - size_type(1)) * h(p - size_type(1), p - size_type(1)) - h(q - size_type(1), p - size_type(1)) * h(p - size_type(1), q - size_type(1));
                    value_type x = h(size_type(0), size_type(0)) * h(size_type(0), size_type(0)) +
                        h(size_type(0), size_type(1)) * h(size_type(1), size_type(0)) -
                        s * h(size_type(0), size_type(0)) + t;
                    value_type y = h(size_type(1), size_type(0)) * (h(size_type(0), size_type(0)) + h(size_type(1), size_type(1)) - s);
                    value_type z = h(size_type(1), size_type(0)) * h(size_type(2), size_type(1));

                    //Ask mentor if we can assume that size_type will support signed type (otherwise we need a different mechanism
                    // On second thought lets make no such assumptions
                    for (size_type k = size_type(0); k <= p - size_type(3); k++) {
                        vector<value_type> tx(3);
                        tx(0) = x; tx(1) = y; tx(2) = z;

                        //Add some scaling factor here
                        value_type scale = (std::abs)(tx(0)) + (std::abs)(tx(1)) + (std::abs)(tx(2));
                        if (scale != value_type(0)) {
                            tx(0) /= scale;
                            tx(1) /= scale;
                            tx(2) /= scale;
                        }

                        vector<value_type> v;
                        value_type beta;
                        householder<vector<value_type> >(tx, v, beta);

                        size_type r = (std::max)(size_type(1), k);

                        matrix<value_type> vvt = outer_prod(v, v);
                        vvt *= beta;
                        size_type n_vvt = vvt.size1();
                        matrix<value_type> imvvt = identity_matrix<value_type>(n_vvt) -vvt;

                        matrix<value_type> t1 = prod(imvvt, project(h, range(k, k + size_type(3)), range(r - size_type(1), n)));
                        project(h, range(k, k + size_type(3)), range(r - size_type(1), n)).assign(t1);

                        r = (std::min)(k + size_type(4), p);

                        matrix<value_type> t2 = prod(project(h, range(0, r), range(k, k + size_type(3))), imvvt);
                        project(h, range(0, r), range(k, k + size_type(3))).assign(t2);

                        matrix<value_type> t3 = prod(project(qv, range(0, n), range(k, k + size_type(3))), imvvt);
                        project(qv, range(0, n), range(k, k + size_type(3))).assign(t3);

                        x = h(k + size_type(1), k);
                        y = h(k + size_type(2), k);

                        if (k < (p - size_type(3))) {
                            z = h(k + size_type(3), k);
                        }

                    }

                    vector<value_type> tx(2);
                    tx(0) = x; tx(1) = y;

                    value_type scale = (std::abs)(tx(0)) + (std::abs)(tx(1)) ;
                    if (scale != value_type(0)) {
                        tx(0) /= scale;
                        tx(1) /= scale;
                    }

                    value_type cg, sg;
                    givens_rotation<value_type>(tx(0), tx(1), cg, sg);

                    //Apply rotation to matrices as needed
                    matrix<value_type> t1 = project(h, range(q - size_type(1), p), range(p - size_type(3), n));
                    size_type t1_cols = t1.size2();
                    size_type t1_rows = t1.size1();
                    for (size_type j = size_type(0); j < t1_cols; j++) {
                        value_type tau1 = t1(0, j);
                        value_type tau2 = t1(t1_rows - 1, j);
                        t1(0, j) = cg*tau1 - sg*tau2;
                        t1(t1_rows - 1, j) = sg*tau1 + cg*tau2;
                    }
                    project(h, range(q - size_type(1), p), range(p - size_type(3), n)).assign(t1);

                    matrix<value_type> t2 = project(h, range(0, p), range(p - size_type(2), p));
                    size_type t2_cols = t2.size2();
                    size_type t2_rows = t2.size1();
                    for (size_type j = size_type(0); j < t2_rows; j++) {
                        value_type tau1 = t2(j, 0);
                        value_type tau2 = t2(j,t2_cols - 1);
                        t2(j, 0) = cg*tau1 - sg*tau2;
                        t2(j, t2_cols - 1) = sg*tau1 + cg*tau2;
                    }
                    project(h, range(0, p), range(p - size_type(2), p)).assign(t2);

                    matrix<value_type> t3 = project(qv, range(0, n), range(p - size_type(2), p));
                    size_type t3_cols = t3.size2();
                    size_type t3_rows = t3.size1();
                    for (size_type j = size_type(0); j < t3_rows; j++) {
                        value_type tau1 = t3(j, 0);
                        value_type tau2 = t3(j, t3_cols - 1);
                        t3(j, 0) = cg*tau1 - sg*tau2;
                        t3(j, t3_cols - 1) = sg*tau1 + cg*tau2;
                    }
                    project(qv, range(0, n), range(p - size_type(2), p)).assign(t3);

                    //Pollution Cleanup
                    /*for (size_type ci = 0; ci < p; ++ci)
                    {
                        for (size_type ri = p; ri < n; ri++) {
                            h(ri, ci) = value_type(0);
                        }
                    }
                    for (size_type ci = p; ci < n - q; ci++) {
                        for (size_type ri = n - q; ri < n; ri++){
                            h(ri, ci) = value_type(0);
                        }
                    }*/

                    if ((std::abs)(h(p - size_type(1), q - size_type(1))) < (std::numeric_limits<value_type>::epsilon())*((std::abs)(h(p - size_type(1), p - size_type(1))) + (std::abs)(h(q - size_type(1), q - size_type(1))))) {
                        h(p - size_type(1), q - size_type(1)) = value_type(0);
                        p = p - size_type(1);
                        q = p - size_type(1);
                    }
                    else if ((std::abs)(h(p - size_type(2), q - size_type(2))) < (std::numeric_limits<value_type>::epsilon())*((std::abs)(h(q - size_type(2), q - size_type(2))) + (std::abs)(h(q - size_type(1), q - size_type(1))))) {
                        h(p - size_type(2), q - size_type(2)) = value_type(0);
                        p = p - size_type(2);
                        q = p - size_type(1);
                    }
                }
            }

            template<class M>
            void find_small_diag_entry(M &h, typename M::size_type end, typename M::value_type l1_norm, typename M::size_type &small_index)
            {
                typedef typename M::size_type size_type;
                typedef typename M::value_type value_type;

                size_type k = end;
                while (k > size_type(0))
                {
                    value_type s = (std::abs)(h(k - size_type(1), k - size_type(1))) + (std::abs)(h(k, k));
                    if (s == value_type(0))
                        s = l1_norm;
                    if ((std::abs)(h(k, k - size_type(1))) < (std::numeric_limits<value_type>::epsilon() * s))
                        break;
                    --k;
                }
                small_index = k;
            }

            template<class M>
            void row_split(M &h, M &qv, typename M::size_type end, typename M::value_type exceptional_shift_sum)
            {
                typedef typename M::size_type size_type;
                typedef typename M::value_type value_type;

                size_type n = h.size1();

                value_type p = value_type(0.5) * (h(end - size_type(1), end - size_type(1)) - h(end, end));
                value_type q = p * p + h(end, end - size_type(1)) *h(end - size_type(1), end);
                h(end, end) += exceptional_shift_sum;
                h(end - size_type(1), end - size_type(1)) += exceptional_shift_sum;

                //Two real eigenvalues are present so separate them
                if (q >= value_type(0))
                {
                    value_type z = (std::sqrt)((std::abs)(q));
                    vector<value_type> tx(2);
                    if (p >= value_type(0))
                    {
                        tx(0) = p + z;
                    }
                    else
                    {
                        tx(0) = p - z;
                    }
                    tx(1) = h(end, end - size_type(1));
                    value_type cg, sg;
                    givens_rotation<value_type>(tx(0), tx(1), cg, sg);

                    //Apply rotation to matrices as needed
                    matrix<value_type> t1 = project(h, range(end - size_type(1), end + size_type(1)), range(end - size_type(1), n));
                    size_type t1_cols = t1.size2();
                    size_type t1_rows = t1.size1();
                    for (size_type j = size_type(0); j < t1_cols; ++j) {
                        value_type tau1 = t1(0, j);
                        value_type tau2 = t1(t1_rows - 1, j);
                        t1(0, j) = cg*tau1 - sg*tau2;
                        t1(t1_rows - 1, j) = sg*tau1 + cg*tau2;
                    }
                    project(h, range(end - size_type(1), end + size_type(1)), range(end - size_type(1), n)).assign(t1);

                    matrix<value_type> t2 = project(h, range(0, end + size_type(1)), range(end - size_type(1), end + size_type(1)));
                    size_type t2_cols = t2.size2();
                    size_type t2_rows = t2.size1();
                    for (size_type j = size_type(0); j < t2_rows; ++j) {
                        value_type tau1 = t2(j, 0);
                        value_type tau2 = t2(j, t2_cols - 1);
                        t2(j, 0) = cg*tau1 - sg*tau2;
                        t2(j, t2_cols - 1) = sg*tau1 + cg*tau2;
                    }
                    project(h, range(0, end + size_type(1)), range(end - size_type(1), end + size_type(1))).assign(t2);

                    matrix<value_type> t3 = project(qv, range(0, n), range(end - size_type(1), end + size_type(1)));
                    size_type t3_cols = t3.size2();
                    size_type t3_rows = t3.size1();
                    for (size_type j = size_type(0); j < t3_rows; ++j) {
                        value_type tau1 = t3(j, 0);
                        value_type tau2 = t3(j, t3_cols - 1);
                        t3(j, 0) = cg*tau1 - sg*tau2;
                        t3(j, t3_cols - 1) = sg*tau1 + cg*tau2;
                    }
                    project(qv, range(0, n), range(end - size_type(1), end + size_type(1))).assign(t3);
                }

                if (end > size_type(1))
                    h(end - size_type(1), end - size_type(2)) = value_type(0);
            }

            template<class M>
            void infer_shifts(M &h, typename M::size_type end, typename M::size_type iter_nos, typename M::value_type exceptional_shift_sum, vector<typename M::value_type> &shift_vector)
            {
                typedef typename M::size_type size_type;
                typedef typename M::value_type value_type;

                shift_vector = vector<value_type>(3);

                shift_vector(size_type(0)) = h(end, end);
                shift_vector(size_type(1)) = h(end - size_type(1), end - size_type(1));
                shift_vector(size_type(2)) = h(end, end - size_type(1)) * h(end - size_type(1), end);

                //Original Shift
                if (iter_nos == size_type(10))
                {
                    exceptional_shift_sum += shift_vector(0);
                    for (size_type i = size_type(0); i <= end; ++i)
                    {
                        h(i, i) -= shift_vector(0);
                    }
                    value_type s = (std::abs)(h(end, end - size_type(1))) + (std::abs)(h(end - size_type(1), end - size_type(2)));
                    shift_vector(0) = value_type(0.75) * s;
                    shift_vector(1) = value_type(0.75) * s;
                    shift_vector(2) = value_type(-0.4375) * s * s;
                }

                // Matlabs ad hoc shift (somehow Eigen people got this and its not in public domain)
                // Sometimes I just think Numerical Computing is a big insider job
                if (iter_nos == size_type(30))
                {
                    value_type s = (shift_vector(1) - shift_vector(0))*value_type(0.5);
                    s = s * s + shift_vector(2);
                    if (s > value_type(0))
                    {
                        s = (std::sqrt)(s);
                        if (shift_vector(1) < shift_vector(0))
                        {
                            s = -s;
                        }
                        s = s + ((shift_vector(1) - shift_vector(0))*value_type(0.5));
                        s = (shift_vector(0) - shift_vector(2)) / s;
                        exceptional_shift_sum += s;
                        for (size_type i = size_type(0); i <= end; ++i)
                            h(i, i) -= s;
                        shift_vector(0) = shift_vector(1) = shift_vector(2) = value_type(0.964);
                    }
                }

            }

            template<class M>
            void francis_qr_step(M &h, M &qv, typename M::size_type rowStart, typename M::size_type end, vector<typename M::value_type> shifts)
            {
                typedef typename M::size_type size_type;
                typedef typename M::value_type value_type;

                size_type colStart;
                vector<value_type> householderVec(3);

                for (colStart = end - size_type(2); colStart >= rowStart; --colStart)
                {
                    value_type tmm = h(colStart, colStart);
                    value_type r = shifts(0) - tmm;
                    value_type s = shifts(1) - tmm;
                    householderVec(0) = (r * s - shifts(2)) / h(colStart + size_type(1), colStart) + h(colStart, colStart + size_type(1));
                    householderVec(1) = h(colStart + size_type(1), colStart + size_type(1)) - tmm - r - s;
                    householderVec(2) = h(colStart + size_type(2), colStart + size_type(1));
                    if (colStart == rowStart) {
                        break;
                    }
                    value_type lhs = h(colStart, colStart - size_type(1)) * ((std::abs)(householderVec(1)) + (std::abs)(householderVec(2)));
                    value_type rhs = householderVec(0) * ((std::abs)(h(colStart - size_type(1), colStart - size_type(1))) + (std::abs)(tmm)+(std::abs)(h(colStart + size_type(1), colStart + size_type(1))));
                    if ((std::abs)(lhs) < (std::numeric_limits<value_type>::epsilon()) * rhs)
                    {
                        break;
                    }
                }

                size_type n = h.size1();
                bool firstTime = true;
                for (size_type k = colStart; k <= end - size_type(2); ++k)
                {
                    vector<value_type> tx(3);
                    if (firstTime)
                    {
                        tx = householderVec;
                        firstTime = false;
                    }
                    else
                    {
                        tx(0) = h(k, k - size_type(1));
                        tx(1) = h(k + size_type(1), k - size_type(1));
                        tx(2) = h(k + size_type(1), k - size_type(1));
                    }
                    vector<value_type> v;
                    value_type beta;
                    householder<vector<value_type> >(tx, v, beta);

                    if (beta != value_type(0))
                    {
                        if (firstTime && k > rowStart)
                        {
                            h(k, k - size_type(1)) = -h(k, k - size_type(1));
                        }
                        else if (!firstTime)
                        {
                            h(k, k - size_type(1)) = beta;
                        }

                        matrix<value_type> vvt = outer_prod(v, v);
                        vvt *= beta;
                        size_type n_vvt = vvt.size1();
                        matrix<value_type> imvvt = identity_matrix<value_type>(n_vvt) - vvt;

                        matrix<value_type> t1 = prod(imvvt, project(h, range(k, k + size_type(3)), range(k, n)));
                        project(h, range(k, k + size_type(3)), range(k, n)).assign(t1);

                        value_type r = (std::min)(end, k + size_type(3)) + size_type(1);
                        matrix<value_type> t2 = prod(project(h, range(0, r), range(k, k + size_type(3))), imvvt);
                        project(h, range(0, r), range(k, k + size_type(3))).assign(t2);

                        matrix<value_type> t3 = prod(project(qv, range(0, n), range(k, k + size_type(3))), imvvt);
                        project(qv, range(0, n), range(k, k + size_type(3))).assign(t3);

                    }
                }

                vector<value_type> tx(2);
                vector<value_type> v;
                tx(0) = h(end - size_type(1), end - size_type(2));
                tx(1) = h(end , end - size_type(2));
                value_type beta;
                householder<vector<value_type> >(tx, v, beta);

                if (beta != value_type(0))
                {
                    h(end - size_type(1), end - size_type(2)) = beta;

                    matrix<value_type> vvt = outer_prod(v, v);
                    vvt *= beta;
                    size_type n_vvt = vvt.size1();
                    matrix<value_type> imvvt = identity_matrix<value_type>(n_vvt) -vvt;

                    matrix<value_type> t1 = prod(imvvt, project(h, range(end - size_type(1), end + size_type(1)), range(end - size_type(1), n)));
                    project(h, range(end - size_type(1), end + size_type(1)), range(end - size_type(1), n)).assign(t1);

                    matrix<value_type> t2 = prod(project(h, range(0, end + size_type(1)), range(end - size_type(1), end + size_type(1))), imvvt);
                    project(h, range(0, end + size_type(1)), range(end - size_type(1), end + size_type(1))).assign(t2);

                    matrix<value_type> t3 = prod(project(qv, range(0, n), range(end - size_type(1), end + size_type(1))), imvvt);
                    project(qv, range(0, n), range(end - size_type(1), end + size_type(1))).assign(t3);
                }

                // Round off errors (this creates a lot of issues)
                for (size_type i = colStart + size_type(2); i <= end; ++i)
                {
                    h(i, i - size_type(2)) = value_type(0);
                    if (i > colStart + size_type(2))
                        h(i, i - size_type(3)) = value_type(0);
                }

            }

}}}


#endif
