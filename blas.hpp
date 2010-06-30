//
//  Copyright (c) 2000-2002
//  Joerg Walter, Mathias Koch
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  GeNeSys mbH & Co. KG in producing this work.
//

#ifndef _BOOST_UBLAS_BLAS_
#define _BOOST_UBLAS_BLAS_

#include <boost/numeric/ublas/traits.hpp>

namespace boost { namespace numeric { namespace ublas {
	

    /** Interface and implementation of BLAS level 1
     *
     * This includes functions which perform \b vector-vector operations.
     * More information about BLAS can be found at 
     * <a href="http://en.wikipedia.org/wiki/BLAS">http://en.wikipedia.org/wiki/BLAS</a>
     */
    namespace blas_1 {

        /** 1-Norm: \f$\sum_i |x_i|\f$ (also called \f$\f$mathcal{L}_1 or Manhattan norm)
	 *
	 * \tparam V type of the vector (not needed by default)
	 * \param v a vector or vector expression
	 * \return the 1-Norm with type of the vector's type
	 */
        template<class V>
        typename type_traits<typename V::value_type>::real_type
        asum (const V &v) {
            return norm_1 (v);
        }

        /** 2-Norm: \f$\sum_i |x_i|^2\f$ (also called \f$\f$mathcal{L}_2 or Euclidean norm)
	 *
	 * \tparam V type of the vector (not needed by default)
	 * \param v a vector or vector expression
	 * \return the 2-Norm with type of the vector's type
	 */
        template<class V>
        typename type_traits<typename V::value_type>::real_type
        nrm2 (const V &v) {
            return norm_2 (v);
        }

        /** Infinite-norm: \f$\max_i |x_i|\f$ (also called \f$\f$mathcal{L}_\infty norm)
	 *
	 * \tparam V type of the vector (not needed by default)
	 * \param v a vector or vector expression
	 * \return the Infinite-Norm with type of the vector's type
	 */
        template<class V>
        typename type_traits<typename V::value_type>::real_type
        amax (const V &v) {
            return norm_inf (v);
        }

        /** Inner product of vectors \a v1 and \a v2
	 *
	 * \tparam V1 type of first vector (not needed by default)
	 * \tparam V2 type of second vector (not needed by default)
	 * \param v1 first vector of the inner product
	 * \param v2 second vector of the inner product
	 * \return the inner product of the type of the most generic type of the 2 vectors
	 */
        template<class V1, class V2>
        typename promote_traits<typename V1::value_type, typename V2::value_type>::promote_type
        dot (const V1 &v1, const V2 &v2) {
            return inner_prod (v1, v2);
        }

        /** Copy vector \a v2 to \a v1
	 *
	 * \tparam V1 type of first vector (not needed by default)
	 * \tparam V2 type of second vector (not needed by default)
	 * \param v1 target vector
	 * \param v2 source vector
	 * \return a reference to the target vector
	 */
        template<class V1, class V2>
        V1 &
        copy (V1 &v1, const V2 &v2) {
            return v1.assign (v2);
        }

        /** Swap vectors \a v1 and \a v2
	 *
         * \tparam V1 type of first vector (not needed by default)
	 * \tparam V2 type of second vector (not needed by default)
	 * \param v1 first vector
	 * \param v2 second vector
	 */
	template<class V1, class V2>
        void swap (V1 &v1, V2 &v2) {
            v1.swap (v2);
        }

        /** scale vector \a v with scalar \a t
	 *
	 * \tparam V type of the vector (not needed by default)
	 * \tparam T type of the scalar (not needed by default)
	 * \param v vector to be scaled
	 * \param t the scalar
	 * \return \c t*v
	 */
        template<class V, class T>
        V &
        scal (V &v, const T &t) {
            return v *= t;
        }

        /** Compute \f$v_1= v_1 +  t.v_2\f$
	 *
	 * \tparam V1 type of the first vector (not needed by default)
	 * \tparam T type of the scalar (not needed by default)
	 * \tparam V2 type of the second vector (not needed by default)
	 * \param v1 target and first vector
	 * \param t the scalar
	 * \param v2 second vector
	 * \return a reference to the first and target vector
	 */
        template<class V1, class T, class V2>
        V1 &
        axpy (V1 &v1, const T &t, const V2 &v2) {
            return v1.plus_assign (t * v2);
        }

	/** Performs rotation of points in the plane and assign the result to the first vector
	 *
	 * Each point is defined as a pair \c v1(i) and \c v2(i), being respectively 
	 * the \f$x\f$ and \f$y\f$ coordinates. The parameters \c t1 and \t2 are respectively 
	 * the cosine and sine of the angle of the rotation.
	 * Results are not returned but directly written into \c v1.
	 *
	 * \param t1 cosine of the rotation
	 * \param v1 vector of \f$x\f$ values
	 * \param t2 sine of the rotation 
	 * \param v2 vector of \f$y\f$ values
	 *
	 * \tparam T1 type of the cosine value (not needed by default)
	 * \tparam V1 type of the \f$x\f$ vector (not needed by default)
	 * \tparam T2 type of the sine value (not needed by default)
	 * \tparam V2 type of the \f$y\f$ vector (not needed by default)
	 */
        template<class T1, class V1, class T2, class V2>
        void rot (const T1 &t1, V1 &v1, const T2 &t2, V2 &v2) 
	{
            typedef typename promote_traits<typename V1::value_type, typename V2::value_type>::promote_type promote_type;
            vector<promote_type> vt (t1 * v1 + t2 * v2);
            v2.assign (- t2 * v1 + t1 * v2);
            v1.assign (vt);
        }

    }

    /** \brief Interface and implementation of BLAS level 2
     * This includes functions which perform \b matrix-vector operations.
     * More information about BLAS can be found at
     * <a href="http://en.wikipedia.org/wiki/BLAS">http://en.wikipedia.org/wiki/BLAS</a>
     */
    namespace blas_2 {

       /** \brief multiply vector \c v with triangular matrix \c m
	*
	* \param v a vector
	* \param m a triangular matrix
	* \return the result of the product
	*
	* \tparam V type of the vector (not needed by default)
	* \tparam M type of the matrix (not needed by default)
        */                 
        template<class V, class M>
        V & tmv (V &v, const M &m) 
	{
            return v = prod (m, v);
        }

        /** \brief solve \f$m.x = v\f$ in place, where \c m is a triangular matrix
	 *
	 * \param v a vector
	 * \param m a matrix
	 * \param C (this parameter is not needed)
	 * \return a result vector from the above operation
	 *
	 * \tparam V type of the vector (not needed by default)
	 * \tparam M type of the matrix (not needed by default)
	 * \tparam C n/a
         */                 
        template<class V, class M, class C>
        V & tsv (V &v, const M &m, C) 
	{
            return v = solve (m, v, C ());
        }

        /** \brief compute \f$ v1 = t1.v1 + t2.(m.v2)\f$, a general matrix-vector product
	 *
	 * \param v1 a vector
	 * \param t1 a scalar
	 * \param t2 another scalar
	 * \param m a matrix
	 * \param v2 another vector
	 * \return the vector \c v1 with the result from the above operation
	 *
	 * \tparam V1 type of first vector (not needed by default)
	 * \tparam T1 type of first scalar (not needed by default)
	 * \tparam T2 type of second scalar (not needed by default)
	 * \tparam M type of matrix (not needed by default)
	 * \tparam V2 type of second vector (not needed by default)
         */                 
        template<class V1, class T1, class T2, class M, class V2>
        V1 & gmv (V1 &v1, const T1 &t1, const T2 &t2, const M &m, const V2 &v2) 
	{
            return v1 = t1 * v1 + t2 * prod (m, v2);
        }

        /** \brief Rank 1 update: \f$ m = m + t.(v1.v2^T)\f$
	 *
	 * \param m a matrix
	 * \param t a scalar
	 * \param v1 a vector
	 * \param v2 another vector
	 * \return a matrix with the result from the above operation
	 *
	 * \tparam M type of matrix (not needed by default)
	 * \tparam T type of scalar (not needed by default)
	 * \tparam V1 type of first vector (not needed by default)
	 * \tparam V2type of second vector (not needed by default)
	 */
        template<class M, class T, class V1, class V2>
        M & gr (M &m, const T &t, const V1 &v1, const V2 &v2) 
	{
#ifndef BOOST_UBLAS_SIMPLE_ET_DEBUG
            return m += t * outer_prod (v1, v2);
#else
            return m = m + t * outer_prod (v1, v2);
#endif
        }

        /** \brief symmetric rank 1 update: \f$m = m + t.(v.v^T)\f$
	 *
	 * \param m a matrix
	 * \param t a scalar
	 * \param v a vector
	 * \return a matrix with the result from the above operation
	 *
	 * \tparam M type of matrix (not needed by default)
	 * \tparam T type of scalar (not needed by default)
	 * \tparam V type of vector (not needed by default)
	 */
        template<class M, class T, class V>
        M & sr (M &m, const T &t, const V &v) 
	{
#ifndef BOOST_UBLAS_SIMPLE_ET_DEBUG
            return m += t * outer_prod (v, v);
#else
            return m = m + t * outer_prod (v, v);
#endif
        }

        /** \brief hermitian rank 1 update: \f$m = m + t.(v.v^H)\f$
	 *
	 * \param m a matrix
	 * \param t a scalar
	 * \param v a vector
	 * \return a matrix with the result from the above operation
	 *
	 * \tparam M type of matrix (not needed by default)
	 * \tparam T type of scalar (not needed by default)
	 * \tparam V type of vector (not needed by default)
	 */
        template<class M, class T, class V>
        M & hr (M &m, const T &t, const V &v) 
	{
#ifndef BOOST_UBLAS_SIMPLE_ET_DEBUG
            return m += t * outer_prod (v, conj (v));
#else
            return m = m + t * outer_prod (v, conj (v));
#endif
        }

         /** \brief symmetric rank 2 update: \f$ m=m+ t.(v1.v2^T + v2.v1^T)\f$ 
	  *
	  * \param m a matrix
	  * \param t a scalar
	  * \param v1 a vector
	  * \param v2 another vector
	  * \return a matrix with the result from the above operation
	  *
	  * \tparam M type of matrix (not needed by default)
	  * \tparam T type of scalar (not needed by default)
	  * \tparam V1 type of first vector (not needed by default)
	  * \tparam V2type of second vector (not needed by default)
          */                 
        template<class M, class T, class V1, class V2>
        M & sr2 (M &m, const T &t, const V1 &v1, const V2 &v2) 
	{
#ifndef BOOST_UBLAS_SIMPLE_ET_DEBUG
            return m += t * (outer_prod (v1, v2) + outer_prod (v2, v1));
#else
            return m = m + t * (outer_prod (v1, v2) + outer_prod (v2, v1));
#endif
        }

        /** \brief hermitian rank 2 update: \f$m=m+t.(v1.v2^H) + v2.(t.v1)^H) 
	 *
	 * \param m a matrix
	 * \param t a scalar
	 * \param v1 a vector
	 * \param v2 another vector
	 * \return a matrix with the result from the above operation
	 *
	 * \tparam M type of matrix (not needed by default)
	 * \tparam T type of scalar (not needed by default)
	 * \tparam V1 type of first vector (not needed by default)
	 * \tparam V2type of second vector (not needed by default)
         */                 
        template<class M, class T, class V1, class V2>
        M & hr2 (M &m, const T &t, const V1 &v1, const V2 &v2) 
	{
#ifndef BOOST_UBLAS_SIMPLE_ET_DEBUG
            return m += t * outer_prod (v1, conj (v2)) + type_traits<T>::conj (t) * outer_prod (v2, conj (v1));
#else
            return m = m + t * outer_prod (v1, conj (v2)) + type_traits<T>::conj (t) * outer_prod (v2, conj (v1));
#endif
        }

    }

    /** \brief Interface and implementation of BLAS level 3
     * This includes functions which perform \b matrix-matrix operations.
     * More information about BLAS can be found at 
     * <a href="http://en.wikipedia.org/wiki/BLAS">http://en.wikipedia.org/wiki/BLAS</a>
     */
    namespace blas_3 {

        /** \brief triangular matrix multiplication
	 *
	 * \param m1
	 * \param t
	 * \param m2
	 * \param m3
	 *
	 * \tparam M1
	 * \tparam T
	 * \tparam M2
	 * \tparam M3
	 *
        */                 
        template<class M1, class T, class M2, class M3>
        M1 &
        tmm (M1 &m1, const T &t, const M2 &m2, const M3 &m3) {
            return m1 = t * prod (m2, m3);
        }

        /** \brief triangular solve \a m2 * \a x = \a t * \a m1 in place, \a m2 is a triangular matrix
	 *
	 * \param m1
	 * \param t
	 * \param m2
	 * \param C
	 *
	 * \tparam M1
	 * \tparam T
	 * \tparam M2
	 * \tparam C
         */                 
        template<class M1, class T, class M2, class C>
        M1 & tsm (M1 &m1, const T &t, const M2 &m2, C) {
            return m1 = solve (m2, t * m1, C ());
        }

        /** \brief general matrix multiplication
	 *
	 * \param m1
	 * \param t1
	 * \param t2
	 * \param m2
	 * \param m3
	 *
	 * \tparam M1
	 * \tparam T1
	 * \tparam T2
	 * \tparam M2
	 * \tparam M3
         */                 
        template<class M1, class T1, class T2, class M2, class M3>
        M1 &
        gmm (M1 &m1, const T1 &t1, const T2 &t2, const M2 &m2, const M3 &m3) {
            return m1 = t1 * m1 + t2 * prod (m2, m3);
        }

        /** \brief symmetric rank k update: \a m1 = \a t * \a m1 + \a t2 * (\a m2 * \a m2<sup>T</sup>)
	 *
	 * \param m1
	 * \param t1
	 * \param t2
	 * \param m2
	 *
	 * \tparam M1
	 * \tparam T1
	 * \tparam T2
	 * \tparam M2
	 * \todo use opb_prod()
         */                 
        template<class M1, class T1, class T2, class M2>
        M1 & srk (M1 &m1, const T1 &t1, const T2 &t2, const M2 &m2) 
	{
            return m1 = t1 * m1 + t2 * prod (m2, trans (m2));
        }

        /** \brief hermitian rank k update: \a m1 = \a t * \a m1 + \a t2 * (\a m2 * \a m2<sup>H</sup>)
	 *
	 * \param m1
	 * \param t1
	 * \param t2
	 * \param m2
	 *
	 * \tparam M1
	 * \tparam T1
	 * \tparam T2
	 * \tparam M2
	 * \todo use opb_prod()
         */                 
        template<class M1, class T1, class T2, class M2>
        M1 & hrk (M1 &m1, const T1 &t1, const T2 &t2, const M2 &m2) 
	{
            return m1 = t1 * m1 + t2 * prod (m2, herm (m2));
        }

        /** \brief generalized symmetric rank k update: \a m1 = \a t1 * \a m1 + \a t2 * (\a m2 * \a m3<sup>T</sup>) + \a t2 * (\a m3 * \a m2<sup>T</sup>)
	 *
	 * \param m1
	 * \param t1
	 * \param t1
	 * \param m2
	 * \param m3
	 *
	 * \tparam M1
	 * \tparam T1
	 * \tparam T2
	 * \tparam M2
	 * \tparam M3
	 * \todo use opb_prod()
         */                 
        template<class M1, class T1, class T2, class M2, class M3>
        M1 & sr2k (M1 &m1, const T1 &t1, const T2 &t2, const M2 &m2, const M3 &m3) 
	{
            return m1 = t1 * m1 + t2 * (prod (m2, trans (m3)) + prod (m3, trans (m2)));
        }

        /** \brief generalized hermitian rank k update: \a m1 = \a t1 * \a m1 + \a t2 * (\a m2 * \a m3<sup>H</sup>) + (\a m3 * (\a t2 * \a m2)<sup>H</sup>)
	 *
	 * \param m1
	 * \param t1
	 * \param t2
	 * \param m2
	 * \param m3
	 *
	 * \tparam M1
	 * \tparam T1
	 * \tparam T2
	 * \tparam M2
	 * \tparam M3
	 * \todo use opb_prod()
         */                 
        template<class M1, class T1, class T2, class M2, class M3>
        M1 & hr2k (M1 &m1, const T1 &t1, const T2 &t2, const M2 &m2, const M3 &m3) 
	{
            return m1 = 
		      t1 * m1 
		    + t2 * prod (m2, herm (m3)) 
		    + type_traits<T2>::conj (t2) * prod (m3, herm (m2));
        }

    }

}}}

#endif
