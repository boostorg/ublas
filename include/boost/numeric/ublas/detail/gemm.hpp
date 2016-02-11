//
//  Copyright (c) 2016
//  Michael Lehn, Imre Palik
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_UBLAS_GEMM_
#define _BOOST_UBLAS_GEMM_

#include <boost/type_traits/common_type.hpp>
#include <boost/type_traits/aligned_storage.hpp>
#include <boost/align/aligned_allocator.hpp>
#include <boost/align/assume_aligned.hpp>
#include <boost/static_assert.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

namespace boost { namespace numeric { namespace ublas { namespace detail {

    template <typename T>
    struct prod_block_size {
        static const unsigned mc = 256;
        static const unsigned kc = 512; // stripe length
        static const unsigned nc = 4092;
        static const unsigned mr = 4; // stripe width for lhs
        static const unsigned nr = 12; // stripe width for rhs
        static const unsigned align = 64; // align temporary arrays to this boundary
        static const unsigned limit = 26; // Use gemm from this size
        BOOST_STATIC_ASSERT_MSG(mc>0 && kc>0 && nc>0 && mr>0 && nr>0, "Invalid block size.");
        BOOST_STATIC_ASSERT_MSG(mc % mr == 0, "MC must be a multiple of MR.");
        BOOST_STATIC_ASSERT_MSG(nc % nr == 0, "NC must be a multiple of NR.");
    };

    template <>
    struct prod_block_size<float> {
        static const unsigned mc = 256;
        static const unsigned kc = 512; // stripe length
        static const unsigned nc = 4080;
        static const unsigned mr = 4; // stripe width for lhs
        static const unsigned nr = 24; // stripe width for rhs
        static const unsigned align = 64; // align temporary arrays to this boundary
        static const unsigned limit = 26; // Use gemm from this size
        BOOST_STATIC_ASSERT_MSG(mc>0 && kc>0 && nc>0 && mr>0 && nr>0, "Invalid block size.");
        BOOST_STATIC_ASSERT_MSG(mc % mr == 0, "MC must be a multiple of MR.");
        BOOST_STATIC_ASSERT_MSG(nc % nr == 0, "NC must be a multiple of NR.");
    };

    template <>
    struct prod_block_size<long double> {
        static const unsigned mc = 256;
        static const unsigned kc = 512; // stripe length
        static const unsigned nc = 4096;
        static const unsigned mr = 2; // stripe width for lhs
        static const unsigned nr = 2; // stripe width for rhs
        static const unsigned align = 64; // align temporary arrays to this boundary
        static const unsigned limit = 26; // Use gemm from this size
        BOOST_STATIC_ASSERT_MSG(mc>0 && kc>0 && nc>0 && mr>0 && nr>0, "Invalid block size.");
        BOOST_STATIC_ASSERT_MSG(mc % mr == 0, "MC must be a multiple of MR.");
        BOOST_STATIC_ASSERT_MSG(nc % nr == 0, "NC must be a multiple of NR.");
    };

    template<typename T>
    struct is_blocksize {
        struct fallback { static const int nr = 0; };
        struct derived : T, fallback {};
        template<int C1>
       struct nonr {
           static const bool value = false;
           typedef false_type type;
       };

         template<typename C> static char (&f(typename nonr<C::nr>::type*))[1];
         template<typename C> static char (&f(...))[2];

         static bool const value = sizeof(f<derived>(0)) == 2;
    };

    template <typename E>
    void
    gescal(const typename E::value_type &alpha, matrix_expression<E> &X)
    {
        typedef typename E::size_type  size_type;

       for (size_type i=0; i<X().size1(); ++i) {
            for (size_type j=0; j<X().size2(); ++j) {
               X()(i,j) *= alpha;
            }
        }
    }

    template <typename Index, typename T>
    void
    geaxpy(Index m, Index n, const T &alpha,
           const T *X, Index incRowX, Index incColX,
           T       *Y, Index incRowY, Index incColY)
    {
        for (Index j=0; j<n; ++j) {
            for (Index i=0; i<m; ++i) {
                Y[i*incRowY+j*incColY] += alpha*X[i*incRowX+j*incColX];
            }
        }
    }

    template <typename Index, typename TX>
    void
    gescal(Index m, Index n,
           const TX &alpha,
           TX *X, Index incRowX, Index incColX)
    {
      if (alpha != TX(0)) {
            for (Index j=0; j<n; ++j) {
                for (Index i=0; i<m; ++i) {
                    X[i*incRowX+j*incColX] *= alpha;
                }
            }
        } else {
            for (Index j=0; j<n; ++j) {
                for (Index i=0; i<m; ++i) {
                    X[i*incRowX+j*incColX] = 0;
                }
            }
        }
    }

    //-- Micro Kernel --------------------------------------------------------------
    template <typename Index, typename T, typename TC, typename BlockSize>
    void
    ugemm(Index kc, TC alpha, const T *A, const T *B,
          TC beta, TC *C, Index incRowC, Index incColC)
    {
        BOOST_ALIGN_ASSUME_ALIGNED(A, BlockSize::align);
        BOOST_ALIGN_ASSUME_ALIGNED(B, BlockSize::align);
        static const Index MR = BlockSize::mr;
        static const Index NR = BlockSize::nr;
       typename boost::aligned_storage<sizeof(T[MR*NR]),BlockSize::align>::type Pa;
       T *P = reinterpret_cast<T*>(Pa.address());
       for (unsigned c = 0; c < MR * NR; c++)
         P[c] = 0;

        for (Index l=0; l<kc; ++l) {
           for (Index i=0; i<MR; ++i) {
              for (Index j=0; j<NR; ++j) {
                    P[i* NR+j] += A[i]*B[j];
                }
            }
           A += MR;
           B += NR;
        }

       if (alpha!=TC(1)) {
           for (Index i=0; i<MR; ++i) {
               for (Index j=0; j<NR; ++j) {
                   P[i*NR+j] *= alpha;
               }
           }
       }

       if (beta == TC(0)) {
           for (Index i=0; i<MR; ++i) {
               for (Index j=0; j<NR; ++j) {
                C[i*incRowC+j*incColC] = P[i*NR+j];
               }
           }
       } else {
           for (Index i=0; i<MR; ++i) {
               for (Index j=0; j<NR; ++j) {
                    C[i*incRowC+j*incColC] *= beta;
                    C[i*incRowC+j*incColC] += P[i*NR+j];
               }
           }
       }
    }

    //-- Macro Kernel --------------------------------------------------------------
       template <typename Index, typename T, typename TC, typename BlockSize>
    void
    mgemm(Index mc, Index nc, Index kc, TC alpha,
          const T *A, const T *B, TC beta,
          TC *C, Index incRowC, Index incColC)
    {
        static const Index MR = BlockSize::mr;
        static const Index NR = BlockSize::nr;
        const Index mp  = (mc+MR-1) / MR;
        const Index np  = (nc+NR-1) / NR;
        const Index mr_ = mc % MR;
        const Index nr_ = nc % NR;

        // #if defined(_OPENMP)
        // #pragma omp parallel for
        // #endif
        for (Index j=0; j<np; ++j) {
      // __builtin_prefetch(B + j * kc * NR, 0);
      // __builtin_prefetch(B + j * kc * NR + 8, 0);
      // __builtin_prefetch(B + j * kc * NR + 16, 0);
            const Index nr = (j!=np-1 || nr_==0) ? NR : nr_;
            TC C_[MR*NR];

            for (Index i=0; i<mp; ++i) {
                const Index mr = (i!=mp-1 || mr_==0) ? MR : mr_;

                if (mr==MR && nr==NR) {
                 ugemm<Index, T, TC, BlockSize>(kc, alpha,
                          &A[i*kc*MR], &B[j*kc*NR],
                          beta,
                          &C[i*MR*incRowC+j*NR*incColC],
                          incRowC, incColC);
                } else {
                    std::fill_n(C_, MR*NR, T(0));
                    ugemm<Index, T, TC, BlockSize>(kc, alpha,
                          &A[i*kc*MR], &B[j*kc*NR],
                          T(0),
                          C_, NR, Index(1));
                    gescal(mr, nr, beta,
                           &C[i*MR*incRowC+j*NR*incColC],
                           incRowC, incColC);
                    geaxpy(mr, nr, TC(1), C_, NR, Index(1),
                           &C[i*MR*incRowC+j*NR*incColC],
                           incRowC, incColC);
                }
            }
        }
    }

    //-- Packing blocks ------------------------------------------------------------
       template <typename E, typename T, typename BlockSize>
    void
    pack_A(const matrix_expression<E> &A, T *p)
    {
        typedef typename E::size_type  size_type;

        const size_type mc = A ().size1();
        const size_type kc = A ().size2();
        static const size_type MR = BlockSize::mr;
        const size_type mp = (mc+MR-1) / MR;

        for (size_type j=0; j<kc; ++j) {
            for (size_type l=0; l<mp; ++l) {
                for (size_type i0=0; i0<MR; ++i0) {
                    size_type i  = l*MR + i0;
                    size_type nu = l*MR*kc + j*MR + i0;
                    p[nu]        = (i<mc) ? A()(i,j) : T(0);
                }
            }
        }
    }

       template <typename E, typename T, typename BlockSize>
    void
    pack_B(const matrix_expression<E> &B, T *p)
    {
        typedef typename E::size_type  size_type;

        const size_type kc = B ().size1();
        const size_type nc = B ().size2();
        static const size_type NR = BlockSize::nr;
        const size_type np = (nc+NR-1) / NR;

        for (size_type l=0; l<np; ++l) {
            for (size_type j0=0; j0<NR; ++j0) {
                for (size_type i=0; i<kc; ++i) {
                    size_type j  = l*NR+j0;
                    size_type nu = l*NR*kc + i*NR + j0;
                    p[nu]        = (j<nc) ? B()(i,j) : T(0);
                }
            }
        }
    }

    //-- Frame routine -------------------------------------------------------------
    template <typename E1, typename E2, typename E3, typename BlockSize>
    void
    gemm(typename E3::value_type alpha, const matrix_expression<E1> &e1,
        const matrix_expression<E2> &e2,
         typename E3::value_type beta, matrix_expression<E3> &e3,
        BlockSize)
    {
        typedef typename E3::size_type  size_type;
        typedef typename E1::value_type value_type1;
        typedef typename E2::value_type value_type2;
        typedef typename E3::value_type value_type3;
        typedef typename common_type<value_type1,
                                     value_type2,
                                     value_type3>::type value_type_i;
        typedef unbounded_array<value_type_i,
                                typename alignment::aligned_allocator<value_type_i,
                                                                      BlockSize::align> > array_type_i;

        static const size_type MC = BlockSize::mc;
        static const size_type NC = BlockSize::nc;

        const size_type m = BOOST_UBLAS_SAME (e3 ().size1 (), e1 ().size1 ());
        const size_type n = BOOST_UBLAS_SAME (e3 ().size2 (), e2 ().size2 ());
        const size_type k = BOOST_UBLAS_SAME (e1 ().size2 (), e2 ().size1 ());

        static const size_type KC = BlockSize::kc;
        const size_type mb = (m+MC-1) / MC;
        const size_type nb = (n+NC-1) / NC;
        const size_type kb = (k+KC-1) / KC;
        const size_type mc_ = m % MC;
        const size_type nc_ = n % NC;
        const size_type kc_ = k % KC;

        value_type3 *C_ = &e3()(0,0);
        const size_type incRowC = &e3()(1,0) - &e3()(0,0);
        const size_type incColC = &e3()(0,1) - &e3()(0,0);

        if (alpha==value_type3(0) || k==0) {
            gescal(beta, e3);
            return;
        }

        array_type_i A(MC * KC);
        array_type_i B(NC * KC);

        for (size_type j=0; j<nb; ++j) {
            size_type nc = (j!=nb-1 || nc_==0) ? NC : nc_;

            for (size_type l=0; l<kb; ++l) {
                size_type kc = (l!=kb-1 || kc_==0) ? KC : kc_;
                value_type3 beta_ = (l==0) ? beta : value_type3(1);

                const matrix_range<const E2> Bs =
                    subrange(e2(), l*KC, l*KC+kc, j*NC, j*NC+nc);
                pack_B<matrix_range<const E2>, value_type_i, BlockSize>
                    (Bs, &B[0]);

                for (size_type i=0; i<mb; ++i) {
                    size_type mc = (i!=mb-1 || mc_==0) ? MC : mc_;

                    const matrix_range<const E1> As =
                        subrange(e1(), i*MC, i*MC+mc, l*KC, l*KC+kc);
                    pack_A<matrix_range<const E1>, value_type_i, BlockSize>
                        (As, &A[0]);

                    mgemm<size_type, value_type_i, value_type3, BlockSize>
                        (mc, nc, kc, alpha, &A[0], &B[0], beta_,
                         &C_[i*MC*incRowC+j*NC*incColC],
                         incRowC, incColC);
                }
            }
        }
    }
}}}}
#endif
