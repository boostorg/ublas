//
//  Copyright (c) 2016
//  Michael Lehn, Imre Palik
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_UBLAS_BLOCK_SIZES_
#define _BOOST_UBLAS_BLOCK_SIZES_

#include <boost/numeric/ublas/detail/vector.hpp>

namespace boost { namespace numeric { namespace ublas { namespace detail {

    template <typename T>
    struct prod_block_size {
        static const unsigned vector_length = _BOOST_UBLAS_VECTOR_SIZE/sizeof(T); // Number of elements in a vector register
        static const unsigned mc = 256;
        static const unsigned kc = 512; // stripe length
        static const unsigned nc = (4096/(3 * vector_length)) * (3 * vector_length);
        static const unsigned mr = 4; // stripe width for lhs
        static const unsigned nr = 3 * vector_length; // stripe width for rhs
        static const unsigned align = 64; // align temporary arrays to this boundary
        static const unsigned limit = 14; // Use gemm from this size
        BOOST_STATIC_ASSERT_MSG(mc>0 && kc>0 && nc>0 && mr>0 && nr>0, "Invalid block size.");
        BOOST_STATIC_ASSERT_MSG(mc % mr == 0, "MC must be a multiple of MR.");
        BOOST_STATIC_ASSERT_MSG(nc % nr == 0, "NC must be a multiple of NR.");
    };

    template <>
    struct prod_block_size<float> {
        static const unsigned mc = 256;
        static const unsigned kc = 512; // stripe length
        static const unsigned nc = 4096;
        static const unsigned mr = 4; // stripe width for lhs
        static const unsigned nr = 16; // stripe width for rhs
        static const unsigned align = 64; // align temporary arrays to this boundary
        static const unsigned limit = 14; // Use gemm from this size
        static const unsigned vector_length = _BOOST_UBLAS_VECTOR_SIZE/sizeof(float); // Number of elements in a vector register
        BOOST_STATIC_ASSERT_MSG(mc>0 && kc>0 && nc>0 && mr>0 && nr>0, "Invalid block size.");
        BOOST_STATIC_ASSERT_MSG(mc % mr == 0, "MC must be a multiple of MR.");
        BOOST_STATIC_ASSERT_MSG(nc % nr == 0, "NC must be a multiple of NR.");
    };

    template <>
    struct prod_block_size<long double> {
        static const unsigned mc = 256;
        static const unsigned kc = 512; // stripe length
        static const unsigned nc = 4096;
        static const unsigned mr = 1; // stripe width for lhs
        static const unsigned nr = 4; // stripe width for rhs
        static const unsigned align = 64; // align temporary arrays to this boundary
        static const unsigned limit = 42; // Use gemm from this size
        static const unsigned vector_length = 1; // Number of elements in a vector register
        BOOST_STATIC_ASSERT_MSG(mc>0 && kc>0 && nc>0 && mr>0 && nr>0, "Invalid block size.");
        BOOST_STATIC_ASSERT_MSG(mc % mr == 0, "MC must be a multiple of MR.");
        BOOST_STATIC_ASSERT_MSG(nc % nr == 0, "NC must be a multiple of NR.");
    };

    template <typename T>
    struct prod_block_size<std::complex<T> > {
        static const unsigned vector_length = _BOOST_UBLAS_VECTOR_SIZE/sizeof(T); // Number of elements in a vector register
        static const unsigned mc = 255;
        static const unsigned kc = 512; // stripe length
        static const unsigned nc = 4096;
        static const unsigned mr = 3; // stripe width for lhs
        static const unsigned nr = vector_length; // stripe width for rhs
        static const unsigned align = 64; // align temporary arrays to this boundary
        static const unsigned limit = 23; // Use gemm from this size
        BOOST_STATIC_ASSERT_MSG(mc>0 && kc>0 && nc>0 && mr>0 && nr>0, "Invalid block size.");
        BOOST_STATIC_ASSERT_MSG(mc % mr == 0, "MC must be a multiple of MR.");
        BOOST_STATIC_ASSERT_MSG(nc % nr == 0, "NC must be a multiple of NR.");
    };

    template <>
    struct prod_block_size<std::complex<long double> > {
        static const unsigned mc = 256;
        static const unsigned kc = 512; // stripe length
        static const unsigned nc = 4096;
        static const unsigned mr = 1; // stripe width for lhs
        static const unsigned nr = 1; // stripe width for rhs
        static const unsigned align = 64; // align temporary arrays to this boundary
        static const unsigned limit = 68; // Use gemm from this size
        static const unsigned vector_length = 1; // Number of elements in a vector register
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

    template<typename T>
    struct check_blocksize {
        BOOST_STATIC_ASSERT_MSG(T::mc>0 && T::kc>0 && T::nc>0 && T::mr>0 && T::nr>0,
                                "Invalid block size.");
        BOOST_STATIC_ASSERT_MSG(T::mc % T::mr == 0,
                                "MC must be a multiple of MR.");
        BOOST_STATIC_ASSERT_MSG(T::nc % T::nr == 0,
                                "NC must be a multiple of NR.");
        BOOST_STATIC_ASSERT_MSG(T::vector_length <= 1 || T::nr % T::vector_length == 0,
                                "NR must be a multiple of vector size");
        BOOST_STATIC_ASSERT_MSG(T::limit >= 2,
                                "Minimum matrix size for gemm is 2*2");
    };
}}}}
#endif
