//
//  Copyright (c) 2016
//  Imre Palik
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef _BOOST_UBLAS_DETAIL_VECTOR_
#define _BOOST_UBLAS_DETAIL_VECTOR_
#include <boost/predef/hardware/simd.h>

#if defined(BOOST_HW_SIMD) && (BOOST_HW_SIMD == BOOST_HW_SIMD_X86 || BOOST_HW_SIMD == BOOST_HW_SIMD_X86_AMD)
#if BOOST_HW_SIMD >= BOOST_HW_SIMD_X86_AVX_VERSION
#define _BOOST_UBLAS_VECTOR_SIZE 32
#elif BOOST_HW_SIMD >= BOOST_HW_SIMD_X86_SSE_VERSION
#define _BOOST_UBLAS_VECTOR_SIZE 16
#else
#define _BOOST_UBLAS_VECTOR_SIZE 8
#endif
#endif

#ifndef _BOOST_UBLAS_VECTOR_SIZE
#define _BOOST_UBLAS_VECTOR_SIZE 1
#endif

#if (defined(BOOST_COMP_GNUC_DETECTION)                           \
     && BOOST_COMP_GNUC_DETECTION >= BOOST_VERSION_NUMBER(4,8,0))
#define BOOST_UBLAS_VECTOR_KERNEL = _BOOST_UBLAS_VECTOR_SIZE
#endif

#endif
