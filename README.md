Boost Linear and Multilinear Algebra Library 
=====

[![Language](https://img.shields.io/badge/C%2B%2B-11-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization)
[![License](https://img.shields.io/badge/license-BSL-blue.svg)](https://opensource.org/licenses/BSL-1.0)
[![Documentation](https://img.shields.io/badge/ublas-documentation-blue.svg)](https://www.boost.org/doc/libs/1_69_0/libs/numeric/ublas/doc/index.html)
[![Wiki](https://img.shields.io/badge/ublas-wiki-blue.svg)](https://github.com/boostorg/ublas/wiki)
[![Mailing List](https://img.shields.io/badge/ublas-mailing%20list-4eb899.svg)](https://lists.boost.org/mailman/listinfo.cgi/ublas)
[![Gitter](https://img.shields.io/badge/ublas-chat%20on%20gitter-4eb899.svg)](https://gitter.im/boostorg/ublas)

[![Windows](https://github.com/boostorg/ublas/actions/workflows/windows.yml/badge.svg)](https://github.com/boostorg/ublas/actions/workflows/windows.yml)
[![Linux](https://github.com/boostorg/ublas/actions/workflows/linux.yml/badge.svg)](https://github.com/boostorg/ublas/actions/workflows/linux.yml)
[![Apple MacOS](https://github.com/boostorg/ublas/actions/workflows/apple.yml/badge.svg)](https://github.com/boostorg/ublas/actions/workflows/apple.yml)
[![Clang Sanitizer](https://github.com/boostorg/ublas/actions/workflows/sanitizer.yml/badge.svg)](https://github.com/boostorg/ublas/actions/workflows/sanitizer.yml)
[![Clang Tidy](https://github.com/boostorg/ublas/actions/workflows/clangtidy.yml/badge.svg)](https://github.com/boostorg/ublas/actions/workflows/clangtidy.yml)
[![Codecov](https://codecov.io/gh/boostorg/ublas/branch/master/graph/badge.svg)](https://codecov.io/gh/boostorg/ublas/branch/master) 


**Boost.uBLAS** is part of the [Boost C++ Libraries](http://github.com/boostorg). 
It is directed towards scientific computing on the level of basic linear and multilinear algebra operations with tensors, matrices and vectors. 

## Documentation 
uBLAS is documented at [boost.org](https://www.boost.org/doc/libs/1_69_0/libs/numeric/ublas/doc/index.html).
The tensor extension has also a [wiki page](https://github.com/boostorg/ublas/wiki/Tensor).

## License
Distributed under the [Boost Software License, Version 1.0](http://www.boost.org/LICENSE_1_0.txt).

## Properties
* header-only
* requires C++20 compatible compiler
  * gcc version >= 10.x.x
  * clang version >= 10.x.x 
  * msvc version >= 14.28
* Unit-tests require Boost.Test

## Simple Example

```cpp
#include <boost/numeric/ublas/tensor.hpp> 
#include <iostream>

int main()
{
  using namespace boost::numeric::ublas::index;
  using tensor  = boost::numeric::ublas::tensor_dynamic<float>;
  auto ones     = boost::numeric::ublas::ones<float>{};

  tensor A = ones(3,4,5);
  tensor B = ones(4,6,3,2);

  tensor C = 2*ones(5,6,2) + A(_i,_j,_k)*B(_j,_l,_i,_m) + 5;
  
  // Matlab Compatible Formatted Output
  std::cout << "C=" << C << ";" << std::endl;
}
```

## Directories

| Name         | Purpose                 |
| ------------ | ----------------------- |
| `doc`        | documentation           |
| `examples`   | example files           |
| `include`    | headers                 |
| `test`       | unit tests              |
| `benchmarks` | timing and benchmarking |

## More information

* If you would like to test the library, contribute new feature or a bug fix, see [contribution](https://github.com/boostorg/ublas/wiki/Guidelines-for-Contribution).
* Ask questions in [stackoverflow](http://stackoverflow.com/questions/ask?tags=c%2B%2B,boost,boost-ublas) with `boost-ublas` or `ublas` tags.
* Report [bugs](https://github.com/boostorg/ublas/issues) and be sure to mention Boost version, platform and compiler you're using. A small compilable code sample to reproduce the problem is always good as well.
* Submit your patches as pull requests against **develop** branch. Note that by submitting patches you agree to license your modifications under the [Boost Software License, Version 1.0](http://www.boost.org/LICENSE_1_0.txt).
* Developer discussions about the library are held on the [Boost developers mailing list](https://lists.boost.org/mailman/listinfo.cgi/ublas). Be sure to read the [discussion policy](http://www.boost.org/community/policy.html) before posting and add the `[ublas]` tag at the beginning of the subject line
* For any other questions, you can contact David, Stefan or Cem: david.bellot-AT-gmail-DOT-com, cem.bassoy-AT-gmail-DOT-com stefan-AT-seefeld-DOT-name
