Boost Linear and Multilinear Algebra Library 
=====

[![Language](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization)
[![License](https://img.shields.io/badge/license-BSL-blue.svg)](https://opensource.org/licenses/BSL-1.0)
[![Documentation](https://img.shields.io/badge/ublas-documentation-blue.svg)](https://www.boost.org/doc/libs/1_69_0/libs/numeric/ublas/doc/index.html)
[![Wiki](https://img.shields.io/badge/ublas-wiki-blue.svg)](https://github.com/boostorg/ublas/wiki)
[![Mailing List](https://img.shields.io/badge/ublas-mailing%20list-4eb899.svg)](https://lists.boost.org/mailman/listinfo.cgi/ublas)
[![Discussions](https://img.shields.io/badge/ublas-discussions-blue)](https://github.com/boostorg/ublas/discussions)
[![Gitter](https://img.shields.io/badge/ublas-chat%20on%20gitter-4eb899.svg)](https://gitter.im/boostorg/ublas)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](https://github.com/boostorg/ublas/wiki/Code-of-Conduct)

[![Windows](https://github.com/boostorg/ublas/actions/workflows/windows.yml/badge.svg)](https://github.com/boostorg/ublas/actions/workflows/windows.yml)
[![Linux](https://github.com/boostorg/ublas/actions/workflows/linux.yml/badge.svg)](https://github.com/boostorg/ublas/actions/workflows/linux.yml)
[![Apple MacOS](https://github.com/boostorg/ublas/actions/workflows/apple.yml/badge.svg)](https://github.com/boostorg/ublas/actions/workflows/apple.yml)
[![Clang Sanitizer](https://github.com/boostorg/ublas/actions/workflows/sanitizer.yml/badge.svg)](https://github.com/boostorg/ublas/actions/workflows/sanitizer.yml)
[![Clang Tidy](https://github.com/boostorg/ublas/actions/workflows/clangtidy.yml/badge.svg)](https://github.com/boostorg/ublas/actions/workflows/clangtidy.yml)
[![Codecov](https://codecov.io/gh/boostorg/ublas/branch/master/graph/badge.svg)](https://codecov.io/gh/boostorg/ublas/branch/master) 

Boost.uBlas is a header-only library and part of the [Boost C++ libraries](http://github.com/boostorg).
It provides a set of basic linear and multilinear algebra operations with tensors, matrices and vectors.
uBLAS is documented at [boost.org](https://www.boost.org/doc/libs/1_69_0/libs/numeric/ublas/doc/index.html) or in [docs](https://github.com/BoostGSoC21/ublas/blob/develop/doc/ublas.adoc).
Simple examples of the tensor extension is provided by the [wiki page](https://github.com/boostorg/ublas/wiki/Tensor).

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

## Requirements
* Boost.uBlas requires a C++20 compatible compiler
  * gcc version >= 10.x.x
  * clang version >= 10.x.x 
  * msvc version >= 14.28
* Unit-tests require Boost.Test

## Discussions and Questions
* Be sure to have read our [code of conduct](https://github.com/boostorg/ublas/wiki/Code-of-Conduct) and Boost's [discussion policy](http://www.boost.org/community/policy.html)
* Contributors and maintainers mainly use [Discussions](https://github.com/boostorg/ublas/discussions) (previously [Gitter](https://gitter.im/boostorg/ublas)) to discuss library design or governance topics. 
* Users of the library can choose between the following platforms preferably [Gitter](https://gitter.im/boostorg/ublas), the [Boost developers mailing list](https://lists.boost.org/mailman/listinfo.cgi/ublas) using the `[ublas]` tag at the beginning of the subject line or [Stackoverflow](http://stackoverflow.com/questions/ask?tags=c%2B%2B,boost,boost-ublas) using `boost-ublas` or `ublas` tags
* Contact the maintainers [Cem](https://gitter.im/bassoy), [David](https://gitter.im/yimyom) or [Stefan](https://gitter.im/stefanseefeld) if you have suggestions

## Contribution
* Be sure that you have read our [guidelines for contributors](https://github.com/boostorg/ublas/wiki/Guidelines-for-Contributors) and our [code of conduct](https://github.com/boostorg/ublas/wiki/Code-of-Conduct) 
* If you want to report a bug, create an [issue](https://github.com/boostorg/ublas/issues) 
* If you want to submit a patch, create pull request against **develop** branch.

## Directories

| Name         | Purpose                 |
| ------------ | ----------------------- |
| `doc`        | documentation           |
| `examples`   | example files           |
| `include`    | headers                 |
| `test`       | unit tests              |
| `benchmarks` | timing and benchmarking |
