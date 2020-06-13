Boost Linear and Multilinear Algebra Library 
=====
[![Language](https://img.shields.io/badge/C%2B%2B-11-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization)
[![License](https://img.shields.io/badge/license-BSL-blue.svg)](https://opensource.org/licenses/BSL-1.0)
[![Documentation](https://img.shields.io/badge/ublas-documentation-blue.svg)](https://www.boost.org/doc/libs/1_69_0/libs/numeric/ublas/doc/index.html)
[![Wiki](https://img.shields.io/badge/ublas-wiki-blue.svg)](https://github.com/boostorg/ublas/wiki)
[![Mailing List](https://img.shields.io/badge/ublas-mailing%20list-4eb899.svg)](https://lists.boost.org/mailman/listinfo.cgi/ublas)
[![Gitter](https://img.shields.io/badge/ublas-chat%20on%20gitter-4eb899.svg)](https://gitter.im/boostorg/ublas)

**Boost.uBLAS** is part of the [Boost C++ Libraries](http://github.com/boostorg). It is directed towards scientific computing on the level of basic linear and multilinear algebra operations with tensors, matrices and vectors. 


## Documentation 
uBLAS is documented at [boost.org](https://www.boost.org/doc/libs/1_69_0/libs/numeric/ublas/doc/index.html).
The tensor extension has also a [wiki page](https://github.com/BoostGSoC18/tensor/wiki).

## License
Distributed under the [Boost Software License, Version 1.0](http://www.boost.org/LICENSE_1_0.txt).

## Properties
* Header-only
* Tensor extension requires C++17 compatible compiler, testing on
  * `gcc-7`, `gcc-8`, `gcc-9`, `gcc-10`
  * `clang-6.0`, `clang-8`, `clang-9`, `clang-10`
  * `msvc 14.16`,  `msvc 14.26`
  * Apple Clang on macOS Catalina
* Unit-tests require Boost.Test

## Build Status

| [`master`](https://github.com/boostorg/ublas/tree/master)    | [`develop`](https://github.com/boostorg/ublas/tree/develop)  | [`github-actions`](https://github.com/BoostGSoC20/ublas/tree/github-actions) |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| [![Linux GCC](https://github.com/BoostGSoC20/ublas/workflows/Linux%20GCC/badge.svg?branch=master)](https://github.com/BoostGSoC20/ublas/actions?query=workflow%3A%22Linux+GCC%22+branch%3Amaster) | [![Linux GCC](https://github.com/BoostGSoC20/ublas/workflows/Linux%20GCC/badge.svg?branch=develop)](https://github.com/BoostGSoC20/ublas/actions?query=workflow%3A%22Linux+GCC%22+branch%3Adevelop) | [![Linux GCC](https://github.com/BoostGSoC20/ublas/workflows/Linux%20GCC/badge.svg?branch=github-actions)](https://github.com/BoostGSoC20/ublas/actions?query=workflow%3A%22Linux+GCC%22+branch%3Agithub-actions) |
| [![Linux Clang](https://github.com/BoostGSoC20/ublas/workflows/Linux%20Clang/badge.svg?branch=master)](https://github.com/BoostGSoC20/ublas/actions?query=workflow%3A%22Linux+Clang%22+branch%3Amaster) | [![Linux Clang](https://github.com/BoostGSoC20/ublas/workflows/Linux%20Clang/badge.svg?branch=develop)](https://github.com/BoostGSoC20/ublas/actions?query=workflow%3A%22Linux+Clang%22+branch%3Adevelop) | [![Linux Clang](https://github.com/BoostGSoC20/ublas/workflows/Linux%20Clang/badge.svg?branch=github-actions)](https://github.com/BoostGSoC20/ublas/actions?query=workflow%3A%22Linux+Clang%22+branch%3github-actions) |
| [![Windows MSVC](https://github.com/BoostGSoC20/ublas/workflows/Windows%20MSVC/badge.svg?branch=master)](https://github.com/BoostGSoC20/ublas/actions?query=workflow%3A%22Windows+MSVC%22+branch%3Amaster) | [![Windows MSVC](https://github.com/BoostGSoC20/ublas/workflows/Windows%20MSVC/badge.svg?branch=develop)](https://github.com/BoostGSoC20/ublas/actions?query=workflow%3A%22Windows+MSVC%22+branch%3Adevelop) | [![Windows MSVC](https://github.com/BoostGSoC20/ublas/workflows/Windows%20MSVC/badge.svg?branch=github-actions)](https://github.com/BoostGSoC20/ublas/actions?query=workflow%3A%22Windows+MSVC%22+branch%3Agithub-actions) |
| [![Apple Clang](https://github.com/BoostGSoC20/ublas/workflows/Apple%20Clang/badge.svg?branch=master)](https://github.com/BoostGSoC20/ublas/actions?query=workflow%3A%22Apple+Clang%22+branch%3Amaster) | [![Apple Clang](https://github.com/BoostGSoC20/ublas/workflows/Apple%20Clang/badge.svg?branch=develop)](https://github.com/BoostGSoC20/ublas/actions?query=workflow%3A%22Apple+Clang%22+branch%3Adevelop) | [![Apple Clang](https://github.com/BoostGSoC20/ublas/workflows/Apple%20Clang/badge.svg?branch=github-actions)](https://github.com/BoostGSoC20/ublas/actions?query=workflow%3A%22Apple+Clang%22+branch%3Agithub-actions) |
| [![Code Format](https://github.com/BoostGSoC20/ublas/workflows/Code%20Format/badge.svg?branch=master)](https://github.com/BoostGSoC20/ublas/actions?query=workflow%3A%22Code+Format%22+branch%3Amaster) | [![Code Format](https://github.com/BoostGSoC20/ublas/workflows/Code%20Format/badge.svg?branch=develop)](https://github.com/BoostGSoC20/ublas/actions?query=workflow%3A%22Code+Format%22+branch%3Adevelop) | [![Code Format](https://github.com/BoostGSoC20/ublas/workflows/Code%20Format/badge.svg?branch=github-actions)](https://github.com/BoostGSoC20/ublas/actions?query=workflow%3A%22Code+Format%22+branch%3Agithub-actions) |
| [![Clang tidy checks](https://github.com/BoostGSoC20/ublas/workflows/Clang%20tidy%20checks/badge.svg?branch=master)](https://github.com/BoostGSoC20/ublas/actions?query=workflow%3A%22Clang+tidy+checks%22+branch%3Amaster) | [![Clang tidy checks](https://github.com/BoostGSoC20/ublas/workflows/Clang%20tidy%20checks/badge.svg?branch=develop)](https://github.com/BoostGSoC20/ublas/actions?query=workflow%3A%22Clang+tidy+checks%22+branch%3Adevelop) | [![Clang tidy checks](https://github.com/BoostGSoC20/ublas/workflows/Clang%20tidy%20checks/badge.svg?branch=github-actions)](https://github.com/BoostGSoC20/ublas/actions?query=workflow%3A%22Clang+tidy+checks%22+branch%3github-actions) |
| [![ublas](https://img.shields.io/badge/ublas-master-blue.svg)](https://www.boost.org/development/tests/master/developer/numeric-ublas.html) | [![ublas](https://img.shields.io/badge/ublas-develop-blue.svg)](https://www.boost.org/development/tests/develop/developer/numeric-ublas.html) | [![ublas](https://img.shields.io/badge/ublas-develop-blue.svg)](https://www.boost.org/development/tests/develop/developer/numeric-ublas.html) |
| [![codecov](https://codecov.io/gh/boostorg/ublas/branch/master/graph/badge.svg)](https://codecov.io/gh/boostorg/ublas/branch/master) | [![codecov](https://codecov.io/gh/boostorg/ublas/branch/develop/graph/badge.svg)](https://codecov.io/gh/boostorg/ublas/branch/develop) | [![codecov](https://codecov.io/gh/boostorg/ublas/branch/master/graph/badge.svg)](https://codecov.io/gh/boostorg/ublas/branch/master) |
| [![Documentation](https://img.shields.io/badge/docs-develop-brightgreen.svg)](http://www.boost.org/doc/libs/release/libs/numeric) | [![Documentation](https://img.shields.io/badge/docs-develop-brightgreen.svg)](http://www.boost.org/doc/libs/release/libs/numeric) | [![Documentation](https://img.shields.io/badge/docs-develop-brightgreen.svg)](http://www.boost.org/doc/libs/release/libs/numeric) |



## Directories

| Name        | Purpose                        |
| ----------- | ------------------------------ |
| `doc`       | documentation                  |
| `examples`  | example files                  |
| `include`   | headers                        |
| `test`      | unit tests                     |
| `benchmarks`| timing and benchmarking        |

## More information

* If you would like to test the library, contribute new feature or a bug fix, see [contribution](https://github.com/boostorg/ublas/wiki/Guidelines-for-Contribution) where the whole development infrastructure and the contributing workflow is explained in details.
* Ask questions in [stackoverflow](http://stackoverflow.com/questions/ask?tags=c%2B%2B,boost,boost-ublas) with `boost-ublas` or `ublas` tags.
* Report [bugs](https://github.com/boostorg/ublas/issues) and be sure to mention Boost version, platform and compiler you're using. A small compilable code sample to reproduce the problem is always good as well.
* Submit your patches as pull requests against **develop** branch. Note that by submitting patches you agree to license your modifications under the [Boost Software License, Version 1.0](http://www.boost.org/LICENSE_1_0.txt).
* Developer discussions about the library are held on the [Boost developers mailing list](https://lists.boost.org/mailman/listinfo.cgi/ublas). Be sure to read the [discussion policy](http://www.boost.org/community/policy.html) before posting and add the `[ublas]` tag at the beginning of the subject line
* For any other questions, you can contact David, Stefan or Cem: david.bellot-AT-gmail-DOT-com, cem.bassoy-AT-gmail-DOT-com stefan-AT-seefeld-DOT-name
