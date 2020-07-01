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
* Tensor extension requires C++17 compatible compiler
* Unit-tests require Boost.Test

## Build Status


#### Build & Test

|      Operating System       |      Compiler       |  [`master`](https://github.com/boostorg/ublas/tree/master)   | [`develop`](https://github.com/boostorg/ublas/tree/develop)  |
| :-------------------------: | :-----------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Linux (Ubuntu 20.04 x86_64) |  gcc-{7, 8, 9, 10}  | [![Linux GCC Debug](https://github.com/boostorg/ublas/workflows/Linux%20GCC%20Debug/badge.svg?branch=master)](https://github.com/boostorg/ublas/actions?query=workflow%3A%22Linux+GCC%22+branch%3Amaster) | [![Linux GCC Debug](https://github.com/boostorg/ublas/workflows/Linux%20GCC%20Debug/badge.svg?branch=develop)](https://github.com/boostorg/ublas/actions?query=workflow%3A%22Linux+GCC%22+branch%3Adevelop) |
| Linux (Ubuntu 20.04 x86_64) | clang-{6, 8, 9, 10} | [![Linux Clang Release](https://github.com/boostorg/ublas/workflows/Linux%20Clang%20Release/badge.svg?branch=master)](https://github.com/boostorg/ublas/actions?query=workflow%3A%22Linux+Clang%22+branch%3Amaster) | [![Linux Clang Release](https://github.com/boostorg/ublas/workflows/Linux%20Clang%20Release/badge.svg?branch=develop)](https://github.com/boostorg/ublas/actions?query=workflow%3A%22Linux+Clang%22+branch%3Adevelop) |
|     Windows 10 (x86_64)     | msvc-{14.16, 14.26} | [![Windows MSVC](https://github.com/boostorg/ublas/workflows/Windows%20MSVC/badge.svg?branch=master)](https://github.com/boostorg/ublas/actions?query=workflow%3A%22Windows+MSVC%22+branch%3Amaster) | [![Windows MSVC](https://github.com/boostorg/ublas/workflows/Windows%20MSVC/badge.svg?branch=develop)](https://github.com/boostorg/ublas/actions?query=workflow%3A%22Windows+MSVC%22+branch%3Adevelop) |
|   MacOS Catalina (x86_64)   |      clang-11       | [![Apple Clang](https://github.com/boostorg/ublas/workflows/Apple%20Clang/badge.svg?branch=master)](https://github.com/boostorg/ublas/actions?query=workflow%3A%22Apple+Clang%22+branch%3Amaster) | [![Apple Clang](https://github.com/boostorg/ublas/workflows/Apple%20Clang/badge.svg?branch=develop)](https://github.com/boostorg/ublas/actions?query=workflow%3A%22Apple+Clang%22+branch%3Adevelop) |

#### Additional Checks

|    Checks     |  [`master`](https://github.com/boostorg/ublas/tree/master)   | [`develop`](https://github.com/boostorg/ublas/tree/develop)  |
| :-----------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| UB Sanitizer  | [![Undefined Behaviour Sanitizer](https://github.com/boostorg/ublas/workflows/Undefined%20Behaviour%20Sanitizer/badge.svg?branch=master)](https://github.com/boostorg/ublas/actions?query=workflow%3A%22Undefined+Behaviour+Sanitizer%22+branch%3Amaster) | [![Undefined Behaviour Sanitizer](https://github.com/boostorg/ublas/workflows/Undefined%20Behaviour%20Sanitizer/badge.svg?branch=develop)](https://github.com/boostorg/ublas/actions?query=workflow%3A%22Undefined+Behaviour+Sanitizer%22+branch%3Adevelop) |
| TH Sanitizer  | [![Thread Sanitizer](https://github.com/boostorg/ublas/workflows/Thread%20Sanitizer/badge.svg?branch=master)](https://github.com/boostorg/ublas/actions?query=workflow%3A%22Thread+Sanitizer%22+branch%3Amaster) | [![Thread Sanitizer](https://github.com/boostorg/ublas/workflows/Thread%20Sanitizer/badge.svg?branch=develop)](https://github.com/boostorg/ublas/actions?query=workflow%3A%22Thread+Sanitizer%22+branch%3Adevelop) |
| ADD Sanitizer | [![Address Sanitizer](https://github.com/boostorg/ublas/workflows/Address%20Sanitizer/badge.svg?branch=master)](https://github.com/boostorg/ublas/actions?query=workflow%3A%22Address+Sanitizer%22+branch%3Amaster) | [![Address Sanitizer](https://github.com/boostorg/ublas/workflows/Address%20Sanitizer/badge.svg?branch=develop)](https://github.com/boostorg/ublas/actions?query=workflow%3A%22Address+Sanitizer%22+branch%3Adevelop) |
|    Codecov    | [![codecov](https://codecov.io/gh/boostorg/ublas/branch/master/graph/badge.svg)](https://codecov.io/gh/boostorg/ublas/branch/master) | [![codecov](https://codecov.io/gh/boostorg/ublas/branch/develop/graph/badge.svg)](https://codecov.io/gh/boostorg/ublas/branch/develop) |
| Clang-Format  | [![Code Format](https://github.com/boostorg/ublas/workflows/Code%20Format/badge.svg?branch=master)](https://github.com/boostorg/ublas/actions?query=workflow%3A%22Code+Format%22+branch%3Amaster) | [![Code Format](https://github.com/boostorg/ublas/workflows/Code%20Format/badge.svg?branch=develop)](https://github.com/boostorg/ublas/actions?query=workflow%3A%22Code+Format%22+branch%3Adevelop) |
|  Clang-Tidy   | [![Clang tidy checks](https://github.com/boostorg/ublas/workflows/Clang%20tidy%20checks/badge.svg?branch=master)](https://github.com/boostorg/ublas/actions?query=workflow%3A%22Clang+tidy+checks%22+branch%3Amaster) | [![Clang tidy checks](https://github.com/boostorg/ublas/workflows/Clang%20tidy%20checks/badge.svg?branch=develop)](https://github.com/boostorg/ublas/actions?query=workflow%3A%22Clang+tidy+checks%22+branch%3Adevelop) |

#### Build Information

|         OS         |    Toolchain    |                  Compiler Flags                   |
| :----------------: | :-------------: | :-----------------------------------------------: |
| Linux Ubuntu 20.04 |       GCC       |                       `-O0`                       |
| Linux Ubuntu 20.04 |      Clang      |                       `-O3`                       |
|     Windows 10     |      MSVC       |                 No Special Flags                  |
|   MacOS Catalina   |      Clang      |                 No Special Flags                  |
|    UB Sanitizer    |  GCC and Clang  |            `"-g -fsanitize=undefined"`            |
|    TH Sanitizer    |  GCC and Clang  |           `"-g -fsanitize=thread -O2"`            |
|   ADD Sanitizer    |  GCC and Clang  | `"-g -fsanitize=address -fno-omit-frame-pointer"` |
|     Clang Tidy     |  Clang-Tidy-10  |         [Configuration File](.clang-tidy)         |
|    Clang Format    | Clang-Format-10 |        [Configuration File](.clang-format)        |



## Directories

| Name         | Purpose                 |
| ------------ | ----------------------- |
| `doc`        | documentation           |
| `examples`   | example files           |
| `include`    | headers                 |
| `test`       | unit tests              |
| `benchmarks` | timing and benchmarking |

## More information

* If you would like to test the library, contribute new feature or a bug fix, see [contribution](https://github.com/boostorg/ublas/wiki/Guidelines-for-Contribution) where the whole development infrastructure and the contributing workflow is explained in details.
* Ask questions in [stackoverflow](http://stackoverflow.com/questions/ask?tags=c%2B%2B,boost,boost-ublas) with `boost-ublas` or `ublas` tags.
* Report [bugs](https://github.com/boostorg/ublas/issues) and be sure to mention Boost version, platform and compiler you're using. A small compilable code sample to reproduce the problem is always good as well.
* Submit your patches as pull requests against **develop** branch. Note that by submitting patches you agree to license your modifications under the [Boost Software License, Version 1.0](http://www.boost.org/LICENSE_1_0.txt).
* Developer discussions about the library are held on the [Boost developers mailing list](https://lists.boost.org/mailman/listinfo.cgi/ublas). Be sure to read the [discussion policy](http://www.boost.org/community/policy.html) before posting and add the `[ublas]` tag at the beginning of the subject line
* For any other questions, you can contact David, Stefan or Cem: david.bellot-AT-gmail-DOT-com, cem.bassoy-AT-gmail-DOT-com stefan-AT-seefeld-DOT-name
