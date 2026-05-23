# Use, modification, and distribution are
# subject to the Boost Software License, Version 1.0. (See accompanying
# file LICENSE.txt)
#
# Copyright Rene Rivera 2020.

# For Drone CI we use the Starlark scripting language to reduce duplication.
# As the yaml syntax for Drone CI is rather limited.
#
#
globalenv={}
linuxglobalimage="cppalliance/droneubuntu1804:1"
windowsglobalimage="cppalliance/dronevs2019"

def main(ctx):
  return [
  linux_cxx("TOOLSET=gcc COMPILER=g++-7 CXXSTD=11 Job 0", "g++-7", packages="g++-7 g++-9 clang-7 clang-10 libopenblas-base rpm2cpio cpio clinfo opencl-headers ocl-icd-opencl-dev", llvm_os="bionic", llvm_ver="10", buildtype="boost", buildscript="drone", image=linuxglobalimage, environment={'TOOLSET': 'gcc', 'COMPILER': 'g++-7', 'CXXSTD': '11', 'DRONE_JOB_UUID': 'b6589fc6ab'}, globalenv=globalenv),
  linux_cxx("TOOLSET=gcc COMPILER=g++-7 CXXSTD=17 Job 1", "g++-7", packages="g++-7 g++-9 clang-7 clang-10 libopenblas-base rpm2cpio cpio clinfo opencl-headers ocl-icd-opencl-dev", llvm_os="bionic", llvm_ver="10", buildtype="boost", buildscript="drone", image=linuxglobalimage, environment={'TOOLSET': 'gcc', 'COMPILER': 'g++-7', 'CXXSTD': '17', 'DRONE_JOB_UUID': '356a192b79'}, globalenv=globalenv),
  linux_cxx("TOOLSET=gcc COMPILER=g++-9 CXXSTD=2a Job 2", "g++-9", packages="g++-7 g++-9 clang-7 clang-10 libopenblas-base rpm2cpio cpio clinfo opencl-headers ocl-icd-opencl-dev", llvm_os="bionic", llvm_ver="10", buildtype="boost", buildscript="drone", image=linuxglobalimage, environment={'TOOLSET': 'gcc', 'COMPILER': 'g++-9', 'CXXSTD': '2a', 'DRONE_JOB_UUID': 'da4b9237ba'}, globalenv=globalenv),
  linux_cxx("TOOLSET=clang COMPILER=clang++-7 CXXSTD=17 Job 3", "clang++-7", packages="g++-7 g++-9 clang-7 clang-10 libopenblas-base rpm2cpio cpio clinfo opencl-headers ocl-icd-opencl-dev", llvm_os="bionic", llvm_ver="10", buildtype="boost", buildscript="drone", image=linuxglobalimage, environment={'TOOLSET': 'clang', 'COMPILER': 'clang++-7', 'CXXSTD': '17', 'DRONE_JOB_UUID': '77de68daec'}, globalenv=globalenv),
  linux_cxx("TOOLSET=clang COMPILER=clang++-10 CXXSTD=2a Job 4", "clang++-10", packages="g++-7 g++-9 clang-7 clang-10 libopenblas-base rpm2cpio cpio clinfo opencl-headers ocl-icd-opencl-dev", llvm_os="bionic", llvm_ver="10", buildtype="boost", buildscript="drone", image=linuxglobalimage, environment={'TOOLSET': 'clang', 'COMPILER': 'clang++-10', 'CXXSTD': '2a', 'DRONE_JOB_UUID': '1b64538924'}, globalenv=globalenv),
    ]

# from https://github.com/boostorg/boost-ci
load("@boost_ci//ci/drone/:functions.star", "linux_cxx","windows_cxx","osx_cxx","freebsd_cxx")
