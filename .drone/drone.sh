#!/bin/bash

# Copyright 2020 Rene Rivera, Sam Darwin
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE.txt or copy at http://boost.org/LICENSE_1_0.txt)

set -e
export TRAVIS_BUILD_DIR=$(pwd)
export DRONE_BUILD_DIR=$(pwd)
export TRAVIS_BRANCH=$DRONE_BRANCH
export VCS_COMMIT_ID=$DRONE_COMMIT
export GIT_COMMIT=$DRONE_COMMIT
export REPO_NAME=$DRONE_REPO
export PATH=~/.local/bin:/usr/local/bin:$PATH

if [ "$DRONE_JOB_BUILDTYPE" == "boost" ]; then

echo '==================================> BEFORE_INSTALL'

. .drone/before-install.sh

echo '==================================> INSTALL'

cd ..
git clone -b master --depth 1 https://github.com/boostorg/boost.git boost-root
cd boost-root
git submodule update --init --jobs 8 tools/build
git submodule update --init --jobs 8 libs/config
git submodule update --init --jobs 8 tools/boostdep
mkdir -p libs/numeric/
cp -rp $TRAVIS_BUILD_DIR/. libs/numeric/ublas
python tools/boostdep/depinst/depinst.py -I benchmarks numeric/ublas
./bootstrap.sh
./b2 -j 8 headers
export BOOST_ROOT="`pwd`"

echo '==================================> SCRIPT'

echo "using $TOOLSET : : $COMPILER ;" >> ~/user-config.jam;
echo "using clblas : : <include>${CLBLAS_PREFIX}/include <search>${CLBLAS_PREFIX}/lib ;" >> ~/user-config.jam;
cp $TRAVIS_BUILD_DIR/opencl.jam ~/
cp $TRAVIS_BUILD_DIR/clblas.jam ~/
cd libs/numeric/ublas
$BOOST_ROOT/b2 -j 8 test toolset=$TOOLSET cxxstd=$CXXSTD

fi
