#!/bin/bash

# Copyright 2020 Rene Rivera, Sam Darwin
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE.txt or copy at http://boost.org/LICENSE_1_0.txt)

if [ "$DRONE_JOB_UUID" = "b6589fc6ab" ] || [ "$DRONE_JOB_UUID" = "356a192b79" ] || [ "$DRONE_JOB_UUID" = "da4b9237ba" ] || [ "$DRONE_JOB_UUID" = "77de68daec" ] || [ "$DRONE_JOB_UUID" = "1b64538924" ] ; then
    export CLBLAS_PREFIX="${TRAVIS_BUILD_DIR}/CLBLAS/"
    export PATH="${CLBLAS_PREFIX}/bin:$PATH"
    export LD_LIBRARY_PATH="${CLBLAS_PREFIX}/lib:$LD_LIBRARY_PATH"
    if [ ${TRAVIS_OS_NAME} == "linux" ]; then .ci/install-ocl-ubuntu.sh; fi
    .ci/install-clblas.sh
    cmake --version;
    true ${CC} --version;
    ${CXX} --version;
fi

