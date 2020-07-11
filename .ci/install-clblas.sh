#! /bin/bash -e

wget https://github.com/clMathLibraries/clBLAS/releases/download/v2.12/clBLAS-2.12.0-Linux-x64.tar.gz
tar xvf clBLAS-2.12.0-Linux-x64.tar.gz
if [ ${CLBLAS_PREFIX} ]; then
    echo "Installing clBLAS-2.12.0"
    mkdir -p ${CLBLAS_PREFIX}/lib/
    cp -r clBLAS-2.12.0-Linux-x64/include ${CLBLAS_PREFIX}/include
    cp -r clBLAS-2.12.0-Linux-x64/lib64/* ${CLBLAS_PREFIX}/lib/
    cp -r clBLAS-2.12.0-Linux-x64/bin ${CLBLAS_PREFIX}/bin
else
    echo "CLBLAS_PREFIX is not set. Skipping CLBLAS Installation"
fi
