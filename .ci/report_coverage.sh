#! /bin/bash -e

sudo apt-get install lcov
lcov --directory $BOOST_ROOT --capture --output-file coverage.info 

# Remove all unwanted coverages libs. 
# Boost.uBLAS depends uses many internal boost libs, we don't want them to be in coverage. 

lcov --extract coverage.info "*/boost/numeric/ublas/tensor/*" "*/libs/numeric/ublas/tensor/*" --output-file coverage.info
lcov --list coverage.info
