#! /bin/bash -e

sudo apt-get install lcov
lcov --rc lcov_branch_coverage=1 --directory $BOOST_ROOT --capture --output-file coverage.info 

# Remove all unwanted coverages libs. 
# Boost.uBLAS depends uses many internal boost libs, we don't want them to be in coverage. 

lcov --rc lcov_branch_coverage=1 --extract coverage.info "*/boost/numeric/ublas/*" "*/libs/numeric/ublas/*" --output-file coverage.info
lcov --rc lcov_branch_coverage=1 --list coverage.info
