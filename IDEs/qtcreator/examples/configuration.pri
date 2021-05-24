CONFIG -= qt
CONFIG += depend_includepath
win*: CONFIG += console
CONFIG += object_parallel_to_source

QMAKE_CXXFLAGS  =-std=c++20
QMAKE_CXXFLAGS +=-Wall -Wpedantic -Wextra
QMAKE_CXXFLAGS +=-Wno-unknown-pragmas
QMAKE_CXXFLAGS +=-Wno-unused-but-set-variable

gcc:QMAKE_CXXFLAGS_RELEASE =-O3 -march=native -fopenmp
clang: QMAKE_CXXFLAGS_RELEASE =-O3 -march=native -fopenmp=libiomp5

gcc:QMAKE_CXXFLAGS_DEBUG += -g
clang: QMAKE_CXXFLAGS_DEBUG +=-g

BOOST_ROOT=../../../../../../../..
QMAKE_RPATHDIR += $${BOOST_ROOT}/stage/lib
INCLUDEPATH+=$$BOOST_ROOT/libs/numeric/ublas/include
LIBS+=-L$${BOOST_ROOT}/stage/lib

#message("INCLUDEPATH: $${INCLUDEPATH}")

INCLUDE_DIR=$${BOOST_ROOT}/libs/numeric/ublas/include
