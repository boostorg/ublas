TEMPLATE = app
TARGET = test_tensor

CONFIG += staticlib depend_includepath console
CONFIG -= qt
CONFIG += c++20

#QMAKE_CXXFLAGS += -fno-inline
QMAKE_CXXFLAGS  =-std=c++20
QMAKE_CXXFLAGS +=-Wall -Wpedantic -Wextra
QMAKE_CXXFLAGS +=-Wno-unknown-pragmas
#QMAKE_CXXFLAGS +=-Wno-unused-but-set-variable


gcc:QMAKE_CXXFLAGS_RELEASE =-O3 -march=native -fopenmp
clang: QMAKE_CXXFLAGS_RELEASE =-O3 -march=native -fopenmp=libiomp5

gcc:QMAKE_CXXFLAGS_DEBUG = -g
clang: QMAKE_CXXFLAGS_DEBUG =-g

#QMAKE_CXXFLAGS += --coverage

BOOST_ROOT=../../../../../..

QMAKE_RPATHDIR += $${BOOST_ROOT}/stage/lib
INCLUDEPATH+=$$BOOST_ROOT/libs/numeric/ublas/include
LIBS+=-L$${BOOST_ROOT}/stage/lib -lboost_unit_test_framework -lgomp

#message("INCLUDEPATH: $${INCLUDEPATH}")

INCLUDE_DIR=$${BOOST_ROOT}/libs/numeric/ublas/include
TEST_DIR = ../../../test/tensor

include(../include/tensor/tensor.pri)

HEADERS += $${TEST_DIR}/utility.hpp

SOURCES += $${TEST_DIR}/algorithm/test_*.cpp
SOURCES += $${TEST_DIR}/extents/test_*.cpp
SOURCES += $${TEST_DIR}/functions/test_*.cpp
SOURCES += $${TEST_DIR}/multiplication/test_*.cpp
SOURCES += $${TEST_DIR}/tensor/test_*.cpp
SOURCES += $${TEST_DIR}/test_*.cpp
