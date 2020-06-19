TEMPLATE = app
TARGET = test_tensor

CONFIG += staticlib depend_includepath console
CONFIG -= qt
CONFIG += c++20

CONFIG += c++17

#QMAKE_CXXFLAGS += -fno-inline
QMAKE_CXXFLAGS  =-std=c++20
QMAKE_CXXFLAGS +=-Wall -Wpedantic -Wextra
QMAKE_CXXFLAGS +=-Wno-unknown-pragmas
QMAKE_CXXFLAGS +=-Wno-unused-but-set-variable


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

HEADERS += \
  $${TEST_DIR}/utility.hpp

HEADERS += \
  $${TEST_DIR}/utility.hpp

SOURCES += \
  $${TEST_DIR}/test_access.cpp \
  $${TEST_DIR}/test_algorithms.cpp \
  $${TEST_DIR}/test_einstein_notation.cpp \
  $${TEST_DIR}/test_expression.cpp \
  $${TEST_DIR}/test_expression_evaluation.cpp \
  $${TEST_DIR}/test_extents_dynamic.cpp \
  $${TEST_DIR}/test_extents_dynamic_rank_static.cpp \
  $${TEST_DIR}/test_fixed_rank_expression_evaluation.cpp \
  $${TEST_DIR}/test_fixed_rank_extents.cpp \
  $${TEST_DIR}/test_fixed_rank_functions.cpp \
  $${TEST_DIR}/test_fixed_rank_operators_arithmetic.cpp \
  $${TEST_DIR}/test_fixed_rank_operators_comparison.cpp \
  $${TEST_DIR}/test_fixed_rank_strides.cpp \
  $${TEST_DIR}/test_fixed_rank_tensor.cpp \
  $${TEST_DIR}/test_fixed_rank_tensor_matrix_vector.cpp \
  $${TEST_DIR}/test_functions.cpp \
  $${TEST_DIR}/test_multi_index.cpp \
  $${TEST_DIR}/test_multi_index_utility.cpp \
  $${TEST_DIR}/test_multiplication.cpp \
  $${TEST_DIR}/test_operators_arithmetic.cpp \
  $${TEST_DIR}/test_operators_comparison.cpp \
  $${TEST_DIR}/test_static_expression_evaluation.cpp \
  $${TEST_DIR}/test_static_extents.cpp \
  # $${TEST_DIR}/test_static_functions.cpp \
  $${TEST_DIR}/test_static_operators_arithmetic.cpp \
  $${TEST_DIR}/test_static_operators_comparison.cpp \
  $${TEST_DIR}/test_static_strides.cpp \
  $${TEST_DIR}/test_static_tensor.cpp \
  $${TEST_DIR}/test_static_tensor_matrix_vector.cpp \
  $${TEST_DIR}/test_strides.cpp \
  $${TEST_DIR}/test_tensor.cpp \
  $${TEST_DIR}/test_tensor_matrix_vector.cpp \
  $${TEST_DIR}/test_extents_functions.cpp \
  $${TEST_DIR}/test_span.cpp \
  $${TEST_DIR}/test_subtensor.cpp \
  $${TEST_DIR}/test_subtensor_utility.cpp
