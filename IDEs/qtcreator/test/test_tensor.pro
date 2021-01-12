TEMPLATE = app
TARGET = test

CONFIG -= qt
CONFIG += depend_includepath debug
win*: CONFIG += console

#QMAKE_CXXFLAGS += -fno-inline
QMAKE_CXXFLAGS += -std=c++17
#QMAKE_CXXFLAGS += -Wno-unknown-pragmas
#QMAKE_CXXFLAGS += --coverage


DEFINES += BOOST_UBLAS_NO_EXCEPTIONS
win*: DEFINES += _SCL_SECURE_NO_WARNINGS

#Visual age IBM
xlc: DEFINES += BOOST_UBLAS_NO_ELEMENT_PROXIES

# If ublas tests are build with boost source code then,
# then boost headers and boost libraries should be used.
#exists(../../../../../../boost-build.jam) {
#	INCLUDEPATH += ../../../../../..
#	LIBS += -L../../../../../../stage/lib
#	QMAKE_RPATHDIR += ../../../../../../stage/lib
#}

INCLUDEPATH += /usr/local/include
INCLUDEPATH += ../../../include
LIBS += -L/usr/local/lib
LIBS +=-lboost_unit_test_framework
# -lgcov

TEST_DIR = ../../../test/tensor

HEADERS += \
  $${TEST_DIR}/utility.hpp

SOURCES += \
  $${TEST_DIR}/test_algorithms.cpp \
  $${TEST_DIR}/test_einstein_notation.cpp \
  $${TEST_DIR}/test_expression.cpp \
  $${TEST_DIR}/test_expression_evaluation.cpp \
  $${TEST_DIR}/test_extents.cpp \
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
  $${TEST_DIR}/test_tensor_matrix_vector.cpp


INCLUDEPATH += \
	../../../include
