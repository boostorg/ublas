TEMPLATE = app
TARGET = test

CONFIG -= qt
CONFIG += depend_includepath debug
win*: CONFIG += console

QMAKE_CXXFLAGS += -fno-inline
QMAKE_CXXFLAGS += -std=c++17

DEFINES += BOOST_UBLAS_NO_EXCEPTIONS
win*: DEFINES += _SCL_SECURE_NO_WARNINGS

#Visual age IBM
xlc: DEFINES += BOOST_UBLAS_NO_ELEMENT_PROXIES

# If ublas tests are build with boost source code then,
# then boost headers and boost libraries should be used.
exists(../../../../../../boost-build.jam) {
	INCLUDEPATH += ../../../../../..
	LIBS += -L../../../../../../stage/lib
	QMAKE_RPATHDIR += ../../../../../../stage/lib
}


LIBS +=-lboost_unit_test_framework

HEADERS += \
	../../../test/tensor/utility.hpp

SOURCES += \
    ../../../test/tensor/test_tensor.cpp \
	../../../test/tensor/test_extents.cpp \
	../../../test/tensor/test_strides.cpp \
	../../../test/tensor/test_expression.cpp \
	../../../test/tensor/test_operators_comparison.cpp \	
	../../../test/tensor/test_operators_arithmetic.cpp \
    ../../../test/tensor/test_tensor_matrix_vector.cpp


INCLUDEPATH += \
	../../../include
