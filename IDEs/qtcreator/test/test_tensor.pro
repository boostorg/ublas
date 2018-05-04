TEMPLATE = app
TARGET = test

#include (configuration.pri)

QMAKE_CXXFLAGS += -std=c++17

LIBS +=-lboost_unit_test_framework

SOURCES += \
    ../../../test/tensor/test_tensor.cpp \
	../../../test/tensor/test_extents.cpp \
	../../../test/tensor/test_strides.cpp


INCLUDEPATH += \
    ../../../include
