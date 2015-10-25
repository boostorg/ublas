TEMPLATE = app
TARGET = sparse_view_test

QMAKE_CXXFLAGS += /EHa

include (configuration.pri)

SOURCES += \
    ../../../test/sparse_view_test.cpp
