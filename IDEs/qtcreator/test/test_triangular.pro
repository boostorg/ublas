TEMPLATE = app
TARGET = test_triangular

!include (configuration.pri)

SOURCES += \
    ../../../test/test_triangular.cpp

LIBS += -lboost_timer -lboost_system -lboost_chrono
