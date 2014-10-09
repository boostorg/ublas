TEMPLATE = app
TARGET = bench3

CONFIG -= qt
CONFIG += depend_includepath

OTHER_FILES += \
    ../../../../benchmarks/bench3/Jamfile.v2

HEADERS += \
    ../../../../benchmarks/bench3/bench3.hpp

SOURCES += \
    ../../../../benchmarks/bench3/bench33.cpp \
    ../../../../benchmarks/bench3/bench32.cpp \
    ../../../../benchmarks/bench3/bench31.cpp \
    ../../../../benchmarks/bench3/bench3.cpp

INCLUDEPATH += \
    ../../../../include

# If ublas tests are build with boost source code then,
# then boost headers and boost libraries should be used.
exists(../../../../../../../boost-build.jam) {
    INCLUDEPATH += ../../../../../../..
    #LIBS += -L../../../../../../../stage/lib
}
