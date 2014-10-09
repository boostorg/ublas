TEMPLATE = app
TARGET = bench1

CONFIG -= qt
CONFIG += depend_includepath

OTHER_FILES += \
    ../../../../benchmarks/bench1/Jamfile.v2

HEADERS += \
    ../../../../benchmarks/bench1/bench1.hpp

SOURCES += \
    ../../../../benchmarks/bench1/bench13.cpp \
    ../../../../benchmarks/bench1/bench12.cpp \
    ../../../../benchmarks/bench1/bench11.cpp \
    ../../../../benchmarks/bench1/bench1.cpp

INCLUDEPATH += \
    ../../../../include

# If ublas tests are build with boost source code then,
# then boost headers and boost libraries should be used.
exists(../../../../../../../boost-build.jam) {
    INCLUDEPATH += ../../../../../../..
    #LIBS += -L../../../../../../../stage/lib
}
