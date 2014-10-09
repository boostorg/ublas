TEMPLATE = app
TARGET = bench2

CONFIG -= qt
CONFIG += depend_includepath

OTHER_FILES += \
    ../../../../benchmarks/bench2/Jamfile.v2

HEADERS += \
    ../../../../benchmarks/bench2/bench2.hpp

SOURCES += \
    ../../../../benchmarks/bench2/bench23.cpp \
    ../../../../benchmarks/bench2/bench22.cpp \
    ../../../../benchmarks/bench2/bench21.cpp \
    ../../../../benchmarks/bench2/bench2.cpp

INCLUDEPATH += \
    ../../../../include

# If ublas tests are build with boost source code then,
# then boost headers and boost libraries should be used.
exists(../../../../../../../boost-build.jam) {
    INCLUDEPATH += ../../../../../../..
    #LIBS += -L../../../../../../../stage/lib
}
