TEMPLATE = app
TARGET = bench5

CONFIG -= qt
CONFIG += depend_includepath

OTHER_FILES += \
    ../../../../benchmarks/bench5/Jamfile.v2

SOURCES += \
    ../../../../benchmarks/bench5/assignment_bench.cpp

INCLUDEPATH += \
    ../../../../include

# If ublas tests are build with boost source code then,
# then boost headers and boost libraries should be used.
exists(../../../../../../../boost-build.jam) {
    INCLUDEPATH += ../../../../../../..
    #LIBS += -L../../../../../../../stage/lib
}
