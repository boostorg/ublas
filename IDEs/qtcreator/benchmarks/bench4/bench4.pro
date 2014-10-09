TEMPLATE = app
TARGET = bench4

CONFIG -= qt
CONFIG += depend_includepath

OTHER_FILES += \
    ../../../../benchmarks/bench4/Jamfile.v2

SOURCES += \
    ../../../../benchmarks/bench4/bench43.cpp \
    ../../../../benchmarks/bench4/bench42.cpp \
    ../../../../benchmarks/bench4/bench41.cpp \
    ../../../../benchmarks/bench4/bench4.cpp

INCLUDEPATH += \
    ../../../../include

# If ublas tests are build with boost source code then,
# then boost headers and boost libraries should be used.
exists(../../../../../../../boost-build.jam) {
    INCLUDEPATH += ../../../../../../..
    #LIBS += -L../../../../../../../stage/lib
}
