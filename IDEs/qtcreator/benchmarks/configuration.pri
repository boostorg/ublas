CONFIG -= qt
CONFIG += depend_includepath
win*: CONFIG += console

# ublas include directory
INCLUDEPATH += \
    ../../../../include

# If ublas tests are build with boost source code then,
# then boost headers and boost libraries should be used.
exists(../../../../../../boost-build.jam) {
    INCLUDEPATH += ../../../../../../..
    LIBS += -L../../../../../../../stage/lib
    QMAKE_RPATHDIR += ../../../../../../../stage/lib
}
