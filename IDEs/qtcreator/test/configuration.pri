CONFIG -= qt
CONFIG += depend_includepath

# ublas include and test directory are included
INCLUDEPATH += \
    ../../../include \
    ../../test

# If ublas tests are build with boost source code then,
# then boost headers and boost libraries should be used.
exists(../../../../../../boost-build.jam) {
    INCLUDEPATH += ../../../../../..
    LIBS += -L../../../../../../stage/lib
    QMAKE_RPATHDIR += ../../../../../../stage/lib
}

# Execute test once compiled.
win32: QMAKE_POST_LINK = ./$${DESTDIR}/$${TARGET}.exe
else: QMAKE_POST_LINK = ./$${DESTDIR}/$${TARGET}
