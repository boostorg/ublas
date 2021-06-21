TEMPLATE = subdirs
CONFIG += ordered
SUBDIRS = include # examples # benchmarks
OTHER_FILES += ../../changelog.txt

CONFIG += c++17
QMAKE_CXXFLAGS += -std=c++17

#include (tests.pri)












