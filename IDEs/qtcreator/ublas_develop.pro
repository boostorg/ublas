TEMPLATE = subdirs
CONFIG += ordered
SUBDIRS = include benchmarks
OTHER_FILES += ../../changelog.txt


SUBDIRS += test_tensor
test_tensor.file = test/test_tensor.pro

#include (tests.pri)
