CONFIG -= qt
CONFIG += depend_includepath
win*: CONFIG += console

QMAKE_CXXFLAGS += -std=c++17 -fopenmp -g

# If ublas tests are build with boost source code then,
# then boost headers and boost libraries should be used.

BOOST_ROOT=../../../../../..

exists( $$BOOST_ROOT/boost-build.jam ) {
  message("Boost installed.")
  INCLUDEPATH += $${BOOST_ROOT}/../libs/numeric/ublas/include
  LIBS += -L$${BOOST_ROOT}/../stage/lib -lgomp
  QMAKE_RPATHDIR += $${BOOST_ROOT}/../stage/lib
}
