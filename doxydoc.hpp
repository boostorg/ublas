/// \class boost::numeric::ublas::c_matrix::const_iterator1
/// \brief An iterator on const values from a column perspective
/// An iterator on const values from a column perspective: if it points to a matrix value at \f$(i,j)\f$, incrementing it (resp. decrementing it) will make it point to \f$(i+1,j)\f$ (resp. \f$(i-1,j\f$) assuming this position is valid. It has reached the bottom of the matrix when its value equals end1().
/// Iterator1 follows the same semantic as STL iterators. Setting its value to begin()1 will point it to the position \f$(0,0)\f$ of the matrix. To start at another column the method find() can be used (e.g. find(0,10) to start at the 11-th column).

/// \class boost::numeric::ublas::c_vector
/// \brief An array based vector class which size is defined at type specification or object instanciation.
/// An array based matrix class which size is defined at type specification or object instanciation. This matrix is directly based on a predefinec C-style arry of data, thus providing the fastest implementation possible. The constraint is that dimensions of the matrix must be specified at the instanciation or the type specification. For instance, \code typedef c_matrix<double,4,4> my_4by4_matrix\encode defines a 4x4 double-precision matrix. You can also instantiate it directly with \code c_matrix<int,8,5> my_fast_matrix\endcode. This will make a 8 by 5 integer matrix. The price to pay for this speed is that you cannot resize it to a size larger than the one defined in the template parameters. In the previous example, a size of 4 by 5 or 3 by 2 is acceptable, but a new size of 9 by 5 or even 10 by 10 will raise a bad_size() exception.

/// \namespace boost::numeric::ublas::blas_1
/// \brief Interface and implementation of BLAS level 1
/// Interface and implementation of BLAS level 1. This includes functions which perform vector-vector operations.
/// More information about BLAS can be found at <a href="http://en.wikipedia.org/wiki/BLAS">http://en.wikipedia.org/wiki/BLAS</a>

/// \namespace boost::numeric::ublas::blas_2
/// \brief Interface and implementation of BLAS level 2
/// Interface and implementation of BLAS level 2. This includes functions which perform matrix-vector operations.
/// More information about BLAS can be found at <a href="http://en.wikipedia.org/wiki/BLAS">http://en.wikipedia.org/wiki/BLAS</a>

/// \namespace boost::numeric::ublas::blas_3
/// \brief Interface and implementation of BLAS level 3
/// Interface and implementation of BLAS level 3. This includes functions which perform matrix-matrix operations.
/// More information about BLAS can be found at <a href="http://en.wikipedia.org/wiki/BLAS">http://en.wikipedia.org/wiki/BLAS</a>
