//  Copyright (c) 2012 Oswin Krause
//  Copyright (c) 2013 Joaquim Duran
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef BOOST_UBLAS_MATRIX_VECTOR_HPP
#define BOOST_UBLAS_MATRIX_VECTOR_HPP

#include <boost/numeric/ublas/matrix_proxy.hpp>//for matrix_row, matrix_column and matrix_expression
#include <boost/numeric/ublas/vector.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/utility/enable_if.hpp>

namespace boost { namespace numeric { namespace ublas {

namespace detail{
/// \brief Iterator which represents a matrix as a range of row/column-vectors
///
/// The second argument is the reference to a matrix_row/matrix_column.
/// Whatever type used, it must offer a constructor Reference(sequence,i)
/// which constructs a reference to the i-th proxy-element
/// This iterator is invalidated when the underlying matrix is resized.
template<class Matrix, class Reference>
struct matrix_vector_iterator: public boost::iterator_facade<
    matrix_vector_iterator<Matrix,Reference>,
    typename vector_temporary_traits<Reference>::type,
    boost::random_access_traversal_tag,
    Reference
>{
public:
    matrix_vector_iterator(){}

    ///\brief constructs a matrix_vector_iterator as pointing to the i-th proxy
    matrix_vector_iterator(Matrix& matrix, std::size_t position)
    : matrix_(&matrix),position_(position) {}

    template<class M, class R>
    matrix_vector_iterator(matrix_vector_iterator<M,R> const& other)
    : matrix_(other.matrix_),position_(other.position_) {}

private:
    friend class boost::iterator_core_access;
    template <class M,class R> friend class matrix_vector_iterator;

    void increment() {
        ++position_;
    }
    void decrement() {
        --position_;
    }

    void advance(std::ptrdiff_t n){
        position_ += n;
    }

    template<class M,class R>
    std::ptrdiff_t distance_to(matrix_vector_iterator<M,R> const& other) const{
        BOOST_UBLAS_CHECK (matrix_ == other.matrix_, external_logic ());
        return (std::ptrdiff_t)other.position_ - (std::ptrdiff_t)position_;
    }

    template<class M,class R>
    bool equal(matrix_vector_iterator<M,R> const& other) const{
        BOOST_UBLAS_CHECK (matrix_ == other.matrix_, external_logic ());
        return (position_ == other.position_);
    }
    Reference dereference() const {
        return Reference(*matrix_,position_);
    }

    Matrix* matrix_;//no matrix_closure here to ensure easy usage
    std::size_t position_;
};

}

///\brief Represents a Matrix as a vector of rows.
template<class Matrix>
class matrix_row_vector {
public:
    typedef ublas::matrix_row<Matrix> value_type;
    typedef ublas::matrix_row<Matrix> reference;
    typedef ublas::matrix_row<Matrix const> const_reference;

    typedef ublas::detail::matrix_vector_iterator<Matrix, ublas::matrix_row<Matrix> > iterator;
    typedef ublas::detail::matrix_vector_iterator<Matrix const, ublas::matrix_row<Matrix const> const> const_iterator;
    typedef boost::reverse_iterator<iterator> reverse_iterator;
    typedef boost::reverse_iterator<const_iterator> const_reverse_iterator;

    typedef typename boost::iterator_difference<iterator>::type difference_type;
    typedef typename Matrix::size_type size_type;

    matrix_row_vector(Matrix& matrix) :
        matrix_(matrix) {
    }


    iterator begin(){
        return iterator(matrix_, 0);
    }

    const_iterator begin() const {
        return const_iterator(matrix_, 0);
    }

    const_iterator cbegin() const {
        return begin();
    }

    iterator end() {
        return iterator(matrix_, matrix_.size1());
    }

    const_iterator end() const {
        return const_iterator(matrix_, matrix_.size1());
    }

    const_iterator cend() const {
        return end();
    }

    reverse_iterator rbegin() {
        return reverse_iterator(end());
    }

    const_reverse_iterator rbegin() const {
        return const_reverse_iterator(end());
    }

    const_reverse_iterator crbegin() const {
        return rbegin();
    }  

    reverse_iterator rend() {
        return reverse_iterator(begin());
    }

    const_reverse_iterator rend() const {
        return const_reverse_iterator(begin());
    }

    const_reverse_iterator crend() const {
        return end();
    }

    value_type operator()(difference_type index) const {
        return value_type(matrix_, index);
    }

    reference operator[](difference_type index){
        return reference(matrix_, index);
    }

    const_reference operator[](difference_type index) const {
        return const_reference(matrix_, index);
    }

    size_type size() const {
        return matrix_.size1();
    }

    void resize(size_type size, bool preserve = true) {
        matrix_.resize(size, matrix_.size2(), preserve);
    }

private:
    Matrix& matrix_;
};


///\brief convenience function to create matrix_row_vector.
template<class Matrix>
matrix_row_vector<Matrix> make_row_vector(matrix_expression<Matrix>& matrix){
    return matrix_row_vector<Matrix>(matrix());
}


///\brief convenience function to create matrix_row_vector.
template<class Matrix>
matrix_row_vector<Matrix const> make_row_vector(matrix_expression<Matrix> const& matrix){
    return matrix_row_vector<Matrix const>(matrix());
}


///\brief Represents a Matrix as a vector of rows.
template<class Matrix>
class matrix_column_vector
{
public:
    typedef ublas::matrix_column<Matrix> value_type;
    typedef ublas::matrix_column<Matrix> reference;
    typedef const ublas::matrix_column<Matrix const> const_reference;

    typedef ublas::detail::matrix_vector_iterator<Matrix, ublas::matrix_column<Matrix> > iterator;
    typedef ublas::detail::matrix_vector_iterator<Matrix const, ublas::matrix_column<Matrix const> const > const_iterator;
    typedef boost::reverse_iterator<iterator> reverse_iterator;
    typedef boost::reverse_iterator<const_iterator> const_reverse_iterator;

    typedef typename boost::iterator_difference<iterator>::type difference_type;
    typedef typename Matrix::size_type size_type;

    matrix_column_vector(Matrix& matrix) :
        matrix_(matrix){
    }

    iterator begin() {
        return iterator(matrix_, 0);
    }

    const_iterator begin() const {
        return const_iterator(matrix_, 0);
    }

    const_iterator cbegin() const {
        return begin();
    }

    iterator end() {
        return iterator(matrix_, matrix_.size2());
    }

    const_iterator end() const {
        return const_iterator(matrix_, matrix_.size2());
    }

    const_iterator cend() const {
        return end();
    }

    reverse_iterator rbegin() {
        return reverse_iterator(end());
    }

    const_reverse_iterator rbegin() const {
        return const_reverse_iterator(end());
    }

    const_reverse_iterator crbegin() const {
        return rbegin();
    } 

    reverse_iterator rend() {
        return reverse_iterator(begin());
    }

    const_reverse_iterator rend() const {
        return const_reverse_iterator(begin());
    }

    const_reverse_iterator crend() const {
        return rend();
    }

    value_type operator()(difference_type index) const {
        return value_type(matrix_, index);
    }

    reference operator[](difference_type index) {
        return reference(matrix_, index);
    }

    const_reference operator[](difference_type index) const {
        return const_reference(matrix_, index);
    }

    size_type size() const {
        return matrix_.size2();
    }

    void resize(size_type size, bool preserve = true) {
        matrix_.resize(matrix_.size1(), size, preserve);
    }

private:
    Matrix& matrix_;
};


///\brief convenience function to create matrix_column_vector.
template<class Matrix>
matrix_column_vector<Matrix> make_column_vector(matrix_expression<Matrix>& matrix){
    return matrix_column_vector<Matrix>(matrix());
}


///\brief convenience function to create matrix_column_vector.
template<class Matrix>
matrix_column_vector<Matrix const> make_column_vector(matrix_expression<Matrix> const& matrix){
    return matrix_column_vector<Matrix const>(matrix());
}

}}}

#endif
