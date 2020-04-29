//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//


/// \file tensor.hpp Definition for the tensor template class

#ifndef BOOST_UBLAS_TENSOR_IMPL_HPP
#define BOOST_UBLAS_TENSOR_IMPL_HPP

#include <initializer_list>

#include <boost/numeric/ublas/tensor/algorithms.hpp>
#include <boost/numeric/ublas/tensor/expression.hpp>
#include <boost/numeric/ublas/tensor/expression_evaluation.hpp>
#include <boost/numeric/ublas/tensor/fixed_rank_extents.hpp>
#include <boost/numeric/ublas/tensor/static_extents.hpp>
#include <boost/numeric/ublas/tensor/dynamic_extents.hpp>
#include <boost/numeric/ublas/tensor/strides.hpp>
#include <boost/numeric/ublas/tensor/index.hpp>
#include <boost/numeric/ublas/tensor/type_traits.hpp>

namespace boost { namespace numeric { namespace ublas {

///** \brief Base class for Tensor container models
// *
// * it does not model the Tensor concept but all derived types should.
// * The class defines a common base type and some common interface for all
// * statically derived Tensor classes
// * We implement the casts to the statically derived type.
// */
//template<class C>
//class tensor_container:
//  public detail::tensor_expression<C>
//{
//public:
// static const unsigned complexity = 0;
// typedef C container_type;
// typedef tensor_tag type_category;

// inline
// const container_type &operator () () const {
//  return *static_cast<const container_type *> (this);
//  }
//  inline
//  container_type &operator () () {
//      return *static_cast<container_type *> (this);
//  }
//};

template< class T >
class basic_tensor:
        public detail::tensor_expression< basic_tensor<T>,basic_tensor<T> >
{

    using self_type                 = basic_tensor<T>;

public:

    template<class derived_type>
    using tensor_expression_type    = detail::tensor_expression<self_type,derived_type>;

    template<class derived_type>
    using matrix_expression_type    = matrix_expression<derived_type>;

    template<class derived_type>
    using vector_expression_type    = vector_expression<derived_type>;

    using super_type                = tensor_expression_type<self_type>;

    using array_type                = typename tensor_traits<T>::container_type;
    using layout_type               = typename tensor_traits<T>::layout_type;


    using size_type                 = typename array_type::size_type;
    using difference_type           = typename array_type::difference_type;
    using value_type                = typename array_type::value_type;

    using reference                 = typename array_type::reference;
    using const_reference           = typename array_type::const_reference;

    using pointer                   = typename array_type::pointer;
    using const_pointer             = typename array_type::const_pointer;

    using iterator                  = typename array_type::iterator;
    using const_iterator            = typename array_type::const_iterator;

    using reverse_iterator          = typename array_type::reverse_iterator;
    using const_reverse_iterator    = typename array_type::const_reverse_iterator;

    using tensor_temporary_type     = self_type;
    using storage_category          = dense_tag;
    using container_tag             = typename tensor_traits<T>::container_tag;

    using extents_type              = typename tensor_traits<T>::extents_type;
    using strides_type              = strides_t<extents_type,layout_type>;

    using matrix_type               = matrix<value_type,layout_type, std::vector<value_type> >;
    using vector_type               = vector<value_type, std::vector<value_type> >;

    static_assert( std::is_same<layout_type,first_order>::value || 
                   std::is_same<layout_type,last_order >::value, 
                   "boost::numeric::basic_tensor template class only supports first- or last-order storage formats.");
protected:
    
    /** @brief Constructs a basic_tensor.
     *
     * @note the basic_tensor is empty.
     * @note the basic_tensor needs to reshaped for further use.
     *
     */
    inline
    constexpr basic_tensor ()
        : tensor_expression_type<self_type>() // container_type
        , extents_()
        , strides_()
    {}

    constexpr basic_tensor( extents_type const& e, dynamic_tensor_tag )
        : tensor_expression_type<self_type>()
        , extents_(e)
        , strides_(extents_)
        , data_( product(extents_) )
    {}

    constexpr basic_tensor( extents_type const& e, static_tensor_tag )
        : tensor_expression_type<self_type>()
        , extents_(e)
        , strides_(extents_)
    {}

    /** @brief Constructs a basic_tensor with an initializer list for dynamic_extents
     *
     * By default, its elements are initialized to 0.
     *
     * @code basic_tensor<float> A{4,2,3}; @endcode
     *
     * @param l initializer list for setting the dimension extents of the basic_tensor
     */
    explicit inline
    basic_tensor (std::initializer_list<size_type> l)
        : basic_tensor( std::move(l), container_tag{} )
    {}

    /** @brief Constructs a basic_tensor with a \c shape
     *
     * By default, its elements are initialized to 0.
     *
     * @code basic_tensor<float> A{extents{4,2,3}}; @endcode
     *
     * @param s initial basic_tensor dimension extents
     */
    explicit inline
    basic_tensor (extents_type const& s)
        : basic_tensor( s, container_tag{} )
    {}

    /** @brief Constructs a basic_tensor with a \c shape
     *
     * By default, its elements are initialized to 0.
     *
     * @code basic_tensor<float> A{extents{4,2,3}}; @endcode
     *
     * @param s initial basic_tensor dimension extents
     * @param i initial basic_tensor with this value
     */
    explicit inline
    basic_tensor (extents_type const& s, value_type const& i)
        : basic_tensor( s, container_tag{} )
    {
        std::fill(begin(),end(),i);
    }

    /** @brief Constructs a basic_tensor with a \c shape and initiates it with one-dimensional data
     *
     * @code basic_tensor<float> A{extents{4,2,3}, array }; @endcode
     *
     *
     *  @param s initial basic_tensor dimension extents
     *  @param a container of \c array_type that is copied according to the storage layout
     */
    basic_tensor (extents_type const& s, const array_type &a)
        : basic_tensor( s, container_tag{} )
    {
        if( product(extents_) != a.size() ){
            throw std::runtime_error("boost::numeric::ublas::basic_tensor(extents_type,array_type): array size mismatch with extents");
        }
        std::copy(a.begin(),a.end(),begin());
    }

    // /** @brief Constructs a basic_tensor using a shape tuple and initiates it with a value.
    //  *
    //  *  @code basic_tensor<float> A{extents{4,2,3}, 1 }; @endcode
    //  *
    //  *  @param e initial basic_tensor dimension extents
    //  *  @param i initial value of all elements of type \c value_type
    //  */
    // template<class U = E>
    // inline
    // basic_tensor (shape_t<typename extents_type::value_type, dynamic_rank> const& e, const value_type &i, typename std::enable_if<detail::is_static_extents<U>::value>::type* = nullptr)
    //  : tensor_expression_type<self_type>() //tensor_container<self_type> ()
    //  , extents_ (e.begin(),e.end())
    //  , strides_ (extents_)
    //  , data_    (product(extents_), i)
    // {}


public:

    /** @brief Constructs a basic_tensor with another basic_tensor with a different layout
     *
     * @param other basic_tensor with a different layout to be copied.
     */
    template<typename OtherTensor>
    basic_tensor (const basic_tensor<OtherTensor> &other)
        : basic_tensor( other.extents(), container_tag{} )
    { 
        copy(this->rank(), this->extents().data(),
                this->data(), this->strides().data(),
                other.data(), other.strides().data());
        
    }


    /** @brief Constructs a basic_tensor with an basic_tensor expression
     *
     * @code basic_tensor<float> A = B + 3 * C; @endcode
     *
     * @note type must be specified of basic_tensor must be specified.
     * @note dimension extents are extracted from tensors within the expression.
     *
     * @param expr basic_tensor expression
     * @param size basic_tensor expression
     */
    template<typename derived_type>
    basic_tensor (const tensor_expression_type<derived_type> &expr)
        : basic_tensor( detail::retrieve_extents(expr), container_tag{} )
    {
        static_assert( detail::has_tensor_types<self_type, tensor_expression_type<derived_type>>::value,
                                     "Error in boost::numeric::ublas::basic_tensor: expression does not contain a basic_tensor. cannot retrieve shape.");
        detail::eval( *this, expr );
    }

    /** @brief Constructs a basic_tensor with a matrix expression
     *
     * @code basic_tensor<float> A = B + 3 * C; @endcode
     *
     * @note matrix expression is evaluated and pushed into a temporary matrix before assignment.
     * @note extents are automatically extracted from the temporary matrix
     *
     * @param expr matrix expression
     */
    template<class derived_type>
    basic_tensor (const matrix_expression_type<derived_type> &expr)
        : basic_tensor(  matrix_type ( expr )  )
    {
    }

    /** @brief Constructs a basic_tensor with a vector expression
     *
     * @code basic_tensor<float> A = b + 3 * b; @endcode
     *
     * @note matrix expression is evaluated and pushed into a temporary matrix before assignment.
     * @note extents are automatically extracted from the temporary matrix
     *
     * @param expr vector expression
     */
    template<class derived_type>
    basic_tensor (const vector_expression_type<derived_type> &expr)
        : basic_tensor(  vector_type ( expr )  )
    {
    }


    /** @brief Constructs a basic_tensor from another basic_tensor
     *
     *  @param v basic_tensor to be copied.
     */
    inline
    basic_tensor (const basic_tensor &v)
        : tensor_expression_type<self_type>()
        , extents_ (v.extents_)
        , strides_ (v.strides_)
        , data_    (v.data_   )
    {}



    /** @brief Constructs a basic_tensor from another basic_tensor
     *
     *  @param v basic_tensor to be moved.
     */
    inline
    basic_tensor (basic_tensor &&v)
        : tensor_expression_type<self_type>() //tensor_container<self_type> ()
        , extents_ (std::move(v.extents_))
        , strides_ (std::move(v.strides_))
        , data_    (std::move(v.data_   ))
    {}

    /** @brief Evaluates the tensor_expression and assigns the results to the basic_tensor
     *
     * @code A = B + C * 2;  @endcode
     *
     * @note rank and dimension extents of the tensors in the expressions must conform with this basic_tensor.
     *
     * @param expr expression that is evaluated.
     */
    template<class derived_type>
    basic_tensor &operator = (const tensor_expression_type<derived_type> &expr)
    {
        detail::eval(*this, expr);
        return *this;
    }

    basic_tensor& operator=(basic_tensor other)
    {
        swap (*this, other);
        return *this;
    }

    basic_tensor& operator=(const_reference v)
    {
        std::fill(this->begin(), this->end(), v);
        return *this;
    }

    /** @brief Returns true if the basic_tensor is empty (\c size==0) */
    inline
    bool empty () const {
        return this->data_.empty();
    }


    /** @brief Returns the size of the basic_tensor */
    inline
    size_type size () const {
        return this->data_.size ();
    }

    /** @brief Returns the size of the basic_tensor */
    inline
    size_type size (size_type r) const {
        return this->extents_.at(r);
    }

    /** @brief Returns the number of dimensions/modes of the basic_tensor */
    inline
    size_type rank () const {
        return this->extents_.size();
    }

    /** @brief Returns the number of dimensions/modes of the basic_tensor */
    inline
    size_type order () const {
        return this->extents_.size();
    }

    /** @brief Returns the strides of the basic_tensor */
    inline
    strides_type const& strides () const {
        return this->strides_;
    }

    /** @brief Returns the extents of the basic_tensor */
    inline
    extents_type const& extents () const {
        return this->extents_;
    }


    /** @brief Returns a \c const reference to the container. */
    inline
    const_pointer data () const {
        return this->data_.data();
    }

    /** @brief Returns a \c const reference to the container. */
    inline
    pointer data () {
        return this->data_.data();
    }

    /** @brief Element access using a single index.
     *
     *  @code auto a = A[i]; @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size()
     */
    inline
    const_reference operator [] (size_type i) const {
        return this->data_[i];
    }

    /** @brief Element access using a single index.
     *
     *  @code auto a = A[i]; @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size()
     */
    inline
    reference operator [] (size_type i) {
        return this->data_[i];
    }

    /** @brief Element access using a multi-index or single-index.
     *
     *
     *  @code auto a = A.at(i,j,k); @endcode or
     *  @code auto a = A.at(i);     @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) == 0, else 0<= i < this->size(0)
     *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r < this->rank()
     */
    template<class ... size_types>
    inline
    const_reference at (size_type i, size_types ... is) const {
        if constexpr (sizeof...(is) == 0)
            return this->data_[i];
        else
            return this->data_[detail::access<0ul>(size_type(0),this->strides_,i,std::forward<size_types>(is)...)];
    }

    /** @brief Element access using a multi-index or single-index.
     *
     *
     *  @code A.at(i,j,k) = a; @endcode or
     *  @code A.at(i) = a;     @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) == 0, else 0<= i < this->size(0)
     *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r < this->rank()
     */
    template<class ... size_types>
    inline
    reference at (size_type i, size_types ... is) {
        if constexpr (sizeof...(is) == 0)
            return this->data_[i];
        else{
            auto temp = detail::access<0ul>(size_type(0),this->strides_,i,std::forward<size_types>(is)...);
            return this->data_[temp];
            }
    }

    /** @brief Element access using a single index.
     *
     *
     *  @code A(i) = a; @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size()
     */
    inline
    const_reference operator()(size_type i) const {
        return this->data_[i];
    }


    /** @brief Element access using a single index.
     *
     *  @code A(i) = a; @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size()
     */
    inline
    reference operator()(size_type i){
        return this->data_[i];
    }

    /** @brief Generates a basic_tensor index for basic_tensor contraction
     *
     *
     *  @code auto Ai = A(_i,_j,k); @endcode
     *
     *  @param i placeholder
     *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r < this->rank()
     */
    template<std::size_t I, class ... index_types>
    inline
    decltype(auto) operator() (index::index_type<I> p, index_types ... ps) const
    {
        constexpr auto N = sizeof...(ps)+1;
        if( N != this->rank() )
            throw std::runtime_error("Error in boost::numeric::ublas::operator(): size of provided index_types does not match with the rank.");

        return std::make_pair( std::cref(*this),  std::make_tuple( p, std::forward<index_types>(ps)... ) );
    }

    /** @brief Reshapes the basic_tensor
     *
     *
     * (1) @code A.reshape(extents{m,n,o});     @endcode or
     * (2) @code A.reshape(extents{m,n,o},4);   @endcode
     *
     * If the size of this smaller than the specified extents than
     * default constructed (1) or specified (2) value is appended.
     *
     * @note rank of the basic_tensor might also change.
     *
     * @param e extents with which the basic_tensor is reshaped.
     * @param v value which is appended if the basic_tensor is enlarged.
     */
    inline
    void reshape (extents_type const& e, value_type v = value_type{})
    {
        static_assert(is_dynamic_v<extents_type>,
            "Error in boost::numeric::ublas::basic_tensor: static extents cannot be reshaped");
        this->extents_ = e;
        this->strides_ = strides_type(this->extents_);

        auto p = product(extents_);
        if(p != this->size())
            this->data_.resize (p, v);
    }

    friend void swap(basic_tensor& lhs, basic_tensor& rhs) {
        std::swap(lhs.data_   , rhs.data_   );
        std::swap(lhs.extents_, rhs.extents_);
        std::swap(lhs.strides_, rhs.strides_);
    }


    /// \brief return an iterator on the first element of the basic_tensor
    inline
    const_iterator begin () const {
        return data_.begin ();
    }

    /// \brief return an iterator on the first element of the basic_tensor
    inline
    const_iterator cbegin () const {
        return data_.cbegin ();
    }

    /// \brief return an iterator after the last element of the basic_tensor
    inline
    const_iterator end () const {
        return data_.end();
    }

    /// \brief return an iterator after the last element of the basic_tensor
    inline
    const_iterator cend () const {
        return data_.cend ();
    }

    /// \brief Return an iterator on the first element of the basic_tensor
    inline
    iterator begin () {
        return data_.begin();
    }

    /// \brief Return an iterator at the end of the basic_tensor
    inline
    iterator end () {
        return data_.end();
    }

    /// \brief Return a const reverse iterator before the first element of the reversed basic_tensor (i.e. end() of normal basic_tensor)
    inline
    const_reverse_iterator rbegin () const {
        return data_.rbegin();
    }

    /// \brief Return a const reverse iterator before the first element of the reversed basic_tensor (i.e. end() of normal basic_tensor)
    inline
    const_reverse_iterator crbegin () const {
        return data_.crbegin();
    }

    /// \brief Return a const reverse iterator on the end of the reverse basic_tensor (i.e. first element of the normal basic_tensor)
    inline
    const_reverse_iterator rend () const {
        return data_.rend();
    }

    /// \brief Return a const reverse iterator on the end of the reverse basic_tensor (i.e. first element of the normal basic_tensor)
    inline
    const_reverse_iterator crend () const {
        return data_.crend();
    }

    /// \brief Return a const reverse iterator before the first element of the reversed basic_tensor (i.e. end() of normal basic_tensor)
    inline
    reverse_iterator rbegin () {
        return data_.rbegin();
    }

    /// \brief Return a const reverse iterator on the end of the reverse basic_tensor (i.e. first element of the normal basic_tensor)
    inline
    reverse_iterator rend () {
        return data_.rend();
    }

protected:

    extents_type extents_;
    strides_type strides_;
    array_type data_;
};

}}} // namespaces


namespace boost::numeric::ublas{
    template<typename T, typename...Ts>
    struct tensor_rebind<basic_tensor<T>,Ts...>{
        using type = tensor_rebind_t<T,Ts...>;
    };

} // namespace boost::numeric::ublas::detail



#endif
