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


/// \file tensor_core.hpp Definition for the tensor template class

#ifndef BOOST_UBLAS_TENSOR_CORE_IMPL_HPP
#define BOOST_UBLAS_TENSOR_CORE_IMPL_HPP

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
#include <boost/numeric/ublas/tensor/tags.hpp>

namespace boost::numeric::ublas {

template< class T >
class tensor_core:
        public detail::tensor_expression< tensor_core<T>,tensor_core<T> >
{

    using self_type                 = tensor_core<T>;

public:
    using tensor_traits             = T;

    template<class derived_type>
    using tensor_expression_type    = detail::tensor_expression<self_type,derived_type>;

    template<class derived_type>
    using matrix_expression_type    = matrix_expression<derived_type>;

    template<class derived_type>
    using vector_expression_type    = vector_expression<derived_type>;

    using super_type                = tensor_expression_type<self_type>;
    using storage_traits_type       = typename tensor_traits::storage_traits_type;

    using array_type                = typename storage_traits_type::array_type;
    using layout_type               = typename tensor_traits::layout_type;


    using size_type                 = typename storage_traits_type::size_type;
    using difference_type           = typename storage_traits_type::difference_type;
    using value_type                = typename storage_traits_type::value_type;

    using reference                 = typename storage_traits_type::reference;
    using const_reference           = typename storage_traits_type::const_reference;

    using pointer                   = typename storage_traits_type::pointer;
    using const_pointer             = typename storage_traits_type::const_pointer;

    using iterator                  = typename storage_traits_type::iterator;
    using const_iterator            = typename storage_traits_type::const_iterator;

    using reverse_iterator          = typename storage_traits_type::reverse_iterator;
    using const_reverse_iterator    = typename storage_traits_type::const_reverse_iterator;

    using tensor_temporary_type     = self_type;
    using storage_category          = dense_tag;
    using container_tag             = typename storage_traits_type::container_tag;
    using resizable_tag             = typename storage_traits_type::resizable_tag;

    using extents_type              = typename tensor_traits::extents_type;
    using strides_type              = typename tensor_traits::strides_type;

    using matrix_type               = matrix<value_type,layout_type, std::vector<value_type> >;
    using vector_type               = vector<value_type, std::vector<value_type> >;
    
    /** @brief Constructs a tensor_core.
     *
     * @note the tensor_core is empty.
     * @note the tensor_core needs to reshaped for further use.
     *
     */
    inline
    constexpr tensor_core ()
    {
        if constexpr( is_static_v<extents_type> ){
            auto temp = tensor_core(extents_type{},resizable_tag{});
            swap(*this,temp);
        }
    }

    constexpr tensor_core( extents_type e, [[maybe_unused]] storage_resizable_container_tag t )
        : tensor_expression_type<self_type>()
        , extents_(std::move(e))
        , strides_(extents_)
        , data_( product(extents_) )
    {}

    constexpr tensor_core( extents_type e, [[maybe_unused]] storage_static_container_tag t )
        : tensor_expression_type<self_type>()
        , extents_(std::move(e))
        , strides_(extents_)
    {
        if ( data_.size() < product(extents_) ){
            throw std::length_error("boost::numeric::ublas::tensor_core(extents_type const&, storage_static_container_tag): "
                "size of requested storage exceeds the current container size"
            );
        }
    }

    /** @brief Constructs a tensor_core with an initializer list for dynamic_extents
     *
     * By default, its elements are initialized to 0.
     *
     * @code tensor_core<float> A{4,2,3}; @endcode
     *
     * @param l initializer list for setting the dimension extents of the tensor_core
     */
    template<typename U = extents_type,
        typename = std::enable_if_t<is_dynamic_v<U>>
    >
    explicit inline
    tensor_core (std::initializer_list<size_type> l)
        : tensor_core( std::move( extents_type( std::move(l) ) ), resizable_tag{} )
    {}

    /** @brief Constructs a tensor_core with a \c shape
     *
     * By default, its elements are initialized to 0.
     *
     * @code tensor_core<float> A{extents{4,2,3}}; @endcode
     *
     * @param s initial tensor_core dimension extents
     */
    template<typename U = extents_type,
        typename = std::enable_if_t<is_dynamic_v<U>>
    >
    explicit inline
    tensor_core (extents_type s)
        : tensor_core( std::move(s), resizable_tag{} )
    {}

    /** @brief Constructs a tensor_core with a \c shape
     *
     * By default, its elements are initialized to 0.
     *
     * @code tensor_core<float> A{extents{4,2,3}}; @endcode
     *
     * @param s initial tensor_core dimension extents
     * @param i initial tensor_core with this value
     */
    template<typename U = extents_type,
        typename = std::enable_if_t<is_dynamic_v<U>>
    >
    explicit inline
    tensor_core (extents_type s, value_type const& i)
        : tensor_core( std::move(s), resizable_tag{} )
    {
        std::fill(begin(),end(),i);
    }

    /** @brief Constructs a tensor_core with a \c shape
     *
     * By default, its elements are initialized to 0.
     *
     * @code tensor_core<float> A{}; @endcode
     *
     * @param i initial tensor_core with this value
     */
    template<typename U = extents_type,
        typename = std::enable_if_t<is_static_v<U>>
    >
    explicit inline
    tensor_core (value_type const& i)
        : tensor_core()
    {
        std::fill(begin(),end(),i);
    }

    /** @brief Constructs a tensor_core with a \c shape and initiates it with one-dimensional data
     *
     * @code tensor_core<float> A{extents{4,2,3}, array }; @endcode
     *
     *
     *  @param s initial tensor_core dimension extents
     *  @param a container of \c array_type that is copied according to the storage layout
     */
    template<typename U = extents_type,
        typename = std::enable_if_t<is_dynamic_v<U>>
    >
    inline
    tensor_core (extents_type s, const array_type &a)
        : tensor_core( std::move(s), resizable_tag{} )
    {
        if( size() != a.size() ){
            throw std::runtime_error("boost::numeric::ublas::tensor_core(extents_type,array_type): "
                "array size mismatch with extents"
            );
        }
        std::copy(a.begin(),a.end(),begin());
    }

    /** @brief Constructs a tensor_core with a \c shape and initiates it with one-dimensional data
     *
     * @code tensor_core<float> A{ array }; @endcode
     *
     *  @param a container of \c array_type that is copied according to the storage layout
     */
    template<typename U = extents_type,
        typename = std::enable_if_t<is_static_v<U>>
    >
    inline
    tensor_core (const array_type &a)
        : tensor_core()
    {
        if( size() != a.size() ){
            throw std::runtime_error("boost::numeric::ublas::tensor_core(extents_type,array_type): "
                "array size mismatch with extents"
            );
        }
        std::copy(a.begin(),a.end(),begin());
    }


    /** @brief Constructs a tensor_core with another tensor_core with a different layout
     *
     * @param other tensor_core with a different layout to be copied.
     */
    template<typename OtherTensor>
    tensor_core (const tensor_core<OtherTensor> &other)
        : tensor_core( other.extents(), resizable_tag{} )
    { 
        copy(this->rank(), this->extents().data(),
                this->data(), this->strides().data(),
                other.data(), other.strides().data());
        
    }


    /** @brief Constructs a tensor_core with an tensor_core expression
     *
     * @code tensor_core<float> A = B + 3 * C; @endcode
     *
     * @note type must be specified of tensor_core must be specified.
     * @note dimension extents are extracted from tensors within the expression.
     *
     * @param expr tensor_core expression
     * @param size tensor_core expression
     */
    template<typename other_tensor,typename derived_type>
    tensor_core (const detail::tensor_expression<other_tensor,derived_type> &expr)
        : tensor_core( detail::retrieve_extents(expr), resizable_tag{} )
    {
        static_assert(is_valid_tensor_v<other_tensor>,
            "boost::numeric::ublas::tensor_core(tensor_expression<other_tensor, derived_type> const&) : "
            "other_tensor should be a valid tensor type"
        );
        
        static_assert(std::is_same_v<value_type, typename other_tensor::value_type>,
            "boost::numeric::ublas::tensor_core(tensor_expression<other_tensor, derived_type> const&) : "
            "LHS and RHS should have the same value type"
        );

        detail::eval( *this, expr );
    }

    constexpr tensor_core( matrix_type const& v )
        : tensor_core()
    {
        if constexpr( is_dynamic_v< extents_type > ){
            auto temp = tensor_core(extents_type{v.size1(), v.size2()});
            swap(*this,temp);
        }

        if constexpr( is_static_rank_v<extents_type> ){
            static_assert( extents_type::_size == 2ul, 
                "boost::numeric::ublas::tensor_core(const matrix &v)"
                " : the rank of extents is not correct, it should be of the rank 2"
            );
        }else{
            if( extents_.size() != 2ul ){
                throw std::runtime_error(
                    "boost::numeric::ublas::tensor_core(const matrix &v)"
                    " : the rank of extents is not correct, it should be of the rank 2"
                );
            }
        }


        if( extents_[0] != v.size1() || extents_[1] != v.size2() ){
            throw std::runtime_error(
                "boost::numeric::ublas::tensor_core(const matrix &v)"
                " : please set the extents properly, the extents should contain the row and col of the matrix"
            );
        }

        std::copy(v.data().begin(), v.data().end(), data_.begin());
    }

    constexpr tensor_core( matrix_type && v )
        : tensor_core()
    {
        if constexpr( is_dynamic_v< extents_type > ){
            auto temp = tensor_core(extents_type{v.size1(), v.size2()});
            swap(*this,temp);
        }

        if constexpr( is_static_rank_v<extents_type> ){
            static_assert( extents_type::_size == 2ul, 
                "boost::numeric::ublas::tensor_core(matrix &&v)"
                " : the rank of extents is not correct, it should be of the rank 2"
            );
        }else{
            if( extents_.size() != 2ul ){
                throw std::runtime_error(
                    "boost::numeric::ublas::tensor_core(matrix &&v)"
                    " : the rank of extents is not correct, it should be of the rank 2"
                );
            }
        }

        if( extents_[0] != v.size1() || extents_[1] != v.size2() ){
            throw std::runtime_error(
                "boost::numeric::ublas::tensor_core(matrix &&v)"
                " : please set the extents properly, the extents should contain the row and col of the matrix"
            );
        }

        std::move(v.data().begin(), v.data().end(),data_.begin());
    }

    constexpr tensor_core (const vector_type &v)
        : tensor_core()
    {
        if constexpr( is_dynamic_v< extents_type > ){
            auto temp = tensor_core(extents_type{ v.size(), typename extents_type::value_type{1} });
            swap(*this,temp);
        }

        if constexpr( is_static_rank_v<extents_type> ){
            static_assert( extents_type::_size == 2ul, 
                "boost::numeric::ublas::tensor_core(const vector_type &v)"
                " : the rank of extents is not correct, it should be of the rank 2"
            );
        }else{
            if( extents_.size() != 2ul ){
                throw std::runtime_error(
                    "boost::numeric::ublas::tensor_core(const vector_type &v)"
                    " : the rank of extents is not correct, it should be of the rank 2"
                );
            }
        }

        if( extents_[0] != v.size() || extents_[1] != 1ul ){
            throw std::runtime_error(
                "boost::numeric::ublas::tensor_core(const vector_type &v)"
                " : please set the extents properly, the first extent should be the size of the vector and 1 for the second extent"
            );
        }

        std::copy(v.data().begin(), v.data().end(), data_.begin());
        
    }

    constexpr tensor_core (vector_type &&v)
        : tensor_core()
    {
        if constexpr( is_dynamic_v< extents_type > ){
            auto temp = tensor_core(extents_type{ v.size(), typename extents_type::value_type{1} });
            swap(*this,temp);
        }
        
        if constexpr( is_static_rank_v<extents_type> ){
            static_assert( extents_type::_size == 2ul, 
                "boost::numeric::ublas::tensor_core(vector_type &&v)"
                " : the rank of extents is not correct, it should be of the rank 2"
            );
        }else{
            if( extents_.size() != 2ul ){
                throw std::runtime_error(
                    "boost::numeric::ublas::tensor_core(vector_type &&v)"
                    " : the rank of extents is not correct, it should be of the rank 2"
                );
            }
        }

        if( extents_[0] != v.size() || extents_[1] != 1ul ){
            throw std::runtime_error(
                "boost::numeric::ublas::tensor_core(vector_type &&v)"
                " : please set the extents properly, the first extent should be the size of the vector and 1 for the second extent"
            );
        }

        std::move(v.data().begin(), v.data().end(),data_.begin());
        
    }

    /** @brief Constructs a tensor_core with a matrix expression
     *
     * @code tensor_core<float> A = B + 3 * C; @endcode
     *
     * @note matrix expression is evaluated and pushed into a temporary matrix before assignment.
     * @note extents are automatically extracted from the temporary matrix
     *
     * @param expr matrix expression
     */
    template<class derived_type>
    tensor_core (const matrix_expression_type<derived_type> &expr)
        : tensor_core(  matrix_type ( expr )  )
    {
    }

    /** @brief Constructs a tensor_core with a vector expression
     *
     * @code tensor_core<float> A = b + 3 * b; @endcode
     *
     * @note matrix expression is evaluated and pushed into a temporary matrix before assignment.
     * @note extents are automatically extracted from the temporary matrix
     *
     * @param expr vector expression
     */
    template<class derived_type>
    tensor_core (const vector_expression_type<derived_type> &expr)
        : tensor_core(  vector_type ( expr )  )
    {
    }


    /** @brief Constructs a tensor_core from another tensor_core
     *
     *  @param v tensor_core to be copied.
     */
    inline
    tensor_core (const tensor_core &v)
        : tensor_expression_type<self_type>()
        , extents_ (v.extents_)
        , strides_ (v.strides_)
        , data_    (v.data_   )
    {}



    /** @brief Constructs a tensor_core from another tensor_core
     *
     *  @param v tensor_core to be moved.
     */
    inline
    tensor_core (tensor_core &&v) noexcept
        : tensor_expression_type<self_type>() //tensor_container<self_type> ()
        , extents_ (std::move(v.extents_))
        , strides_ (std::move(v.strides_))
        , data_    (std::move(v.data_   ))
    {}


    /** @brief Move assignsment operator
     *
     *  @param v tensor_core to be moved.
     */
    inline
    tensor_core& operator=(tensor_core &&v) noexcept
    {
        swap(*this,v);
        return *this;
    }

    /// @brief Default destructor
    ~tensor_core() = default;

    /** @brief Evaluates the tensor_expression and assigns the results to the tensor_core
     *
     * @code A = B + C * 2;  @endcode
     *
     * @note rank and dimension extents of the tensors in the expressions must conform with this tensor_core.
     *
     * @param expr expression that is evaluated.
     */
    template<class derived_type>
    tensor_core &operator = (const tensor_expression_type<derived_type> &expr)
    {
        detail::eval(*this, expr);
        return *this;
    }

    tensor_core& operator=(tensor_core const& other)
    {
        tensor_core temp(other);
        swap (*this, temp);
        return *this;
    }

    constexpr tensor_core& operator=(const_reference v)
    {
        std::fill_n(this->begin(), this->size(), v);
        return *this;
    }

    /** @brief Returns true if the tensor_core is empty (\c size==0) */
    [[nodiscard]] inline
    constexpr bool empty () const noexcept{
        return this->data_.empty();
    }

    /** @brief Returns the upper bound or max size of the tensor_core */
    [[nodiscard]] inline
    constexpr size_type size() const noexcept{
        return this->data_.size();
    }

    /** @brief Returns the size of the tensor_core */
    [[nodiscard]] inline
    constexpr size_type size (size_type r) const {
        return this->extents_.at(r);
    }

    /** @brief Returns the number of dimensions/modes of the tensor_core */
    [[nodiscard]] inline
    constexpr size_type rank () const noexcept{
        return this->extents_.size();
    }

    /** @brief Returns the number of dimensions/modes of the tensor_core */
    [[nodiscard]] inline
    constexpr size_type order () const noexcept{
        return this->extents_.size();
    }

    /** @brief Returns the strides of the tensor_core */
    [[nodiscard]] inline
    constexpr strides_type const& strides () const noexcept{
        return this->strides_;
    }

    /** @brief Returns the extents of the tensor_core */
    [[nodiscard]] inline
    constexpr extents_type const& extents () const noexcept{
        return this->extents_;
    }

    /** @brief Returns the strides of the tensor_core */
    [[nodiscard]] inline
    constexpr strides_type& strides () noexcept{
        return this->strides_;
    }

    /** @brief Returns the extents of the tensor_core */
    [[nodiscard]] inline
    constexpr extents_type& extents () noexcept{
        return this->extents_;
    }
    
    /** @brief Returns a \c const reference to the container. */
    [[nodiscard]] inline
    constexpr const_pointer data () const noexcept{
        return this->data_.data();
    }

    /** @brief Returns a \c const reference to the container. */
    [[nodiscard]] inline
    constexpr pointer data () noexcept{
        return this->data_.data();
    }

    /** @brief Returns a \c const reference to the underlying container. */
    [[nodiscard]] inline
    constexpr array_type const& base () const noexcept{
        return data_;
    }

    /** @brief Returns a reference to the underlying container. */
    [[nodiscard]] inline
    constexpr array_type& base () noexcept{
        return data_;
    }

    /** @brief Element access using a single index.
     *
     *  @code auto a = A[i]; @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size()
     */
    [[nodiscard]] inline
    constexpr const_reference operator [] (size_type i) const {
        return this->data_[i];
    }

    /** @brief Element access using a single index.
     *
     *  @code auto a = A[i]; @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size()
     */
    [[nodiscard]] inline
    constexpr reference operator [] (size_type i) {
        return this->data_[i];
    }

    /** @brief Element access using a multi-index or single-index with bound checking
     *  and it throws the exception.
     *
     *  @code auto a = A.at(i,j,k); @endcode or
     *  @code auto a = A.at(i);     @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) == 0, else 0<= i < this->size(0)
     *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r < this->rank()
     */
    template<class ... Indices>
    [[nodiscard]] inline
    constexpr const_reference at (size_type i, Indices ... is) const {
        if constexpr( sizeof...(is) == 0ul ){
            return this->data_.at(i);
        }else{
            if( sizeof...(is) + 1 > strides_.size() ){
                throw std::runtime_error("Error in boost::numeric::ublas::at(size_type, Indices...): "
                    "number of variadic argument exceeds the strides size."
                );
            }
            static_assert(
                std::conjunction_v< std::is_convertible<Indices,size_type>... >,
                "boost::numeric::ublas::tensor_core::at(size_type,Indices...) : "
                "provided variadic argument is not convertible to tensor size_type"
            );
            using strides_value_type = typename strides_type::value_type;
            auto const idx = detail::access(this->strides_, 
                                            static_cast<strides_value_type>(i), 
                                            static_cast<strides_value_type>(is)...
                                            );
            return this->data_.at(idx);
        }
    }

    /** @brief Element access using a multi-index or single-index with bound checking
     *  and it throws the exception.
     *
     *
     *  @code A.at(i,j,k) = a; @endcode or
     *  @code A.at(i) = a;     @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) == 0, else 0<= i < this->size(0)
     *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r < this->rank()
     */
    template<class ... Indices>
    [[nodiscard]] inline
    constexpr reference at (size_type i, Indices ... is) {
        if constexpr( sizeof...(is) == 0ul ){
            return this->data_.at(i);
        }else{
            if( sizeof...(is) + 1 > strides_.size() ){
                throw std::runtime_error("Error in boost::numeric::ublas::at(size_type, Indices...): "
                    "number of variadic argument exceeds the strides size."
                );
            }
            static_assert(
                std::conjunction_v< std::is_convertible<Indices,size_type>... >,
                "boost::numeric::ublas::tensor_core::at(size_type,Indices...) : "
                "provided variadic argument is not convertible to tensor size_type"
            );
            using strides_value_type = typename strides_type::value_type;
            auto const idx = detail::access(this->strides_, 
                                            static_cast<strides_value_type>(i), 
                                            static_cast<strides_value_type>(is)...
                                            );
            return this->data_.at(idx);
        }
    }

    /** @brief Element access using a multi-index or single-index with no bound checking
     *  and it does not throw.
     *
     *
     *  @code auto a = A(i,j,k); @endcode or
     *  @code auto a = A(i);     @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) == 0, else 0<= i < this->size(0)
     *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r < this->rank()
     */
    template<class ... Indices>
    [[nodiscard]] inline
    constexpr const_reference operator() (size_type i, Indices ... is) const {
        if constexpr( sizeof...(is) == 0ul ){
            return this->data_[i];
        }else{
            static_assert(
                std::conjunction_v< std::is_convertible<Indices,size_type>... >,
                "boost::numeric::ublas::tensor_core::operator()(size_type,Indices...) : "
                "provided variadic argument is not convertible to tensor size_type"
            );
            using strides_value_type = typename strides_type::value_type;
            auto const idx = detail::access(this->strides_, 
                                            static_cast<strides_value_type>(i), 
                                            static_cast<strides_value_type>(is)...
                                            );
            return this->data_[idx];
        }
    }

    /** @brief Element access using a multi-index or single-index with no bound checking
     *  and it does not throw.
     *
     *
     *  @code A(i,j,k) = a; @endcode or
     *  @code A(i) = a;     @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) == 0, else 0<= i < this->size(0)
     *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r < this->rank()
     */
    template<class ... Indices>
    [[nodiscard]] inline
    constexpr reference operator() (size_type i, Indices ... is) {
        if constexpr( sizeof...(is) == 0ul ){
            return this->data_[i];
        }else{
            static_assert(
                std::conjunction_v< std::is_convertible<Indices,size_type>... >,
                "boost::numeric::ublas::tensor_core::operator()(size_type,Indices...) : "
                "provided variadic argument is not convertible to tensor size_type"
            );
            using strides_value_type = typename strides_type::value_type;
            auto const idx = detail::access(this->strides_, 
                                            static_cast<strides_value_type>(i), 
                                            static_cast<strides_value_type>(is)...
                                            );
            return this->data_[idx];
        }
    }

    /** @brief Generates a tensor_core index for tensor_core contraction
     *
     *
     *  @code auto Ai = A(_i,_j,k); @endcode
     *
     *  @param i placeholder
     *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r < this->rank()
     */
    template<std::size_t I, class ... index_types>
    [[nodiscard]] inline
    constexpr decltype(auto) operator() (index::index_type<I> p, index_types ... ps) const
    {
        constexpr auto N = sizeof...(ps)+1;
        if( N != this->rank() )
            throw std::runtime_error("Error in boost::numeric::ublas::operator(index::index_type<I>,index_types&&): "
                "size of provided index_types does not match with the rank."
            );

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
        static_assert(is_dynamic_v<extents_type> && is_dynamic_v<strides_type>,
            "Error in boost::numeric::ublas::basic_tensor::reshape(extents_type const&,value_type) : "
            "static extents or static strides cannot used inside reshape function"
        );

        this->extents_ = e;
        this->strides_ = strides_type(this->extents_);

        auto p = product(extents_);
        if constexpr( !std::is_same_v< resizable_tag, storage_resizable_container_tag > ){
            if( p != this->size() ){
                throw std::runtime_error(
                    "boost::numeric::ublas::basic_tensor::reshape(extents_type const&,value_type) : "
                    "cannot resize the non-resizable container, change the extents such a way that the product does not change"
                );
            }
        }else{
            if(p != this->size())
                this->data_.resize (p, v);
        }
    }

    friend void swap(tensor_core& lhs, tensor_core& rhs){
        std::swap(lhs.data_   , rhs.data_   );
        std::swap(lhs.extents_, rhs.extents_);
        std::swap(lhs.strides_, rhs.strides_);
    }


    /// \brief return an iterator on the first element of the tensor_core
    [[nodiscard]] inline
    constexpr const_iterator begin () const noexcept{
        return data_.begin ();
    }

    /// \brief return an iterator on the first element of the tensor_core
    [[nodiscard]] inline
    constexpr const_iterator cbegin () const noexcept{
        return data_.cbegin ();
    }

    /// \brief return an iterator after the last element of the tensor_core
    [[nodiscard]] inline
    constexpr const_iterator end () const noexcept{
        return data_.end();
    }

    /// \brief return an iterator after the last element of the tensor_core
    [[nodiscard]] inline
    constexpr const_iterator cend () const noexcept{
        return data_.cend ();
    }

    /// \brief Return an iterator on the first element of the tensor_core
    [[nodiscard]] inline
    constexpr iterator begin () noexcept{
        return data_.begin();
    }

    /// \brief Return an iterator at the end of the tensor_core
    [[nodiscard]] inline
    constexpr iterator end () noexcept{
        return data_.end();
    }

    /// \brief Return a const reverse iterator before the first element of the reversed tensor_core (i.e. end() of normal tensor_core)
    [[nodiscard]] inline
    constexpr const_reverse_iterator rbegin () const noexcept{
        return data_.rbegin();
    }

    /// \brief Return a const reverse iterator before the first element of the reversed tensor_core (i.e. end() of normal tensor_core)
    [[nodiscard]] inline
    constexpr const_reverse_iterator crbegin () const noexcept{
        return data_.crbegin();
    }

    /// \brief Return a const reverse iterator on the end of the reverse tensor_core (i.e. first element of the normal tensor_core)
    [[nodiscard]] inline
    constexpr const_reverse_iterator rend () const noexcept{
        return data_.rend();
    }

    /// \brief Return a const reverse iterator on the end of the reverse tensor_core (i.e. first element of the normal tensor_core)
    [[nodiscard]] inline
    constexpr const_reverse_iterator crend () const noexcept{
        return data_.crend();
    }

    /// \brief Return a const reverse iterator before the first element of the reversed tensor_core (i.e. end() of normal tensor_core)
    [[nodiscard]] inline
    constexpr reverse_iterator rbegin () noexcept{
        return data_.rbegin();
    }

    /// \brief Return a const reverse iterator on the end of the reverse tensor_core (i.e. first element of the normal tensor_core)
    [[nodiscard]] inline
    constexpr reverse_iterator rend () noexcept{
        return data_.rend();
    }

private:

    extents_type extents_;
    strides_type strides_;
    array_type data_;
};

} // namespaces

#endif
