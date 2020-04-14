//
// 	Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
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
#include <boost/numeric/ublas/tensor/storage.hpp>
#include <boost/numeric/ublas/tensor/expression.hpp>
#include <boost/numeric/ublas/tensor/expression_evaluation.hpp>
#include <boost/numeric/ublas/tensor/extents.hpp>
#include <boost/numeric/ublas/tensor/strides.hpp>
#include <boost/numeric/ublas/tensor/index.hpp>
#include <boost/numeric/ublas/tensor/detail/type_traits.hpp>

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
//		public detail::tensor_expression<C>
//{
//public:
//	static const unsigned complexity = 0;
//	typedef C container_type;
//	typedef tensor_tag type_category;

//	inline
//	const container_type &operator () () const {
//			return *static_cast<const container_type *> (this);
//	}
//	inline
//	container_type &operator () () {
//			return *static_cast<container_type *> (this);
//	}
//};



	/** @brief A dense tensor of values of type \c T.
	*
	* For a \f$n\f$-dimensional tensor \f$v\f$ and \f$0\leq i < n\f$ every element \f$v_i\f$ is mapped
	* to the \f$i\f$-th element of the container. A storage type \c A can be specified which defaults to \c unbounded_array.
	* Elements are constructed by \c A, which need not initialise their value.
	*
	* @tparam T type of the objects stored in the tensor (like int, double, complex,...)
	* @tparam A The type of the storage array of the tensor. Default is \c unbounded_array<T>. \c <bounded_array<T> and \c std::vector<T> can also be used
	*/
template<class T = float, class E = dynamic_extents<>, class F = first_order, class A = storage::dense_tensor::default_storage_t< T, E ,std::allocator<T>> >
class tensor:
		public detail::tensor_expression<tensor<T, E, F, A>,tensor<T, E, F, A>>
{

	static_assert( std::is_same<F,first_order>::value || 
				   std::is_same<F,last_order >::value, 
				   "boost::numeric::tensor template class only supports first- or last-order storage formats.");

	using self_type  = tensor<T, E, F, A>;
public:



	template<class derived_type>
	using tensor_expression_type = detail::tensor_expression<self_type,derived_type>;

	template<class derived_type>
	using matrix_expression_type = matrix_expression<derived_type>;

	template<class derived_type>
	using vector_expression_type = vector_expression<derived_type>;

	using super_type = tensor_expression_type<self_type>;

//	static_assert(std::is_same_v<tensor_expression_type<self_type>, detail::tensor_expression<tensor<T,E,F,A>,tensor<T,E,F,A>>>, "tensor_expression_type<self_type>");

	using array_type  = A;
	using layout_type = F;


	using size_type       = typename array_type::size_type;
	using difference_type = typename array_type::difference_type;
	using value_type      = typename array_type::value_type;

	using reference       = typename array_type::reference;
	using const_reference = typename array_type::const_reference;

	using pointer         = typename array_type::pointer;
	using const_pointer   = typename array_type::const_pointer;

	using iterator        = typename array_type::iterator;
	using const_iterator  = typename array_type::const_iterator;

	using reverse_iterator        = typename array_type::reverse_iterator;
	using const_reverse_iterator  = typename array_type::const_reverse_iterator;

	using tensor_temporary_type = self_type;
	using storage_category = dense_tag;

	using strides_type = strides_t<E,layout_type>;
	using extents_type = E;

	using matrix_type     = matrix<value_type,layout_type,std::vector<T>>;
	using vector_type     = vector<value_type,std::vector<T>>;


	/** @brief Constructs a tensor.
	 *
	 * @note the tensor is empty.
	 * @note the tensor needs to reshaped for further use.
	 *
	 */
	inline
	constexpr tensor ()
		: tensor_expression_type<self_type>() // container_type
		, extents_()
		, strides_()
	{
		resize(extents_);
	}


	/** @brief Constructs a tensor with an initializer list for dynamic_extents
	 *
	 * By default, its elements are initialized to 0.
	 *
	 * @code tensor<float> A{4,2,3}; @endcode
	 *
	 * @param l initializer list for setting the dimension extents of the tensor
	 */
	explicit inline
	tensor (std::initializer_list<size_type> l)
		: tensor_expression_type<self_type>()
		, extents_ (std::move(l))
		, strides_ (extents_)
	{
		resize(extents_);
	}

	/** @brief Constructs a tensor with a \c shape
	 *
	 * By default, its elements are initialized to 0.
	 *
	 * @code tensor<float> A{extents{4,2,3}}; @endcode
	 *
	 * @param s initial tensor dimension extents
	 */
	explicit inline
	tensor (extents_type const& s)
		: tensor_expression_type<self_type>() //tensor_container<self_type>()
		, extents_ (s)
		, strides_ (extents_)
	{
		resize(extents_);
	}


	/** @brief Constructs a tensor with a \c shape and initiates it with one-dimensional data
	 *
	 * @code tensor<float> A{extents{4,2,3}, array }; @endcode
	 *
	 *
	 *  @param s initial tensor dimension extents
	 *  @param a container of \c array_type that is copied according to the storage layout
	 */
	inline
	tensor (extents_type const& s, const array_type &a)
		: tensor_expression_type<self_type>() //tensor_container<self_type>()
		, extents_ (s)
		, strides_ (extents_)
		, data_    (a)
	{
		if(product(extents_) != this->data_.size())
			throw std::runtime_error("Error in boost::numeric::ublas::tensor: size of provided data and specified extents do not match.");
	}



	/** @brief Constructs a tensor using a shape tuple and initiates it with a value.
	 *
	 *  @code tensor<float> A{extents{4,2,3}, 1 }; @endcode
	 *
	 *  @param e initial tensor dimension extents
	 *  @param i initial value of all elements of type \c value_type
	 */
	inline
	tensor (extents_type const& e, const value_type &i)
		: tensor_expression_type<self_type>() //tensor_container<self_type> ()
		, extents_ (e)
		, strides_ (extents_)
	{
		resize(extents_);
		std::fill(begin(),end(),i);
	}

	// /** @brief Constructs a tensor using a shape tuple and initiates it with a value.
	//  *
	//  *  @code tensor<float> A{extents{4,2,3}, 1 }; @endcode
	//  *
	//  *  @param e initial tensor dimension extents
	//  *  @param i initial value of all elements of type \c value_type
	//  */
	// template<class U = E>
	// inline
	// tensor (shape_t<typename extents_type::value_type, dynamic_rank> const& e, const value_type &i, typename std::enable_if<detail::is_static_extents<U>::value>::type* = nullptr)
	// 	: tensor_expression_type<self_type>() //tensor_container<self_type> ()
	// 	, extents_ (e.begin(),e.end())
	// 	, strides_ (extents_)
	// 	, data_    (product(extents_), i)
	// {}



	/** @brief Constructs a tensor from another tensor
	 *
	 *  @param v tensor to be copied.
	 */
	inline
	tensor (const tensor &v)
		: tensor_expression_type<self_type>()
		, extents_ (v.extents_)
		, strides_ (v.strides_)
		, data_    (v.data_   )
	{}



	/** @brief Constructs a tensor from another tensor
	 *
	 *  @param v tensor to be moved.
	 */
	inline
	tensor (tensor &&v)
		: tensor_expression_type<self_type>() //tensor_container<self_type> ()
		, extents_ (std::move(v.extents_))
		, strides_ (std::move(v.strides_))
		, data_    (std::move(v.data_   ))
	{}

	/** @brief Constructs a tensor with a matrix
	 *
	 * \note Initially the tensor will be two-dimensional.
	 *
	 *  @param v matrix to be copied.
	 */
	inline
	tensor (const matrix_type &v)
		: tensor_expression_type<self_type>()
		, extents_ ()
		, strides_ ()
	{
		auto const sz = v.size1() * v.size2();
		if(sz){
			if constexpr (detail::is_static<extents_type>::value){
				static_assert(extents_type::_size == 2
					, "Error in boost::numeric::ublas::tensor(const matrix_type &v)"
											" : invalid extents size");
				if( !( extents_[0] == v.size1() || extents_[1] == v.size2() ) ){
					throw std::runtime_error("Error in boost::numeric::ublas::tensor(const matrix_type &v)"
											" : rank of extents not correct, please check!");
				}
			}else {
				if constexpr( detail::is_static_rank<extents_type>::value ){
					static_assert(extents_type::size() == 2, "Error in boost::numeric::ublas::tensor(const matrix_type &v)"
											" : rank of extents not correct, please check!");
				}
				resize(sz);
				extents_ = extents_type{v.size1(),v.size2()};
				strides_ = strides_type(extents_);
			}
			std::copy(v.data().begin(), v.data().end(),data_.begin());
		}
	}

	/** @brief Constructs a tensor with a matrix
	 *
	 * \note Initially the tensor will be two-dimensional.
	 *
	 *  @param v matrix to be moved.
	 */
	inline
	tensor (matrix_type &&v)
		: tensor_expression_type<self_type>()
		, extents_ {}
		, strides_ {}
		, data_    {}
	{

		auto const sz = v.size1() * v.size2();
		if(sz){
			if constexpr (detail::is_static<extents_type>::value){
				static_assert(extents_type::_size == 2
					, "Error in boost::numeric::ublas::tensor(const matrix_type &v)"
											" : invalid extents size");
				if( !( extents_[0] == v.size1() || extents_[1] == v.size2() ) ){
					throw std::runtime_error("Error in boost::numeric::ublas::tensor(const matrix_type &v)"
											" : rank of extents not correct, please check!");
				}
			}else {
				if constexpr( detail::is_static_rank<extents_type>::value ){
					static_assert(extents_type::size() == 2, "Error in boost::numeric::ublas::tensor(const matrix_type &v)"
											" : rank of extents not correct, please check!");
				}
				
				resize(sz);
				extents_ = extents_type{v.size1(),v.size2()};
				strides_ = strides_type(extents_);
			}
			for(auto i = size_type{}; i < sz; ++i){
				data_[i] = std::move(v.data()[i]);
			}
		}
	}

	/** @brief Constructs a tensor using a \c vector
	 *
	 * @note It is assumed that vector is column vector
	 * @note Initially the tensor will be one-dimensional.
	 *
	 *  @param v vector to be copied.
	 */
	inline
	tensor (const vector_type &v)
		: tensor_expression_type<self_type>()
		, extents_ ()
		, strides_ ()
	{
		auto const sz = v.size();
		if(sz){
			if constexpr (detail::is_static<extents_type>::value){
				static_assert(extents_type::_size == 2
					, "Error in boost::numeric::ublas::tensor(const matrix_type &v)"
											" : invalid extents size");
				if( !( extents_[0] == v.size() || extents_[1] == 1 ) ){
					throw std::runtime_error("Error in boost::numeric::ublas::tensor(const matrix_type &v)"
											" : rank of extents not correct, please check!");
				}
			}else{
				if constexpr( detail::is_static_rank<extents_type>::value ){
					static_assert(extents_type::size() == 2, "Error in boost::numeric::ublas::tensor(const vector &v)"
											" : rank of extents not correct, please check!");
				}
				
				resize(sz);
				extents_ = extents_type{v.size(),1};
				strides_ = strides_type(extents_);
			}
			std::copy(v.data().begin(), v.data().end(),data_.begin());
		}
		
	}

	/** @brief Constructs a tensor using a \c vector
	 *
	 *  @param v vector to be moved.
	 */
	inline
	tensor (vector_type &&v)
		: tensor_expression_type<self_type>()
		, extents_ {}
		, strides_ {}
		, data_    {}
	{
		auto const sz = v.size();
		if(sz){
			if constexpr (detail::is_static<extents_type>::value){
				static_assert(extents_type::_size == 2
					, "Error in boost::numeric::ublas::tensor(const matrix_type &v)"
											" : invalid extents size");
				if( !( extents_[0] == v.size() || extents_[1] == 1 ) ){
					throw std::runtime_error("Error in boost::numeric::ublas::tensor(const matrix_type &v)"
											" : rank of extents not correct, please check!");
				}
			}else{
				if constexpr( detail::is_static_rank<extents_type>::value ){
					static_assert(extents_type::size() == 2, "Error in boost::numeric::ublas::tensor(const vector &v)"
											" : rank of extents not correct, please check!");
				}
				
				resize(sz);
				extents_ = extents_type{v.size(),1};
				strides_ = strides_type(extents_);
			}
			
			for(auto i = size_type{}; i < sz; ++i){
				data_[i] = std::move(v.data()[i]);
			}
		}
	}




	/** @brief Constructs a tensor with another tensor with a different layout
	 *
	 * @param other tensor with a different layout to be copied.
	 */
	template<class OtherLayout, class Storage>
	tensor (const tensor<value_type, extents_type, OtherLayout, Storage> &other)
		: tensor_expression_type<self_type> ()
		, extents_ (other.extents())
		, strides_ (other.extents())
	{	
		resize(extents_);
		copy(this->rank(), this->extents().data(),
				this->data(), this->strides().data(),
				other.data(), other.strides().data());
		
	}

	/** @brief Constructs a tensor with an tensor expression
	 *
	 * @code tensor<float> A = B + 3 * C; @endcode
	 *
	 * @note type must be specified of tensor must be specified.
	 * @note dimension extents are extracted from tensors within the expression.
	 *
	 * @param expr tensor expression
	 */
	template<class derived_type>
	tensor (const tensor_expression_type<derived_type> &expr)
		: tensor_expression_type<self_type> ()
		, extents_ ( detail::retrieve_extents(expr) )
		, strides_ ( extents_ )
	{
		static_assert( detail::has_tensor_types<self_type, tensor_expression_type<derived_type>>::value,
									 "Error in boost::numeric::ublas::tensor: expression does not contain a tensor. cannot retrieve shape.");
		resize(extents_);
		detail::eval( *this, expr );
	}

	/** @brief Constructs a tensor with a matrix expression
	 *
	 * @code tensor<float> A = B + 3 * C; @endcode
	 *
	 * @note matrix expression is evaluated and pushed into a temporary matrix before assignment.
	 * @note extents are automatically extracted from the temporary matrix
	 *
	 * @param expr matrix expression
	 */
	template<class derived_type>
	tensor (const matrix_expression_type<derived_type> &expr)
		: tensor(  matrix_type ( expr )  )
	{
	}

	/** @brief Constructs a tensor with a vector expression
	 *
	 * @code tensor<float> A = b + 3 * b; @endcode
	 *
	 * @note matrix expression is evaluated and pushed into a temporary matrix before assignment.
	 * @note extents are automatically extracted from the temporary matrix
	 *
	 * @param expr vector expression
	 */
	template<class derived_type>
	tensor (const vector_expression_type<derived_type> &expr)
		: tensor(  vector_type ( expr )  )
	{
	}

	/** @brief Evaluates the tensor_expression and assigns the results to the tensor
	 *
	 * @code A = B + C * 2;  @endcode
	 *
	 * @note rank and dimension extents of the tensors in the expressions must conform with this tensor.
	 *
	 * @param expr expression that is evaluated.
	 */
	template<class derived_type>
	tensor &operator = (const tensor_expression_type<derived_type> &expr)
	{
		detail::eval(*this, expr);
		return *this;
	}

	tensor& operator=(tensor other)
	{
		swap (*this, other);
		return *this;
	}

	tensor& operator=(const_reference v)
	{
		std::fill(this->begin(), this->end(), v);
		return *this;
	}

	/** @brief Returns true if the tensor is empty (\c size==0) */
	inline
	bool empty () const {
		return this->data_.empty();
	}


	/** @brief Returns the size of the tensor */
	inline
	size_type size () const {
		return this->data_.size ();
	}

	/** @brief Returns the size of the tensor */
	inline
	size_type size (size_type r) const {
		return this->extents_.at(r);
	}

	/** @brief Returns the number of dimensions/modes of the tensor */
	inline
	size_type rank () const {
		return this->extents_.size();
	}

	/** @brief Returns the number of dimensions/modes of the tensor */
	inline
	size_type order () const {
		return this->extents_.size();
	}

	/** @brief Returns the strides of the tensor */
	inline
	strides_type const& strides () const {
		return this->strides_;
	}

	/** @brief Returns the extents of the tensor */
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

	/** @brief Generates a tensor index for tensor contraction
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





	/** @brief Reshapes the tensor
	 *
	 *
	 * (1) @code A.reshape(extents{m,n,o});     @endcode or
	 * (2) @code A.reshape(extents{m,n,o},4);   @endcode
	 *
	 * If the size of this smaller than the specified extents than
	 * default constructed (1) or specified (2) value is appended.
	 *
	 * @note rank of the tensor might also change.
	 *
	 * @param e extents with which the tensor is reshaped.
	 * @param v value which is appended if the tensor is enlarged.
	 */
	inline
	void reshape (extents_type const& e, value_type v = value_type{})
	{
		static_assert(detail::is_dynamic_v<extents_type>,
			"Error in boost::numeric::ublas::tensor: static extents cannot be reshaped");
		this->extents_ = e;
		this->strides_ = strides_type(this->extents_);

		auto p = product(extents_);
		if(p != this->size())
			this->data_.resize (p, v);
	}

	friend void swap(tensor& lhs, tensor& rhs) {
		std::swap(lhs.data_   , rhs.data_   );
		std::swap(lhs.extents_, rhs.extents_);
		std::swap(lhs.strides_, rhs.strides_);
	}


	/// \brief return an iterator on the first element of the tensor
	inline
	const_iterator begin () const {
		return data_.begin ();
	}

	/// \brief return an iterator on the first element of the tensor
	inline
	const_iterator cbegin () const {
		return data_.cbegin ();
	}

	/// \brief return an iterator after the last element of the tensor
	inline
	const_iterator end () const {
		return data_.end();
	}

	/// \brief return an iterator after the last element of the tensor
	inline
	const_iterator cend () const {
		return data_.cend ();
	}

	/// \brief Return an iterator on the first element of the tensor
	inline
	iterator begin () {
		return data_.begin();
	}

	/// \brief Return an iterator at the end of the tensor
	inline
	iterator end () {
		return data_.end();
	}

	/// \brief Return a const reverse iterator before the first element of the reversed tensor (i.e. end() of normal tensor)
	inline
	const_reverse_iterator rbegin () const {
		return data_.rbegin();
	}

	/// \brief Return a const reverse iterator before the first element of the reversed tensor (i.e. end() of normal tensor)
	inline
	const_reverse_iterator crbegin () const {
		return data_.crbegin();
	}

	/// \brief Return a const reverse iterator on the end of the reverse tensor (i.e. first element of the normal tensor)
	inline
	const_reverse_iterator rend () const {
		return data_.rend();
	}

	/// \brief Return a const reverse iterator on the end of the reverse tensor (i.e. first element of the normal tensor)
	inline
	const_reverse_iterator crend () const {
		return data_.crend();
	}

	/// \brief Return a const reverse iterator before the first element of the reversed tensor (i.e. end() of normal tensor)
	inline
	reverse_iterator rbegin () {
		return data_.rbegin();
	}

	/// \brief Return a const reverse iterator on the end of the reverse tensor (i.e. first element of the normal tensor)
	inline
	reverse_iterator rend () {
		return data_.rend();
	}

private:

	inline
	void resize( [[maybe_unused]] extents_type const& e ){
		if constexpr( detail::is_resizable_v<array_type>){
			data_.resize(product(e));
		}
	}

	inline
	void resize( [[maybe_unused]] size_type sz ){
		if constexpr( detail::is_resizable_v<array_type>){
			data_.resize(sz);
		}
	}

#if 0
	// -------------
	// Serialization
	// -------------

	/// Serialize a tensor into and archive as defined in Boost
	/// \param ar Archive object. Can be a flat file, an XML file or any other stream
	/// \param file_version Optional file version (not yet used)
	template<class Archive>
	void serialize(Archive & ar, const unsigned int /* file_version */){
		ar & serialization::make_nvp("data",data_);
	}
#endif



private:

	extents_type extents_;
	strides_type strides_;
	array_type data_;
};

#if __cpp_deduction_guides

tensor() -> tensor<float,dynamic_extents<>>;

template<class T, class E,
	typename std::enable_if< detail::is_extents<E>::value >::type* = nullptr>
tensor(E const&, T const& ) ->tensor<T,E>;

template<class E,
	typename std::enable_if< detail::is_extents<E>::value >::type* = nullptr>
tensor(E const&) ->tensor<float,E>;
#endif


}}} // namespaces





#endif
