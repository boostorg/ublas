//
//  Copyright (c) 2018
//  Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB in producing this work.
//
//  And we acknowledge the support from all contributors.

/// \file tensor.hpp Definition for the class vector and its derivative

#ifndef _BOOST_UBLAS_TENSOR_
#define _BOOST_UBLAS_TENSOR_

#include <boost/config.hpp>
#include <boost/numeric/ublas/expression_types.hpp>
#include <boost/numeric/ublas/storage.hpp>

#include <boost/numeric/ublas/tensor/extents.hpp>
#include <boost/numeric/ublas/tensor/strides.hpp>

//#include <boost/numeric/ublas/vector_expression.hpp>
//#include <boost/numeric/ublas/detail/vector_assign.hpp>
//#include <boost/serialization/collection_size_type.hpp>
//#include <boost/serialization/nvp.hpp>

#include <initializer_list>

#ifdef BOOST_UBLAS_CPP_GE_2011
#include <array>
#include <initializer_list>
#if defined(BOOST_MSVC) // For std::forward in fixed_vector
#include <utility>
#endif
#endif

/** \brief Base class for Vector container models
 *
 * it does not model the Vector concept but all derived types should.
 * The class defines a common base type and some common interface for all
 * statically derived Vector classes
 * We implement the casts to the statically derived type.
 */



namespace boost { namespace numeric { namespace ublas {

template<class T, class F, class A>
class tensor;


//TODO: put in fwd.hpp
struct tensor_tag {};


//TODO: put in expression_types.hpp

/** \brief Base class for Tensor Expression models
 *
 * it does not model the Tensor Expression concept but all derived types should.
 * The class defines a common base type and some common interface for all
 * statically derived Tensor Expression classes.
 * We implement the casts to the statically derived type.
 */
template<class E>
class tensor_expression:
		public ublas_expression<E> {
public:
		static const unsigned complexity = 0;
		typedef E expression_type;
		typedef tensor_tag type_category;
		/* E can be an incomplete type - to define the following we would need more template arguments
		typedef typename E::size_type size_type;
		*/

		BOOST_UBLAS_INLINE
		const expression_type &operator () () const {
				return *static_cast<const expression_type *> (this);
		}
		BOOST_UBLAS_INLINE
		expression_type &operator () () {
				return *static_cast<expression_type *> (this);
		}

#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
private:
		// projection types
		typedef tensor_range<E> tensor_range_type;
		typedef tensor_range<const E> const_tensor_range_type;
		typedef tensor_slice<E> tensor_slice_type;
		typedef tensor_slice<const E> const_tensor_slice_type;
		// tensor_indirect_type will depend on the A template parameter
		typedef basic_range<> default_range;    // required to avoid range/slice name confusion
		typedef basic_slice<> default_slice;
public:
		BOOST_UBLAS_INLINE
		const_tensor_range_type operator () (const default_range &r) const {
				return const_tensor_range_type (operator () (), r);
		}
		BOOST_UBLAS_INLINE
		tensor_range_type operator () (const default_range &r) {
				return tensor_range_type (operator () (), r);
		}
		BOOST_UBLAS_INLINE
		const_tensor_slice_type operator () (const default_slice &s) const {
				return const_tensor_slice_type (operator () (), s);
		}
		BOOST_UBLAS_INLINE
		tensor_slice_type operator () (const default_slice &s) {
				return tensor_slice_type (operator () (), s);
		}
		template<class A>
		BOOST_UBLAS_INLINE
		const tensor_indirect<const E, indirect_array<A> > operator () (const indirect_array<A> &ia) const {
				return tensor_indirect<const E, indirect_array<A> >  (operator () (), ia);
		}
		template<class A>
		BOOST_UBLAS_INLINE
		tensor_indirect<E, indirect_array<A> > operator () (const indirect_array<A> &ia) {
				return tensor_indirect<E, indirect_array<A> > (operator () (), ia);
		}

		BOOST_UBLAS_INLINE
		const_tensor_range_type project (const default_range &r) const {
				return const_tensor_range_type (operator () (), r);
		}
		BOOST_UBLAS_INLINE
		tensor_range_type project (const default_range &r) {
				return tensor_range_type (operator () (), r);
		}
		BOOST_UBLAS_INLINE
		const_tensor_slice_type project (const default_slice &s) const {
				return const_tensor_slice_type (operator () (), s);
		}
		BOOST_UBLAS_INLINE
		tensor_slice_type project (const default_slice &s) {
				return tensor_slice_type (operator () (), s);
		}
		template<class A>
		BOOST_UBLAS_INLINE
		const tensor_indirect<const E, indirect_array<A> > project (const indirect_array<A> &ia) const {
				return tensor_indirect<const E, indirect_array<A> > (operator () (), ia);
		}
		template<class A>
		BOOST_UBLAS_INLINE
		tensor_indirect<E, indirect_array<A> > project (const indirect_array<A> &ia) {
				return tensor_indirect<E, indirect_array<A> > (operator () (), ia);
		}
#endif
};


/** \brief output stream operator for tensor expressions
 *
 * Any vector expressions can be written to a standard output stream
 * as defined in the C++ standard library. For example:
 * \code
 * vector<float> v1(3),v2(3);
 * for(size_t i=0; i<3; i++)
 * {
 *       v1(i) = i+0.2;
 *       v2(i) = i+0.3;
 * }
 * cout << v1+v2 << endl;
 * \endcode
 * will display the some of the 2 vectors like this:
 * \code
 * [3](0.5,2.5,4.5)
 * \endcode
 *
 * \param os is a standard basic output stream
 * \param v is a vector expression
 * \return a reference to the resulting output stream
 */
template<class E, class T, class VE>
// BOOST_UBLAS_INLINE This function seems to be big. So we do not let the compiler inline it.
std::basic_ostream<E, T>&
operator << (std::basic_ostream<E, T> &os, const tensor_expression<VE> &v) {
		typedef typename VE::size_type size_type;
		size_type size = v ().size ();
		std::basic_ostringstream<E, T, std::allocator<E> > s;
		s.flags (os.flags ());
		s.imbue (os.getloc ());
		s.precision (os.precision ());
		s << '[' << size << "](";
		if (size > 0)
				s << v () (0);
		for (size_type i = 1; i < size; ++ i)
				s << ',' << v () (i);
		s << ')';
		return os << s.str ().c_str ();
}




/** \brief Base class for Tensor container models
 *
 * it does not model the Tensor concept but all derived types should.
 * The class defines a common base type and some common interface for all
 * statically derived Tensor classes
 * We implement the casts to the statically derived type.
 */
template<class C>
class tensor_container:
		public tensor_expression<C>
{
public:
		static const unsigned complexity = 0;
		typedef C container_type;
		typedef tensor_tag type_category;

		BOOST_UBLAS_INLINE
		const container_type &operator () () const {
				return *static_cast<const container_type *> (this);
		}
		BOOST_UBLAS_INLINE
		container_type &operator () () {
				return *static_cast<container_type *> (this);
		}

#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
		using tensor_expression<C>::operator ();
#endif
};

/** @brief A dense tensor of values of type \c T.
		*
		* For a \f$n\f$-dimensional tensor \f$v\f$ and \f$0\leq i < n\f$ every element \f$v_i\f$ is mapped
		* to the \f$i\f$-th element of the container. A storage type \c A can be specified which defaults to \c unbounded_array.
		* Elements are constructed by \c A, which need not initialise their value.
		*
		* @tparam T type of the objects stored in the tensor (like int, double, complex,...)
		* @tparam A The type of the storage array of the tensor. Default is \c unbounded_array<T>. \c <bounded_array<T> and \c std::vector<T> can also be used
		*/
template<class T, class F = first_order, class A = std::vector<T,std::allocator<T>> >
class tensor:
		public tensor_container<tensor<T, F, A> >
{

	static_assert( std::is_same_v<F,first_order> ||
								 std::is_same_v<F,last_order >, "boost::numeric::tensor template class only supports first- or last-order storage formats.");

	typedef tensor<T, F, A> self_type;
public:
#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
	using tensor_container<self_type>::operator ();
#endif
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

//	typedef const tensor_reference<const self_type> const_closure_type;
//	typedef tensor_reference<self_type> closure_type;
	typedef self_type tensor_temporary_type;
	typedef dense_tag storage_category;

	using strides_type = basic_strides<size_type,layout_type>;
	using extents_type = basic_extents<size_type>;

	// Reverse iterator
//	typedef reverse_iterator_base<const_iterator> const_reverse_iterator;
//	typedef reverse_iterator_base<iterator> reverse_iterator;

	/** @brief Standard constructor of the tensor template class
	 *
	 * By default it is empty, i.e. \c size()==0.
	 */
	BOOST_UBLAS_INLINE
	constexpr tensor ()
		: tensor_container<self_type>()
		, extents_()
		, strides_()
		, data_()
	{
	}


	/** @brief Constructor of the tensor template class with a predefined size
	 *
	 * Layout or storage format is automatically set to first-order.
	 * By default, its elements are initialized to 0.
	 *
	 * @code tensor A{4,2,3}; @endcode
	 *
	 * @param l initializer list for setting the dimension extents of the tensor
	 */
	explicit BOOST_UBLAS_INLINE
	tensor (std::initializer_list<size_type> l)
		: tensor_container<self_type>()
		, extents_ (std::move(l))
		, strides_ (extents_)
		, data_    (extents_.product())
	{
	}


	/** @brief Constructor of the tensor template class
		*
		* Layout or storage format is automatically set to first-order.
		* By default, its elements are initialized to 0.
		*
		* @code tensor A{extents{4,2,3}}; @endcode
		*
		* @param e initial tensor dimension extents
		*/
	explicit BOOST_UBLAS_INLINE
	tensor (extents const& e)
		: tensor_container<self_type>()
		, extents_ (e)
		, strides_ (extents_)
		, data_    (extents_.product())
	{}


	/** @brief Constructor of the tensor template class
	 *
	 *
	 *  @code tensor A{extents{4,2,3}, array }; @endcode
	 *
	 *  @param e initial tensor dimension extents
	 *  @param data container of \c array_type
	 */
	BOOST_UBLAS_INLINE
	tensor (extents const& e, const array_type &data)
		: tensor_container<self_type>()
		, extents_ (e)
		, strides_ (extents_)
		, data_    (data)
	{
		if(this->extents_.product() != this->data_.size())
			throw std::runtime_error("Error in boost::numeric::ublas::tensor: size of provided data and specified extents do not match.");
	}



	/** @brief Constructor of the tensor template class
	 *
	 *  @param e initial tensor dimension extents
	 *  @param i initial value of all elements of type \c value_type
	 */
	BOOST_UBLAS_INLINE
	tensor (extents const& e, const value_type &i)
		: tensor_container<self_type> ()
		, extents_ (e)
		, strides_ (extents_)
		, data_    (extents_.product(), i)
	{}



	/** @brief Copy Constructor of the tensor template class
	 *
	 *  @param v tensor to be copied.
	 */
	BOOST_UBLAS_INLINE
	tensor (const tensor &v)
		: tensor_container<self_type> ()
		, extents_ (v.extents_)
		, strides_ (v.strides_)
		, data_    (v.data_   )
	{}

//	/** @brief Copy Constructor of the tensor template class
//	 *
//	 *  @param v tensor to be copied.
//	 */
//	BOOST_UBLAS_INLINE
//	template<class other_layout>
//	tensor (const tensor<value_type, layout_type> &v)
//		: tensor_container<self_type> ()
//		, extents_ (v.extents_)
//		, strides_ (v.strides_)
//		, data_    (v.data_   )
//	{}

	/** @brief Move Constructor of the tensor template class
	 *
	 *  @param v tensor to be moved.
	 */
	BOOST_UBLAS_INLINE
	tensor (tensor &&v)
		: tensor_container<self_type> ()
		, extents_ (std::move(v.extents_))
		, strides_ (std::move(v.strides_))
		, data_    (std::move(v.data_   ))
	{}


#if 0
	/// \brief Copy-constructor of a tensor from a tensor_expression
	/// Depending on the tensor_expression, this constructor can have the cost of the computations
	/// of the expression (trivial to say it, but it is to take into account in your complexity calculations).
	/// \param ae the tensor_expression which values will be duplicated into the tensor
	template<class AE>
	BOOST_UBLAS_INLINE
	tensor (const tensor_expression<AE> &ae):
		tensor_container<self_type> (),
		data_ (ae ().size ())
	{
		tensor_assign<scalar_assign> (*this, ae);
	}
#endif

	/** @brief Returns true if the tensor is empty (\c size==0) */
	BOOST_UBLAS_INLINE
	bool empty () const {
		return this->data_.empty();
	}


	/** @brief Returns the size of the tensor */
	BOOST_UBLAS_INLINE
	size_type size () const {
		return this->data_.size ();
	}

	/** @brief Returns the size of the tensor */
	BOOST_UBLAS_INLINE
	size_type size (size_type r) const {
		return this->extents_.at(r);
	}

	/** @brief Returns the size of the tensor */
	BOOST_UBLAS_INLINE
	size_type rank () const {
		return this->extents_.size();
	}

	/** @brief Returns the strides of the tensor */
	BOOST_UBLAS_INLINE
	strides_type const& strides () const {
		return this->strides_;
	}

	/** @brief Returns the extents of the tensor */
	BOOST_UBLAS_INLINE
	extents_type const& extents () const {
		return this->extents_;
	}


	/** @brief Returns a \c const reference to the container. */
	BOOST_UBLAS_INLINE
	const_pointer data () const {
		return this->data_.data();
	}

	/** @brief Returns a \c const reference to the container. */
	BOOST_UBLAS_INLINE
	pointer data () {
		return this->data_.data();
	}

	/** @brief Element access using a single index.
	 *
	 *  @code auto a = A[i]; @endcode
	 *
	 *  @param i zero-based index where 0 <= i < this->size()
	 */
	BOOST_UBLAS_INLINE
	const_reference operator [] (size_type i) const {
		return this->data_[i];
	}

	/** @brief Element access using a single index.
	 *
	 *
	 *  @code A[i] = a; @endcode
	 *
	 *  @param i zero-based index where 0 <= i < this->size()
	 */
	BOOST_UBLAS_INLINE
	reference operator [] (size_type i)
	{
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
	BOOST_UBLAS_INLINE
	const_reference at (size_type i, size_types ... is) const
	{
		if constexpr (sizeof...(is) == 0)
			return this->data_[i];
		else
			return this->data_[ access<0>(0,i,is...)];
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
	BOOST_UBLAS_INLINE
	template<class ... size_types>
	reference at (size_type i, size_types ... is)
	{
		if constexpr (sizeof...(is) == 0)
			return this->data_[i];
		else
			return this->data_[ access<0>(0,i,is...)];
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
	BOOST_UBLAS_INLINE
	void reshape (extents_type const& e, value_type v = value_type{})
	{
		this->extents_ = e;
		this->strides_ = strides_type(this->extents_);

		if(e.product() != this->size())
			this->data_.resize (this->extents_.product(), v);
	}


	tensor& operator=(tensor other)
	{
		swap (*this, other);
		return *this;
	}
#if 0
	/// \brief Assign a full tensor (\e RHS-tensor) to the current tensor (\e LHS-tensor)
	/// Assign a full tensor (\e RHS-tensor) to the current tensor (\e LHS-tensor). This method does not create any temporary.
	/// \param v is the source tensor container
	/// \return a reference to a tensor (i.e. the destination tensor)
	template<class C>          // Container assignment without temporary
	BOOST_UBLAS_INLINE
	tensor &operator = (const tensor_container<C> &v) {
		resize (v ().size (), false);
		assign (v);
		return *this;
	}

	/// \brief Assign a full tensor (\e RHS-tensor) to the current tensor (\e LHS-tensor)
	/// \param v is the source tensor
	/// \return a reference to a tensor (i.e. the destination tensor)
	BOOST_UBLAS_INLINE
	tensor &assign_temporary (tensor &v) {
		swap (v);
		return *this;
	}

	/// \brief Assign the result of a tensor_expression to the tensor
	/// Assign the result of a tensor_expression to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// \tparam AE is the type of the tensor_expression
	/// \param ae is a const reference to the tensor_expression
	/// \return a reference to the resulting tensor
	template<class AE>
	BOOST_UBLAS_INLINE
	tensor &operator = (const tensor_expression<AE> &ae)
	{
		self_type temporary (ae);
		return assign_temporary (temporary);
	}

	/// \brief Assign the result of a tensor_expression to the tensor
	/// Assign the result of a tensor_expression to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// \tparam AE is the type of the tensor_expression
	/// \param ae is a const reference to the tensor_expression
	/// \return a reference to the resulting tensor
	template<class AE>
	BOOST_UBLAS_INLINE
	tensor &assign (const tensor_expression<AE> &ae)
	{
//		tensor_assign<scalar_assign> (*this, ae);
		return *this;
	}
#endif

#if 0
	// -------------------
	// Computed assignment
	// -------------------

	/// \brief Assign the sum of the tensor and a tensor_expression to the tensor
	/// Assign the sum of the tensor and a tensor_expression to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// A temporary is created for the computations.
	/// \tparam AE is the type of the tensor_expression
	/// \param ae is a const reference to the tensor_expression
	/// \return a reference to the resulting tensor
	template<class AE>
	BOOST_UBLAS_INLINE
	tensor &operator += (const tensor_expression<AE> &ae) {
		self_type temporary (*this + ae);
		return assign_temporary (temporary);
	}

	/// \brief Assign the sum of the tensor and a tensor_expression to the tensor
	/// Assign the sum of the tensor and a tensor_expression to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// No temporary is created. Computations are done and stored directly into the resulting tensor.
	/// \tparam AE is the type of the tensor_expression
	/// \param ae is a const reference to the tensor_expression
	/// \return a reference to the resulting tensor
	template<class C>          // Container assignment without temporary
	BOOST_UBLAS_INLINE
	tensor &operator += (const tensor_container<C> &v) {
		plus_assign (v);
		return *this;
	}

	/// \brief Assign the sum of the tensor and a tensor_expression to the tensor
	/// Assign the sum of the tensor and a tensor_expression to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// No temporary is created. Computations are done and stored directly into the resulting tensor.
	/// \tparam AE is the type of the tensor_expression
	/// \param ae is a const reference to the tensor_expression
	/// \return a reference to the resulting tensor
	template<class AE>
	BOOST_UBLAS_INLINE
	tensor &plus_assign (const tensor_expression<AE> &ae)
	{
//		tensor_assign<scalar_plus_assign> (*this, ae);
		return *this;
	}

	/// \brief Assign the difference of the tensor and a tensor_expression to the tensor
	/// Assign the difference of the tensor and a tensor_expression to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// A temporary is created for the computations.
	/// \tparam AE is the type of the tensor_expression
	/// \param ae is a const reference to the tensor_expression
	template<class AE>
	BOOST_UBLAS_INLINE
	tensor &operator -= (const tensor_expression<AE> &ae) {
		self_type temporary (*this - ae);
		return assign_temporary (temporary);
	}

	/// \brief Assign the difference of the tensor and a tensor_expression to the tensor
	/// Assign the difference of the tensor and a tensor_expression to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// No temporary is created. Computations are done and stored directly into the resulting tensor.
	/// \tparam AE is the type of the tensor_expression
	/// \param ae is a const reference to the tensor_expression
	/// \return a reference to the resulting tensor
	template<class C>          // Container assignment without temporary
	BOOST_UBLAS_INLINE
	tensor &operator -= (const tensor_container<C> &v) {
		minus_assign (v);
		return *this;
	}

	/// \brief Assign the difference of the tensor and a tensor_expression to the tensor
	/// Assign the difference of the tensor and a tensor_expression to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// No temporary is created. Computations are done and stored directly into the resulting tensor.
	/// \tparam AE is the type of the tensor_expression
	/// \param ae is a const reference to the tensor_expression
	/// \return a reference to the resulting tensor
	template<class AE>
	BOOST_UBLAS_INLINE
	tensor &minus_assign (const tensor_expression<AE> &ae)
	{
//		tensor_assign<scalar_minus_assign> (*this, ae);
		return *this;
	}

	/// \brief Assign the product of the tensor and a scalar to the tensor
	/// Assign the product of the tensor and a scalar to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// No temporary is created. Computations are done and stored directly into the resulting tensor.
	/// \tparam AE is the type of the tensor_expression
	/// \param at is a const reference to the scalar
	/// \return a reference to the resulting tensor
	template<class AT>
	BOOST_UBLAS_INLINE
	tensor &operator *= (const AT &at)
	{
//		tensor_assign_scalar<scalar_multiplies_assign> (*this, at);
		return *this;
	}

	/// \brief Assign the division of the tensor by a scalar to the tensor
	/// Assign the division of the tensor by a scalar to the tensor. This is lazy-compiled and will be optimized out by the compiler on any type of expression.
	/// No temporary is created. Computations are done and stored directly into the resulting tensor.
	/// \tparam AE is the type of the tensor_expression
	/// \param at is a const reference to the scalar
	/// \return a reference to the resulting tensor
	template<class AT>
	BOOST_UBLAS_INLINE
	tensor &operator /= (const AT &at)
	{
//		tensor_assign_scalar<scalar_divides_assign> (*this, at);
		return *this;
	}
#endif



	friend void swap(tensor& lhs, tensor& rhs) {
		std::swap(lhs.data_   , rhs.data_   );
		std::swap(lhs.extents_, rhs.extents_);
		std::swap(lhs.strides_, rhs.strides_);
	}


	/// \brief return an iterator on the first element of the tensor
	BOOST_UBLAS_INLINE
	const_iterator begin () const {
		return data_.begin ();
	}

	/// \brief return an iterator on the first element of the tensor
	BOOST_UBLAS_INLINE
	const_iterator cbegin () const {
		return data_.cbegin ();
	}

	/// \brief return an iterator after the last element of the tensor
	BOOST_UBLAS_INLINE
	const_iterator end () const {
		return data_.end();
	}

	/// \brief return an iterator after the last element of the tensor
	BOOST_UBLAS_INLINE
	const_iterator cend () const {
		return data_.cend ();
	}

	/// \brief Return an iterator on the first element of the tensor
	BOOST_UBLAS_INLINE
	iterator begin () {
		return data_.begin();
	}

	/// \brief Return an iterator at the end of the tensor
	BOOST_UBLAS_INLINE
	iterator end () {
		return data_.end();
	}

	/// \brief Return a const reverse iterator before the first element of the reversed tensor (i.e. end() of normal tensor)
	BOOST_UBLAS_INLINE
	const_reverse_iterator rbegin () const {
		return data_.rbegin();
	}

	/// \brief Return a const reverse iterator before the first element of the reversed tensor (i.e. end() of normal tensor)
	BOOST_UBLAS_INLINE
	const_reverse_iterator crbegin () const {
		return data_.crbegin();
	}

	/// \brief Return a const reverse iterator on the end of the reverse tensor (i.e. first element of the normal tensor)
	BOOST_UBLAS_INLINE
	const_reverse_iterator rend () const {
		return data_.rend();
	}

	/// \brief Return a const reverse iterator on the end of the reverse tensor (i.e. first element of the normal tensor)
	BOOST_UBLAS_INLINE
	const_reverse_iterator crend () const {
		return data_.crend();
	}

	/// \brief Return a const reverse iterator before the first element of the reversed tensor (i.e. end() of normal tensor)
	BOOST_UBLAS_INLINE
	reverse_iterator rbegin () {
		return data_.rbegin();
	}

	/// \brief Return a const reverse iterator on the end of the reverse tensor (i.e. first element of the normal tensor)
	BOOST_UBLAS_INLINE
	reverse_iterator rend () {
		return data_.rend();
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

	/** @brief Memory access function with multi-indices
	 *
	 * @code auto m = access(0, 3,4,5); @endcode
	 *
	 * @param[in] i multi-index vector of length p
	 * @returns relative memory location depending on \c i
	*/
	BOOST_UBLAS_INLINE
	size_type access(std::vector<size_type> const& i)
	{
		const auto p = this->rank();
		size_type sum = 0u;
		for(auto r = 0u; r < p; ++r)
			sum += i[r]*strides_[r];
		return sum;
	}

	/** @brief Memory access function with multi-indices
	 *
	 * @code auto m = access(0, 3,4,5); @endcode
	 *
	 *
	 * @param[in] i   first element of the partial multi-index
	 * @param[in] is  the following elements of the partial multi-index
	 * @param[in] sum the current
	 * @returns relative memory location depending on \c i
	*/
	BOOST_UBLAS_INLINE
	template<std::size_t r, class ... size_types>
	size_type access(size_type sum, size_type i, size_types ... is) const
	{
		sum+=i*strides_[r];
		if constexpr (sizeof...(is) == 0)
			return sum;
		else
			return access<r+1>(sum,std::forward<size_type>(is)...);
	}

	extents_type extents_;
	strides_type strides_;
	array_type data_;
};

}}} // namespaces



#endif
