//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//


/// \file tensor_core.hpp Definition for the tensor template class

#ifndef BOOST_UBLAS_TENSOR_STATIC_RANK_HPP
#define BOOST_UBLAS_TENSOR_STATIC_RANK_HPP

#include <initializer_list>

#include "../algorithms.hpp"
#include "../expression.hpp"
#include "../expression_evaluation.hpp"
#include "../extents.hpp"
#include "../index.hpp"
#include "../index_functions.hpp"
#include "../layout.hpp"
#include "../type_traits.hpp"
#include "../tags.hpp"
#include "../concepts.hpp"

#include "tensor_engine.hpp"



namespace boost::numeric::ublas {

template<class V, class L, std::size_t N>
using engine_tensor_static_rank = tensor_engine<extents<N>, L, std::vector<V>>;

template<class V, class L, std::size_t N>
  class tensor_core<engine_tensor_static_rank<V,L,N>>
  : public detail::tensor_expression<
      tensor_core<engine_tensor_static_rank<V,L,N>>,
      tensor_core<engine_tensor_static_rank<V,L,N>>>
{
public:
  using engine_type               = engine_tensor_static_rank<V,L,N>;
  using self_type                 = tensor_core<engine_type>;

  template<class derived_type>
  using tensor_expression_type    = detail::tensor_expression<self_type,derived_type>;
  template<class derived_type>
  using matrix_expression_type    = matrix_expression<derived_type>;
  template<class derived_type>
  using vector_expression_type    = vector_expression<derived_type>;

  using super_type                = tensor_expression_type<self_type>;

  using container_type            = typename engine_type::container_type;
  using layout_type               = typename engine_type::layout_type;
  using extents_type              = typename engine_type::extents_type;
  using strides_type              = typename extents_type::base_type;

  using container_traits_type     = container_traits<container_type>;

  using size_type                 = typename container_traits_type::size_type;
  using difference_type           = typename container_traits_type::difference_type;
  using value_type                = typename container_traits_type::value_type;

  using reference                 = typename container_traits_type::reference;
  using const_reference           = typename container_traits_type::const_reference;

  using pointer                   = typename container_traits_type::pointer;
  using const_pointer             = typename container_traits_type::const_pointer;

  using iterator                  = typename container_traits_type::iterator;
  using const_iterator            = typename container_traits_type::const_iterator;

  using reverse_iterator          = typename container_traits_type::reverse_iterator;
  using const_reverse_iterator    = typename container_traits_type::const_reverse_iterator;

  using container_tag             = typename container_traits_type::container_tag;
  using resizable_tag             = typename container_traits_type::resizable_tag;

  using matrix_type               = matrix<value_type, layout_type, std::vector<value_type> >;
  using vector_type               = vector<value_type, std::vector<value_type> >;

  tensor_core () = default;

  /** @brief Constructs a tensor_core with a \c shape
     *
     * @code auto t = tensor<float,3>(extents<3>{3,4,2}); @endcode
     *
     */
  explicit inline tensor_core (extents_type e)
    : tensor_expression_type<self_type>{}
    , _extents(std::move(e))
    , _strides(ublas::to_strides(_extents,layout_type{}))
    , _container(ublas::product(_extents))
  {
  }

  /** @brief Constructs a tensor_core with a \c shape
     *
     * @code auto t = tensor<float,3>{3,4,2}; @endcode
     *
     */
  template<integral ... Is>
  explicit inline tensor_core (Is ... is)
    : tensor_core(extents_type{size_type(is)...})
  {
  }

  /** @brief Constructs a tensor_core with a \c shape and initial value
     *
     * @code auto t = tensor<float>(extents<>{4,3,2},5); @endcode
     *
     * @param i initial tensor_core with this value
     */
  inline tensor_core (extents_type e, value_type i)
    : tensor_expression_type<self_type>{}
    , _extents(std::move(e))
    , _strides(to_strides(_extents,layout_type{}))
    , _container(product(_extents))
  {
    std::fill(begin(),end(),i);
  }

  /** @brief Constructs a tensor_core with a \c shape and initiates it with one-dimensional data
     *
     * @code auto t = tensor<float,3>(extents<>{3,4,2},std::vector<float>(3*4*2,1.f)); @endcode
     *
     *  @param e instance of \c extents<> specifying the dimensions of tensor
     *  @param a instance of \c std::vector<value_type> to be copied
     */
  inline tensor_core (extents_type e, container_type a)
    : tensor_expression_type<self_type>{}
    , _extents(std::move(e))
    , _strides(ublas::to_strides(_extents,layout_type{}))
    , _container(std::move(a))
  {
    if(std::size(_container) != ublas::product(_extents)){
      throw std::length_error("boost::numeric::ublas::tensor_static_rank : "
        "Cannot construct tensor with specified container and extents. "
        "Number of container elements do not match with the specified extents.");
    }
  }

  /** @brief Constructs a tensor_core with another tensor_core with a different layout
     *
     * @param other tensor_core with a different layout to be copied.
     */
  template<typename OTE>
  explicit inline tensor_core (const tensor_core<OTE> &other)
    : tensor_expression_type<self_type>{}
    , _extents  (ublas::begin(other.extents()),ublas::end (other.extents  ()))
    , _strides  (ublas::to_strides(_extents))
    , _container(std::begin(other.container()),std::end   (other.container()))
  {    
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
  template<typename T,typename D>
  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  inline tensor_core (detail::tensor_expression<T,D> const& expr)
    : tensor_expression_type<self_type>{}
    , _extents  (ublas::detail::retrieve_extents(expr))
    , _strides  (ublas::to_strides(_extents,layout_type{}))
    , _container(ublas::product(_extents))
  {
    detail::eval(*this, expr);
  }

  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  explicit tensor_core( matrix_type const& m )
    : tensor_expression_type<self_type>{}
    , _extents  {m.size1(),m.size2()}
    , _strides  (ublas::to_strides(_extents,layout_type{}))
    , _container(m.data().begin(), m.data().end())
  {
  }

  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  explicit tensor_core (vector_type const& v)
    : tensor_expression_type<self_type>{}
    , _extents  {v.size(),1}
    , _strides  (ublas::to_strides(_extents,layout_type{}))
    , _container(v.data().begin(), v.data().end())
  {
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
  template<class D>
  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  inline tensor_core (const matrix_expression_type<D> &expr)
    : tensor_core(matrix_type(expr))
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
  template<class D>
  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  inline tensor_core (const vector_expression_type<D> &expr)
    : tensor_core(  vector_type ( expr )  )
  {
  }


  /** @brief Constructs a tensor_core from another tensor_core
     *
     *  @param t tensor_core to be copied.
     */
  inline tensor_core (const tensor_core &t) noexcept
    : tensor_expression_type<self_type>{}
    , _extents  (t._extents  )
    , _strides  (t._strides  )
    , _container(t._container)
  {}



  /** @brief Constructs a tensor_core from another tensor_core
     *
     *  @param t tensor_core to be moved.
     */
  inline tensor_core (tensor_core &&t) noexcept
    : tensor_expression_type<self_type>{}
    , _extents  (std::move(t._extents  ))
    , _strides  (std::move(t._strides  ))
    , _container(std::move(t._container))
  {}

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

  // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
  tensor_core& operator=(tensor_core other) noexcept
  {
    swap (*this, other);
    return *this;
  }

  tensor_core& operator=(container_type c)
  {
    if( c.size() != this->size()){
      throw std::length_error("boost::numeric::ublas::tensor_core: "
        "Cannot assign provided container to tensor."
        "Number of elements do not match.");
    }
    _container = std::move(c);
    return *this;
  }

  tensor_core& operator=(const_reference v)
  {
    std::fill_n(_container.begin(), _container.size(), v);
    return *this;
  }

  /** @brief Element access using a multi-index with bound checking which can throw an exception.
     *
     *  @code auto a = A.at(i,j,k); @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) == 0, else 0<= i < this->size(0)
     *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r < this->rank()
     */
  template<integral I1, integral I2, integral ... Is>
  [[nodiscard]] inline const_reference at (I1 i1, I2 i2, Is ... is) const
  {
    static_assert (sizeof...(is)+2 == std::tuple_size_v<extents_type>);
    const auto idx = ublas::detail::to_index(_strides,i1,i2,is...);
    return _container.at(idx);
  }

  /** @brief Element access using a multi-index with bound checking which can throw an exception.
     *
     *  @code auto a = A.at(i,j,k); @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) == 0, else 0<= i < this->size(0)
     *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r < this->rank()
     */
  template<integral I1, integral I2, integral ... Is>
  [[nodiscard]] inline reference at (I1 i1, I2 i2, Is ... is)
  {
    static_assert (sizeof...(Is)+2 == std::tuple_size_v<extents_type>);
    const auto idx = ublas::detail::to_index(_strides,i1,i2,is...);
    return _container.at(idx);
  }

  /** @brief Element access using a multi-index with bound checking which can throw an exception.
     *
     *  @code auto a = A(i,j,k); @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) == 0, else 0<= i < this->size(0)
     *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r < this->rank()
     */
  template<integral ... Is>
  [[nodiscard]] inline const_reference operator()(Is ... is) const
  {
    return this->at(is...);
  }

  /** @brief Element access using a multi-index with bound checking which can throw an exception.
     *
     *  @code auto a = A(i,j,k); @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) == 0, else 0<= i < this->size(0)
     *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r < this->rank()
     */
  template<integral ... Is>
  [[nodiscard]] inline reference operator()(Is ... is)
  {
    return this->at(is...);
  }

  /** @brief Element access using a single index.
     *
     *  @code auto a = A[i]; @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size()
     */
  [[nodiscard]] inline const_reference operator [] (size_type i) const {
    return this->_container[i];
  }

  /** @brief Element access using a single index.
     *
     *  @code auto a = A[i]; @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size()
     */
  [[nodiscard]] inline reference operator [] (size_type i) {
    return this->_container[i];
  }

  /** @brief Element access using a single-index with bound checking which can throw an exception.
     *
     *  @code auto a = A.at(i);     @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size()
     */
  [[nodiscard]] inline const_reference at (size_type i) const {
    return this->_container.at(i);
  }

  /** @brief Read tensor element of a tensor \c t with a single-index \c i
     *
     *  @code auto a = t.at(i);  @endcode
     *
     *  @param i zero-based index where 0 <= i < t.size()
     */
  [[nodiscard]] inline reference at (size_type i) {
    return this->_container.at(i);
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
  [[nodiscard]] inline constexpr decltype(auto) operator() (index::index_type<I> p, index_types ... ps) const
  {
    constexpr auto size = sizeof...(index_types)+1;
    static_assert(size == std::tuple_size_v<extents_type>);
    return std::make_pair( std::cref(*this),  std::make_tuple( p, std::forward<index_types>(ps)... ) );
  }

  friend void swap(tensor_core& lhs, tensor_core& rhs)
  {
    std::swap(lhs._extents   , rhs._extents  );
    std::swap(lhs._strides   , rhs._strides  );
    std::swap(lhs._container , rhs._container);
  }


  [[nodiscard]] inline           auto begin  () const noexcept -> const_iterator { return _container.begin  (); }
  [[nodiscard]] inline           auto end    () const noexcept -> const_iterator { return _container.end    (); }
  [[nodiscard]] inline           auto begin  ()       noexcept ->       iterator { return _container.begin  (); }
  [[nodiscard]] inline           auto end    ()       noexcept ->       iterator { return _container.end    (); }
  [[nodiscard]] inline           auto cbegin () const noexcept -> const_iterator { return _container.cbegin (); }
  [[nodiscard]] inline           auto cend   () const noexcept -> const_iterator { return _container.cend   (); }

  [[nodiscard]] inline           auto crbegin() const noexcept -> const_reverse_iterator { return _container.crbegin(); }
  [[nodiscard]] inline           auto crend  () const noexcept -> const_reverse_iterator { return _container.crend  (); }
  [[nodiscard]] inline           auto rbegin () const noexcept -> const_reverse_iterator { return _container.rbegin (); }
  [[nodiscard]] inline           auto rend   () const noexcept -> const_reverse_iterator { return _container.rend   (); }
  [[nodiscard]] inline           auto rbegin ()       noexcept ->       reverse_iterator { return _container.rbegin (); }
  [[nodiscard]] inline           auto rend   ()       noexcept ->       reverse_iterator { return _container.rend   (); }

  [[nodiscard]] inline           auto empty ()            const noexcept { return _container.empty();    }
  [[nodiscard]] inline           auto size  ()            const noexcept { return _container.size();     }
  [[nodiscard]] inline           auto size  (size_type r) const          { return _extents.at(r);        }
  [[nodiscard]] inline constexpr auto rank  ()            const noexcept { return std::tuple_size_v<extents_type>; }
  [[nodiscard]] inline constexpr auto order ()            const noexcept { return this->rank();          }

  [[nodiscard]] inline           auto const& strides () const noexcept                  { return _strides; }
  [[nodiscard]] inline           auto const& extents () const noexcept                  { return _extents; }
  [[nodiscard]] inline           auto        data    () const noexcept -> const_pointer { return _container.data();}
  [[nodiscard]] inline           auto        data    ()       noexcept -> pointer       { return _container.data();}
  [[nodiscard]] inline           auto const& base    () const noexcept                  { return _container; }


private:
  extents_type _extents;
  strides_type _strides;
  container_type _container;
};

/** @brief Type for create a dynamic tensor instance with dynamic non-resizable extents
 *
 * @code
 * // defines a 4-dimensional tensor type
 * // tensor_core<tensor_engine<extents<4>,layout::first_order,std::vector<float>>>
 *
 * using ftensor = tensor_mixed<float,4>;
 *
 * // instantiates a 4-dimension
 * auto t = ftensor{{5,6,4,3}};
 *
 * @endcode
 *
 *  */

template<class V, std::size_t N, class L = layout::first_order>
using tensor_static_rank = tensor_core<engine_tensor_static_rank<V, L, N>>;

} // namespace boost::numeric::ublas


namespace boost::numeric::ublas::experimental
{

template<class V, class L = layout::first_order>
using matrix = tensor_core<engine_tensor_static_rank<V, L, 2>>;

template<class V, class L = layout::first_order>
using vector = tensor_core<engine_tensor_static_rank<V, L, 1>>;

} // namespace boost::numeric::ublas::experimental


#endif // BOOST_UBLAS_TENSOR_STATIC_RANK_HPP

