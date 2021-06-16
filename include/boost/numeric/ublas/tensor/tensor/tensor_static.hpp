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

#ifndef BOOST_UBLAS_TENSOR_STATIC_HPP
#define BOOST_UBLAS_TENSOR_STATIC_HPP

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




namespace boost::numeric::ublas::detail
{
template<class V, class L, std::size_t ... ns>
using engine_tensor_static = tensor_engine<
  extents<ns...>, L, std::array<V,product_v<extents<ns...>>> >;
} // namespace boost::numeric::ublas::detail

namespace boost::numeric::ublas {
template<class V, class L, std::size_t ... ns>
class tensor_core<detail::engine_tensor_static<V,L,ns...>>
  : public detail::tensor_expression<
      tensor_core<detail::engine_tensor_static<V,L,ns...>>,
      tensor_core<detail::engine_tensor_static<V,L,ns...>>>
{
public:
  using engine_type               = detail::engine_tensor_static<V,L,ns...>;
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

  static_assert(std::tuple_size_v<container_type> == ublas::product_v<extents_type>);
  static_assert(0ul != ublas::product_v<extents_type>);

  /** @brief Constructs a tensor_core.
     *
     */
  constexpr inline tensor_core () noexcept = default;

  /** @brief Constructs a tensor_core with a \c shape
     *
     *
     * @code tensor<float,extents<4,2,3>> A(4); @endcode
     *
     * @param v value with which tensor_core is initialized
     */
  constexpr explicit inline tensor_core (value_type v)
    : tensor_core()
  {
    std::fill_n(begin(),this->size(),v);
  }

  /** @brief Constructs a tensor_core with a \c shape and initiates it with one-dimensional data
     *
     * @code auto a = tensor<float,extents<4,2,3>>(array); @endcode
     *
     *  @param s initial tensor_core dimension extents
     *  @param a container of \c array_type that is copied according to the storage layout
     */
  constexpr explicit inline tensor_core (container_type a) noexcept
    : tensor_expression_type<self_type>{}
    , _container{std::move(a)}
  {
  }


  /** @brief Constructs a tensor_core with another tensor_core with a different layout
     *
     * @param other tensor_core with a different layout to be copied.
     */
  template<typename OtherTE>
  explicit inline tensor_core (const tensor_core<OtherTE> &other)
    : tensor_expression_type<self_type>{}
    , _container{}
  {
    if(_extents != other.extents()){
      throw std::invalid_argument("error in boost::numeric::ublas::tensor_core: extents do not match.");
    }

    ublas::copy(this->rank(), this->extents().data(),
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
  template<typename T,typename D>
  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  inline tensor_core (const detail::tensor_expression<T,D> &expr)
    : tensor_expression_type<self_type>{}
    , _container{}
  {
    detail::eval(*this, expr);
  }

  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  explicit tensor_core( matrix_type const& m )
  {
    static_assert(is_matrix_v<extents_type>);
    if(m.size1() != std::get<0>(_extents) || m.size2() != std::get<1>(_extents) ){
      throw std::invalid_argument("error in boost::numeric::ublas::tensor_core: matrix and tensor dimensions do not match.");
    }
    std::copy(m.data().begin(), m.data().end(), this->begin());
  }

  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  constexpr explicit tensor_core (vector_type const& v)
  {
    static_assert(is_vector_v<extents_type>);

    if(v.size() != std::get<0>(_extents) && v.size() != std::get<1>(_extents) ){
      throw std::invalid_argument("error in boost::numeric::ublas::tensor_core: matrix and tensor dimensions do not match.");
    }
    std::copy(v.data().begin(), v.data().end(), this->begin());
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
  constexpr inline tensor_core (const tensor_core &t) noexcept
    : tensor_expression_type<self_type>{}
    , _container{t._container}
  {}



  /** @brief Constructs a tensor_core from another tensor_core
     *
     *  @param t tensor_core to be moved.
     */
  constexpr inline tensor_core (tensor_core &&t) noexcept
    : tensor_expression_type<self_type>{}
    , _container (std::move(t._container))
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
  constexpr tensor_core& operator=(tensor_core other) noexcept
  {
    swap (*this, other);
    return *this;
  }

  constexpr tensor_core& operator=(const_reference v) noexcept
  {
    std::fill_n(this->_container.begin(), this->_container.size(), v);
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
    static_assert (sizeof...(is)+2 == ublas::size_v<extents_type>);
    const auto idx = ublas::detail::to_index(_strides,i1,i2,is... );
    return _container[idx];
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
    static_assert (sizeof...(is)+2 == ublas::size_v<extents_type>);
    const auto idx = ublas::detail::to_index(_strides,i1,i2,is... );
    return _container[idx];
  }

  /** @brief Element access using a multi-index with bound checking which can throw an exception.
     *
     *  @code auto a = A(i,j,k); @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) == 0, else 0<= i < this->size(0)
     *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r < this->rank()
     */
  template<integral ... Is>
  [[nodiscard]] inline constexpr const_reference operator()(Is ... is) const
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
  [[nodiscard]] inline constexpr reference operator()(Is ... is)
  {
    return this->at(is...);
  }

  /** @brief Element access using a single index.
     *
     *  @code auto a = A[i]; @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size()
     */
  [[nodiscard]] inline constexpr const_reference operator [] (size_type i) const {
    return this->_container[i];
  }

  /** @brief Element access using a single index.
     *
     *  @code auto a = A[i]; @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size()
     */
  [[nodiscard]] inline constexpr reference operator [] (size_type i) {
    return this->_container[i];
  }

  /** @brief Element access using a single-index with bound checking which can throw an exception.
     *
     *  @code auto a = A.at(i);     @endcode
     *
     *  @param i zero-based index where 0 <= i < this->size()
     */
  [[nodiscard]] inline constexpr const_reference at (size_type i) const {
    return this->_container.at(i);
  }

  /** @brief Read tensor element of a tensor \c t with a single-index \c i
     *
     *  @code auto a = t.at(i);  @endcode
     *
     *  @param i zero-based index where 0 <= i < t.size()
     */
  [[nodiscard]] inline constexpr reference at (size_type i) {
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
    constexpr auto size = sizeof...(ps)+1;
    static_assert(size == ublas::size_v<extents_type>);
    return std::make_pair( std::cref(*this),  std::make_tuple( p, std::forward<index_types>(ps)... ) );
  }

  friend void swap(tensor_core& lhs, tensor_core& rhs)
  {
    std::swap(lhs._container, rhs._container);
  }


  [[nodiscard]] inline constexpr auto begin  () const noexcept -> const_iterator         { return _container.begin  (); }
  [[nodiscard]] inline constexpr auto end    () const noexcept -> const_iterator         { return _container.end    (); }
  [[nodiscard]] inline constexpr auto begin  ()       noexcept ->       iterator         { return _container.begin  (); }
  [[nodiscard]] inline constexpr auto end    ()       noexcept ->       iterator         { return _container.end    (); }
  [[nodiscard]] inline constexpr auto cbegin () const noexcept -> const_iterator         { return _container.cbegin (); }
  [[nodiscard]] inline constexpr auto cend   () const noexcept -> const_iterator         { return _container.cend   (); }
  [[nodiscard]] inline constexpr auto crbegin() const noexcept -> const_reverse_iterator { return _container.crbegin(); }
  [[nodiscard]] inline constexpr auto crend  () const noexcept -> const_reverse_iterator { return _container.crend  (); }
  [[nodiscard]] inline constexpr auto rbegin () const noexcept -> const_reverse_iterator { return _container.rbegin (); }
  [[nodiscard]] inline constexpr auto rend   () const noexcept -> const_reverse_iterator { return _container.rend   (); }
  [[nodiscard]] inline constexpr auto rbegin ()       noexcept ->       reverse_iterator { return _container.rbegin (); }
  [[nodiscard]] inline constexpr auto rend   ()       noexcept ->       reverse_iterator { return _container.rend   (); }


  [[nodiscard]] inline constexpr auto empty () const noexcept   { return _container.empty();    }
  [[nodiscard]] inline constexpr auto size  () const noexcept   { return _container.size();     }
  [[nodiscard]] inline constexpr auto size  (size_type r) const { return _extents.at(r);        }
  [[nodiscard]] inline constexpr auto rank  () const noexcept   { return ublas::size_v<extents_type>; }
  [[nodiscard]] inline constexpr auto order () const noexcept   { return this->rank();                }

  [[nodiscard]] constexpr inline auto const& strides   () const noexcept{ return _strides; }
  [[nodiscard]] inline constexpr auto const& extents   () const noexcept{ return _extents; }
  [[nodiscard]] inline constexpr const_pointer data    () const noexcept{ return _container.data();}
  [[nodiscard]] inline constexpr pointer       data    ()       noexcept{ return _container.data();}
  [[nodiscard]] inline constexpr auto const&   base    () const noexcept{ return _container; }




private:
  static constexpr extents_type _extents = extents_type{};
  static constexpr strides_type _strides = to_strides_v<extents_type,layout_type>;
  container_type _container;
};



//template<class V, class E, class L = layout::first_order>
//static constexpr inline auto make_tensor(
//  typename tensor_static<V,E,L>::base_type    && a,
//  typename tensor_static<V,E,L>::extents_type && /*unused*/,
//  typename tensor_static<V,E,L>::layout_type  && /*unused*/)
//{
//  return tensor_static<V,E,L>( a );
//}


} // namespace boost::numeric::ublas


namespace boost::numeric::ublas{

template<class V, class E, class L = layout::first_order>
using tensor_static = tensor_core<tensor_engine<E, L, std::array<V, product_v<E>>>>;

}

namespace boost::numeric::ublas::experimental
{
template<class V, std::size_t n1, std::size_t n2, class L = layout::first_order>
using matrix_static = tensor_static<V, extents<n1,n2>, L>;

template<class V, std::size_t n1, class L = layout::first_order>
using vector_static = tensor_static<V, extents<n1>, L>;
} // namespace boost::numeric::ublas::experimental

#endif
