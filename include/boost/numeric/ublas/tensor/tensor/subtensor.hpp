//
//  Copyright (c) 2021, Kannav Mehta, kmkannavkmehta@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//:


/// \file subtensor.hpp Definition for the subtensor template class

#ifndef BOOST_UBLAS_SUBTENSOR_HPP
#define BOOST_UBLAS_SUBTENSOR_HPP

#include "../access.hpp"
#include "../algorithms.hpp"
#include "../concepts.hpp"
#include "../expression.hpp"
#include "../expression_evaluation.hpp"
#include "../extents.hpp"
#include "../index.hpp"
#include "../index_functions.hpp"
#include "../layout.hpp"
#include "../span.hpp"
#include "../tags.hpp"
#include "../traits/read_write_traits.hpp"
#include "../type_traits.hpp"
#include "subtensor_engine.hpp"
#include "subtensor_utility.hpp"
#include "tensor_engine.hpp"

#include <initializer_list>

namespace boost::numeric::ublas {

template <class T>
class tensor_core<subtensor_engine<T>>
  : public detail::tensor_expression<tensor_core<subtensor_engine<T>>,
                                     tensor_core<subtensor_engine<T>>> {
public:
  using engine_type = subtensor_engine<T>;
  using self_type   = tensor_core<engine_type>;

  template <class derived_type>
  using tensor_expression_type =
    detail::tensor_expression<self_type, derived_type>;

  template <typename container> struct subtensor_iterator {
  };

  static constexpr bool is_const =
    std::is_const<std::remove_reference_t<T>>::value;

  using container_type = typename engine_type::container_type;
  using layout_type    = typename engine_type::layout_type;
  using extents_type   = typename engine_type::extents_type;
  using strides_type   = typename extents_type::base_type;

  using container_traits_type = container_traits<container_type>;

  using size_type       = typename container_traits_type::size_type;
  using difference_type = typename container_traits_type::difference_type;
  using value_type      = typename container_traits_type::value_type;

  using reference = std::conditional_t<is_const,
                       typename container_traits_type::const_reference,
                       typename container_traits_type::reference>;
  using const_reference = typename container_traits_type::const_reference;

  using pointer = std::conditional_t<is_const,
                       typename container_traits_type::const_pointer,
                       typename container_traits_type::pointer>;
  using const_pointer = typename container_traits_type::const_pointer;

  using iterator = typename self_type::subtensor_iterator<container_type>;
  using const_iterator =
    typename self_type::subtensor_iterator<container_type> const;

  using reverse_iterator = typename container_traits_type::reverse_iterator;
  using const_reverse_iterator =
    typename container_traits_type::const_reverse_iterator;

  using container_tag = typename container_traits_type::container_tag;
  using resizable_tag = typename container_traits_type::resizable_tag;

  using span_type = span<std::size_t>;

  using subtensor_type = self_type;

  explicit tensor_core() = delete;

  tensor_core(const tensor_core&) = default;

  template <class U, class FS, class... SL>
  tensor_core(U&& t, FS&& first, SL&&... spans)
    : _spans(detail::generate_span_vector<span_type>(t.extents(), std::forward<FS>(first), std::forward<SL>(spans)...))
    , _extents{}
    , _strides(t.strides())
    , _tensor(t)
  {
      _extents = detail::to_extents(_spans);
  }

  /// @brief Default destructor
  ~tensor_core() = default;

  /** @brief Evaluates the tensor_expression and assigns the results to the
   * tensor_core
   *
   * @code A = B + C * 2;  @endcode
   *
   * @note rank and dimension extents of the tensors in the expressions must
   * conform with this tensor_core.
   *
   * @param expr expression that is evaluated.
   */
  template <class derived_type>
  tensor_core& operator=(const tensor_expression_type<derived_type>& expr)
  {
    detail::eval(*this, expr);
    return *this;
  }

  // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
  tensor_core& operator=(tensor_core other) noexcept
  {
    swap(*this, other);
    return *this;
  }

  // tensor_core& operator=(const_reference v)
  // {
  //   std::fill_n(_container.begin(), _container.size(), v);
  //   return *this;
  // }

  /** @brief Element access using a multi-index with bound checking which can
   * throw an exception.
   *
   *  @code auto a = A.at(i,j,k); @endcode
   *
   *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) ==
   * 0, else 0<= i < this->size(0)
   *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r
   * < this->rank()
   */
  template <integral I1, integral I2, integral... Is>
  [[nodiscard]] inline const_reference at(I1 i1, I2 i2, Is... is) const
  {
    if (sizeof...(is) + 2 != this->order()) {
      throw std::invalid_argument(
        "boost::numeric::ublas::tensor_core<tensor_dynamic>::at : "
        "Cannot access tensor with multi-index. "
        "Number of provided indices does not match with tensor order.");
    }
    const auto idx = ublas::detail::to_index(_strides, i1, i2, is...);
    return _tensor.at(idx);
  }

  /** @brief Element access using a multi-index with bound checking which can
   * throw an exception.
   *
   *  @code auto a = A.at(i,j,k); @endcode
   *
   *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) ==
   * 0, else 0<= i < this->size(0)
   *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r
   * < this->rank()
   */
  template <integral I1, integral I2, integral... Is>
  [[nodiscard]] inline reference at(I1 i1, I2 i2, Is... is)
  {
    if (sizeof...(is) + 2 != this->order()) {
      throw std::invalid_argument(
        "boost::numeric::ublas::tensor_core<tensor_dynamic>::at : "
        "Cannot access tensor with multi-index."
        "Number of provided indices does not match with tensor order.");
    }
    const auto idx = ublas::detail::to_index(_strides, i1, i2, is...);
    return _tensor.at(idx);
  }

  /** @brief Element access using a multi-index with bound checking which can
   * throw an exception.
   *
   *  @code auto a = A(i,j,k); @endcode
   *
   *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) ==
   * 0, else 0<= i < this->size(0)
   *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r
   * < this->rank()
   */
  template <integral... Is>
  [[nodiscard]] inline const_reference operator()(Is... is) const
  {
    return this->at(is...);
  }

  /** @brief Element access using a multi-index with bound checking which can
   * throw an exception.
   *
   *  @code auto a = A(i,j,k); @endcode
   *
   *  @param i zero-based index where 0 <= i < this->size() if sizeof...(is) ==
   * 0, else 0<= i < this->size(0)
   *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r
   * < this->rank()
   */
  template <integral... Is>[[nodiscard]] inline reference operator()(Is... is)
  {
    return this->at(is...);
  }

  /** @brief Element access using a single index.
   *
   *  @code auto a = A[i]; @endcode
   *
   *  @param i zero-based index where 0 <= i < this->size()
   */
  [[nodiscard]] inline const_reference operator[](size_type i) const
  {
    const auto idx = detail::compute_single_index(i, _tensor.strides().begin(), _tensor.strides().end(), _strides.begin());
    return this->_tensor[idx];
  }

  /** @brief Element access using a single index.
   *
   *  @code auto a = A[i]; @endcode
   *
   *  @param i zero-based index where 0 <= i < this->size()
   */
  [[nodiscard]] inline reference operator[](size_type i)
  {
    const auto idx = detail::compute_single_index(i, _tensor.strides().begin(), _tensor.strides().end(), _strides.begin());
    return this->_tensor[idx];
  }

  /** @brief Element access using a single-index with bound checking which can
   * throw an exception.
   *
   *  @code auto a = A.at(i);     @endcode
   *
   *  @param i zero-based index where 0 <= i < this->size()
   */
  template <class... Indices>
  [[nodiscard]] inline const_reference at(size_type i) const
  {

    const auto idx = detail::compute_single_index(i, _tensor.strides().begin(), _tensor.strides().end(), _strides.begin());
    return this->_tensor.at(idx);
  }

  /** @brief Read tensor element of a tensor \c t with a single-index \c i
   *
   *  @code auto a = t.at(i);  @endcode
   *
   *  @param i zero-based index where 0 <= i < t.size()
   */
  [[nodiscard]] inline reference at(size_type i)
  {
    const auto idx = detail::compute_single_index(i, _tensor.strides().begin(), _tensor.strides().end(), _strides.begin());
    return this->_tensor.at(idx);
  }

  /** @brief Generates a tensor_core index for tensor_core contraction
   *
   *
   *  @code auto Ai = A(_i,_j,k); @endcode
   *
   *  @param i placeholder
   *  @param is zero-based indices where 0 <= is[r] < this->size(r) where  0 < r
   * < this->rank()
   */
  template <std::size_t I, class... index_types>
  [[nodiscard]] inline decltype(auto) operator()(index::index_type<I> p, index_types... ps) const
  {
    constexpr auto size = sizeof...(ps) + 1;
    if (size != this->order()) {
      throw std::invalid_argument(
        "boost::numeric::ublas::tensor_core <engine_dynamic>: "
        "Cannot multiply using Einstein notation. "
        "Number of provided indices does not match with tensor order.");
    }
    return std::make_pair(std::cref(*this), std::make_tuple(p, std::forward<index_types>(ps)...));
  }

  /**
   * @brief Generates a subtensor from a tensor
   *
   * @code auto Ai = A(span(), span(1,end), span(1,end)); @endcode
   *
   * @tparam f
   * @tparam spans
   */
  template <class... SL>
  [[nodiscard]] inline decltype(auto) operator()(span_type&& s, SL&&... spans) const noexcept
  {
    return subtensor_type(_tensor, _strides, std::forward<span_type>(s), std::forward<SL>(spans)...);
  }

  template <class... SL>
  [[nodiscard]] inline decltype(auto) operator()(span_type&& s, SL&&... spans) noexcept
  {
    return subtensor_type(_tensor, _strides, std::forward<span_type>(s), std::forward<SL>(spans)...);
  }

//   [[nodiscard]] inline auto begin  () const noexcept -> const_iterator { return _container.begin  (); }
//   [[nodiscard]] inline auto end    () const noexcept -> const_iterator { return _container.end    (); }
//   [[nodiscard]] inline auto begin  ()       noexcept ->       iterator { return _container.begin  (); }
//   [[nodiscard]] inline auto end    ()       noexcept ->       iterator { return _container.end    (); }
//   [[nodiscard]] inline auto cbegin () const noexcept -> const_iterator { return _container.cbegin (); }
//   [[nodiscard]] inline auto cend   () const noexcept -> const_iterator { return _container.cend   (); }
//   [[nodiscard]] inline auto crbegin() const noexcept -> const_reverse_iterator { return _container.crbegin(); }
//   [[nodiscard]] inline auto crend  () const noexcept -> const_reverse_iterator { return _container.crend  (); }
//   [[nodiscard]] inline auto rbegin () const noexcept -> const_reverse_iterator { return _container.rbegin (); }
//   [[nodiscard]] inline auto rend   () const noexcept -> const_reverse_iterator { return _container.rend   (); }
//   [[nodiscard]] inline auto rbegin ()       noexcept ->       reverse_iterator { return _container.rbegin (); }
//   [[nodiscard]] inline auto rend   ()       noexcept ->       reverse_iterator { return _container.rend   (); }

  [[nodiscard]] inline auto empty ()            const noexcept { return size() == 0;             }
  [[nodiscard]] inline auto size  ()            const noexcept { return ublas::product(_extents);}
  [[nodiscard]] inline auto size  (size_type r) const          { return _extents.at(r);          }
  [[nodiscard]] inline auto rank  ()            const          { return _extents.size();         }
  [[nodiscard]] inline auto order ()            const          { return this->rank();            }

  [[nodiscard]] inline auto const& strides () const noexcept                  { return _strides; }
  [[nodiscard]] inline auto const& extents () const noexcept                  { return _extents; }
  [[nodiscard]] inline auto        data    () const noexcept -> const_pointer { return _tensor.data();}
  [[nodiscard]] inline auto        data    ()       noexcept -> pointer       { return _tensor.data();}
  [[nodiscard]] inline auto const& base    () const noexcept                  { return _tensor.container(); }

private:
  /**
   * @brief There might be cases where spans cannot be computed on creation
   */
  std::vector<span_type> _spans;
  extents_type _extents;
  strides_type _strides;
  T& _tensor;
};


}   // namespace boost::numeric::ublas

#endif
