//
//  Copyright (c) 2018-2019, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef BOOST_UBLAS_TENSOR_STORAGE_HPP
#define BOOST_UBLAS_TENSOR_STORAGE_HPP

#include "extents.hpp"
#include <algorithm>
#include <array>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include "extents_functions.hpp"

namespace boost::numeric::ublas::storage {

struct tensor_storage {};
struct sparse_storage : tensor_storage {};
struct band_storage : tensor_storage {};
struct dense_storage : tensor_storage {};

namespace sparse_tensor {

template <typename T, typename A = std::allocator<std::pair<const ptrdiff_t, T>>>
struct compressed_map : sparse_storage {
  using base_type = std::unordered_map<std::ptrdiff_t, T, std::hash<ptrdiff_t>,
                                       std::equal_to<ptrdiff_t>, A>;
  using value_type = T;
  using key_type = typename base_type::key_type;
  using hasher = typename base_type::hasher;
  using allocator_type = typename base_type::allocator_type;

  using pair_type = typename base_type::value_type;
  using size_type = typename base_type::size_type;
  using difference_type = typename base_type::difference_type;

  using reference = value_type &;
  using const_reference = value_type const &;

  using pointer = typename base_type::pointer;
  using const_pointer = typename base_type::const_pointer;

  using reverse_iterator = void;
  using const_reverse_iterator = void;

  using iterator = typename base_type::iterator;
  using const_iterator = typename base_type::const_iterator;

  compressed_map() = default;

  compressed_map(size_type size) : size_(size) {}
  compressed_map(size_type size, value_type val) : size_(size) {
      if(val != value_type{0}){
          for(auto i = 0; i < size_; i++){
              data_[i] = val;
          }
      }
  }

  template<typename A2>
  compressed_map( std::vector<T,A2> const& v) : size_(v.size()){
    for(auto i = 0u; i < v.size(); i++){
      if(v[i] != value_type{0}){
        data_.insert({i,v[i]});
      }
    }
  }

  template<typename A2>
  compressed_map( std::vector<T,A2> && v) : size_(v.size()){
    for(auto i = 0u; i < v.size(); i++){
      if(v[i] != value_type{0}){
        data_.insert({i,std::move(v[i])});
      }
    }
  }

  compressed_map( std::initializer_list<std::pair<key_type, value_type>> l) : size_(l.size()){
    for(auto const& v : l){
      if(v.second != value_type{0}){
        data_.insert(v);
      }
    }
  }

  compressed_map(compressed_map const &other)
      : size_(other.size_), data_(other.data_), k_(other.k_), v_(other.v_) {
    update_map();
  }

  compressed_map(compressed_map &&other)
      : size_(other.size_), data_(std::move(other.data_)), k_(other.k_),
        v_(other.v_) {
    update_map();
  }

  compressed_map &operator=(compressed_map const &other) {
    auto temp = compressed_map(other);
    swap(*this, temp);
    return *this;
  }

  compressed_map &operator=(compressed_map &&other) {
    auto temp = compressed_map(std::move(other));
    swap(*this, temp);
    return *this;
  }

  compressed_map(const_iterator b, const_iterator e)
      : data_(b, e), size_(std::distance(b, e)) {
        prune();
      }

  ~compressed_map() = default;

  constexpr iterator begin() noexcept {
    update_map();
    return data_.begin();
  }

  constexpr iterator end() noexcept {
    update_map();
    return data_.end();
  }

  constexpr const_iterator cbegin() noexcept {
    update_map();
    return data_.cbegin();
  }

  constexpr const_iterator cend() noexcept {
    update_map();
    return data_.cend();
  }

  auto to_vector(){
    std::vector<T> temp(size_);
    for(auto i = 0u; i < size_; i++){
      temp[i] = this->at(i);
    }
    return temp;
  }

  constexpr auto empty() const noexcept{
    return data_.empty();
  }

  friend auto swap(compressed_map &lhs, compressed_map &rhs) {
    lhs.update_map();
    rhs.update_map();
    std::swap(lhs.size_, rhs.size_);
    std::swap(lhs.data_, rhs.data_);
  }

  constexpr auto size() const noexcept { return size_; }
  constexpr auto container_size() noexcept {
    update_map();
    return data_.size();
  }

  auto prune(value_type val = value_type{0}) noexcept {
    update_map();
    std::vector<key_type> keys;
    for (auto i = begin(); i != end(); i++) {
      if (i->second == val) {
        keys.push_back(i->first);
      }
    }
    for (auto const &key : keys) {
      data_.erase(key);
    }
  }

  constexpr auto contains(key_type k) const noexcept {
    update_map();
    return data_.find(k) != data_.end();
  }

  constexpr reference at(key_type k) noexcept {
    if (!contains(k)) {
      k_ = k;
      return v_;
    }
    return data_.at(k);
  }

  constexpr const_reference at(key_type k) const noexcept {
    if (!contains(k)) {
      return v_;
    }
    return data_.at(k);
  }

  constexpr reference operator[](key_type k) {
    if (!contains(k)) {
      k_ = k;
      return v_;
    }
    return data_[k];
  }

  constexpr const_reference operator[](key_type k) const noexcept {
    return this->at(k);
  }

private:
  constexpr auto update_map() noexcept {
    if (k_ != key_type{-1} && v_ != value_type{0} ) {
      data_[k_] = v_;
      k_ = key_type{-1};
      v_ = value_type{0};
    }
  }

  constexpr auto update_map() const noexcept {
    auto& self = const_cast<compressed_map&>(*this);
    if (k_ != key_type{-1} && v_ != value_type{0}) {
      self.data_[k_] = self.v_;
      self.k_ = key_type{-1};
      self.v_ = value_type{0};
    }
  }

private:
  size_type size_{0};
  key_type k_{-1};
  value_type v_{0};
  base_type data_;
};

} // namespace sparse_tensor

namespace dense_tensor {

template <typename E>
inline static constexpr bool is_static_v =
    boost::numeric::ublas::detail::is_static<E>::value;

template <typename E>
inline static constexpr bool is_dynamic_v =
    boost::numeric::ublas::detail::is_dynamic<E>::value;

template <typename T, typename E, typename A>
struct default_storage<T, E, A,
                       typename std::enable_if<is_dynamic_v<E>>::type> {
  using type = std::vector<T, A>;
};

template <typename T, typename E, typename A>
struct default_storage<T, E, A, typename std::enable_if<is_static_v<E>>::type> {
  using type = std::array<T, product(E{})>;
};

template <typename T, typename E, typename A>
using default_storage_t = typename default_storage<T, E, A>::type;

} // namespace dense_tensor

} // namespace boost::numeric::ublas::storage

#endif