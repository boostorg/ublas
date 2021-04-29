//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef BOOST_UBLAS_TENSOR_MULTI_INDEX_HPP
#define BOOST_UBLAS_TENSOR_MULTI_INDEX_HPP



#include <array>
#include <cstddef>
#include <vector>

#include "index.hpp"
#include "multi_index_utility.hpp"


namespace boost::numeric::ublas {

/** @brief Proxy class for the einstein summation notation
 *
 * Denotes an array of index_type types ::_a for 0<=K<=16 is used in tensor::operator()
*/
template<std::size_t N>
class multi_index
{
public:
    multi_index() = delete;

    template<std::size_t I, class ... indexes>
    constexpr explicit inline
      multi_index(index::index_type<I> const& i, indexes ... is )
      : _base{i(), is()... }
    {
        static_assert( sizeof...(is)+1 == N,
                       "Static assert in boost::numeric::ublas::multi_index: number of constructor arguments is not equal to the template parameter." );

        static_assert( valid_multi_index<std::tuple<index::index_type<I>, indexes ...> >::value,
                       "Static assert in boost::numeric::ublas::multi_index: indexes occur twice in multi-index." );
    }

    multi_index(multi_index const& other) = default;
    multi_index(multi_index&& other) noexcept = default ;

    multi_index& operator=(multi_index other)
    {
      std::swap(this->_base,other._base);
      return *this;
    }

    multi_index& operator=(multi_index&& other) noexcept
    {
      this->_base = std::move(other._base);
      return *this;
    }

    ~multi_index() = default;

    [[nodiscard]] inline           auto const& base()             const { return _base;        }
    [[nodiscard]] inline constexpr auto size()                    const { return _base.size(); }
    [[nodiscard]] inline constexpr auto at(std::size_t i)         const { return _base.at(i);  }
    [[nodiscard]] inline constexpr auto operator[](std::size_t i) const { return _base.at(i);  }

private:
    std::array<std::size_t, N> _base;
};

template<std::size_t K, std::size_t N>
inline constexpr auto get(multi_index<N> const& m) { return std::get<K>(m.base()); }

template<std::size_t M, std::size_t N>
auto array_to_vector(multi_index<M> const& lhs, multi_index<N> const& rhs)
{
    using vtype = std::vector<std::size_t>;

    auto pair_of_vector = std::make_pair( vtype {}, vtype{}  );

    for(auto i = 0ul; i < N; ++i){
        for(auto j = 0ul; j < M; ++j){
            if ( lhs.at(i) == rhs.at(j) && lhs.at(i) != boost::numeric::ublas::index::_()){
                pair_of_vector.first .push_back( i+1 );
                pair_of_vector.second.push_back( j+1 );
            }
        }
    }
    return pair_of_vector;
}

} // namespace boost::numeric::ublas

#endif // MULTI_INDEX_HPP
