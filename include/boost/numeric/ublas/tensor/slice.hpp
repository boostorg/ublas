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

#ifndef _BOOST_UBLAS_TENSOR_SLICE_HPP
#define _BOOST_UBLAS_TENSOR_SLICE_HPP

#include "slice_helper.hpp"
#include <cmath>

namespace boost::numeric::ublas::span
{

template <typename T, ptrdiff_t f_, ptrdiff_t l_, ptrdiff_t s_, ptrdiff_t sz_>
struct slice_helper<T, f_, l_, s_, sz_>
{
    using self_type = slice_helper;
    using value_type = T;
    using size_type = size_t;

    static constexpr value_type first_ = f_;
    static constexpr value_type last_ = l_;
    static constexpr value_type step_ = s_;
    static constexpr value_type size_ = sz_;

    static constexpr auto &first() noexcept
    {
        return self_type::first_;
    }

    static constexpr auto &last() noexcept
    {
        return self_type::last_;
    }

    static constexpr auto &step() noexcept
    {
        return self_type::step_;
    }

    static constexpr auto &size() noexcept
    {
        return self_type::size_;
    }

    value_type operator[](std::size_t idx) const
    {
        return first_ + idx * step_;
    }

    template <ptrdiff_t... Args>
    constexpr decltype(auto) operator()(basic_slice<T, Args...> const &) const
    {
        using lhs_type = self_type;
        using rhs_type = typename basic_slice<T, Args...>::self_type;
        return basic_slice<T,
                           rhs_type::first() * lhs_type::step() + lhs_type::first(),
                           rhs_type::last() * lhs_type::step() + lhs_type::first(),
                           lhs_type::step() * rhs_type::step()>{};
    }

    friend std::ostream& operator<<(std::ostream& os, self_type const& rhs){
        os<<"slice( "<<rhs.first()<<", "<<rhs.last()<<", "<<rhs.step()<<" )";
        return os;
    } 
};

template <typename T, ptrdiff_t f_, ptrdiff_t l_, ptrdiff_t s_>
struct basic_slice<T, f_, l_, s_> : detail::slice_helper_t<T, f_, l_, s_>
{
};

template <typename T, ptrdiff_t f_, ptrdiff_t l_>
struct basic_slice<T, f_, l_> : detail::slice_helper_t<T, f_, l_, 1>
{
};

template <typename T, ptrdiff_t N>
struct basic_slice<T, N> : detail::slice_helper_t<T, N, N, 1>
{
};

template <typename T>
struct basic_slice<T>
{
    using value_type = T;
    using size_type = size_t;

    constexpr explicit basic_slice()
        : first_{}, last_{}, step_{1}, size_{-1}
    {
    }

    basic_slice(value_type f, value_type l, value_type s = 1)
        : first_(f), last_(l), step_(s)
    {
        if (f == l)
        {
            size_ = value_type(1);
        }
        else
        {
            if ( s < 0 ){
                throw std::runtime_error("Error in boost::numeric::ublas::basic_slice: step is < 0");
            }

            if (step_ == 0 && f != l)
                throw std::runtime_error("Error in boost::numeric::ublas::basic_slice : cannot have a step_ equal to zero.");

            if ( f >= 0 && l >= 0 ){
                if (f > l)
                throw std::runtime_error("Error in boost::numeric::ublas::basic_slice: last_ is smaller than first");

                last_ = l - ((l - f) % abs(step_));
                size_ = ( (last_ - first_) / abs(step_) ) + value_type(1);
            } else{
                size_ = 1l;
            }
        }
    }

    basic_slice(value_type n)
        : basic_slice(n, n, 1)
    {
    }

    basic_slice(basic_slice const &other)
        : first_(other.first_), last_(other.last_), step_(other.step_), size_(other.size_)
    {
    }

    basic_slice &operator=(basic_slice const &other)
    {
        first_ = other.first_;
        last_ = other.last_;
        step_ = other.step_;
        size_ = other.size_;
        return *this;
    }

    TENSOR_AUTO_CONSTEXPR_RETURN first() const noexcept { return first_; }
    TENSOR_AUTO_CONSTEXPR_RETURN last() const noexcept { return last_; }
    TENSOR_AUTO_CONSTEXPR_RETURN step() const noexcept { return step_; }
    TENSOR_CONSTEXPR_RETURN(size_type) size() const noexcept { return size_; }
    TENSOR_AUTO_CONSTEXPR_RETURN empty() const noexcept{ return size_ == -1;}

    ~basic_slice() = default;

    value_type operator[](std::size_t idx) const
    {
        return first_ + idx * step_;
    }

    constexpr decltype(auto) operator()(basic_slice<T> const &rhs) const
    {
        auto const &lhs = *this;
        return basic_slice<T>{
            rhs.first() * lhs.step() + lhs.first(),
            rhs.last() * lhs.step() + lhs.first(),
            lhs.step() * rhs.step()};
    }

    friend std::ostream& operator<<(std::ostream& os, basic_slice const& rhs){
        os<<"slice( "<<rhs.first()<<", "<<rhs.last()<<", "<<rhs.step()<<" )";
        return os;
    } 

private:
    value_type first_, last_, step_, size_;
};

template <ptrdiff_t... Args>
using slice = basic_slice<ptrdiff_t, Args...>;

} // namespace boost::numeric::ublas::span

#endif