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

/** @brief basic_slice specialization which inherits from slice_helper for static slice
 * 
 * @code 
 * using s1 = slice<0,10,2>; 
 * using s2 = slice<0,-2,2>;
 * @endcode
 * 
 * @tparam T slice type
 * @tparam f_ starting index of slice
 * @tparam l_ ending index of slice
 * @tparam s_ steps for slice
 * 
 */
template <typename T, ptrdiff_t f_, ptrdiff_t l_, ptrdiff_t s_>
struct basic_slice<T, f_, l_, s_> : detail::slice_helper_t < T, f_, l_, s_>
{
    using self_type = detail::slice_helper_t < T, f_, l_, s_>;
    template<typename U, ptrdiff_t... Args> 
    
    TENSOR_AUTO_CONSTEXPR_RETURN operator==( basic_slice<U,Args...> const& rhs ) const noexcept{
        return self_type::first() == rhs.first() && self_type::last() == rhs.last() && self_type::step() == rhs.step() && self_type::size() == rhs.size();
    }

};

/** @brief basic_slice specialization which inherits from slice_helper for static slice
 * 
 * @code 
 * using s1 = slice<0,10>; 
 * using s2 = slice<0,-2>;
 * @endcode
 * 
 * @tparam T slice type
 * @tparam f_ starting index of slice
 * @tparam l_ ending index of slice
 * 
 */
template <typename T, ptrdiff_t f_, ptrdiff_t l_>
struct basic_slice<T, f_, l_> : detail::slice_helper_t <T, f_, l_, 1l>
{
    using self_type = detail::slice_helper_t <T, f_, l_, 1l>;
    template<typename U, ptrdiff_t... Args> 
    TENSOR_AUTO_CONSTEXPR_RETURN operator==( basic_slice<U,Args...> const& rhs ) const noexcept{
        return self_type::first() == rhs.first() && self_type::last() == rhs.last() && self_type::step() == rhs.step() && self_type::size() == rhs.size();
    }
};

/** @brief basic_slice specialization which inherits from slice_helper for static slice
 * 
 * @code 
 * using s1 = slice<0>; 
 * using s2 = slice<-5>;
 * @endcode
 * 
 * @tparam T slice type
 * @tparam N start and end of slice with step 1
 * 
 */
template <typename T, ptrdiff_t N>
struct basic_slice<T, N> : detail::slice_helper_t <T, N, N, 1l>
{
    using self_type = detail::slice_helper_t <T, N, N, 1l>;
    template<typename U, ptrdiff_t... Args> 
    TENSOR_AUTO_CONSTEXPR_RETURN operator==( basic_slice<U,Args...>& rhs ) const noexcept{
        return self_type::first() == rhs.first() && self_type::last() == rhs.last() && self_type::step() == rhs.step() && self_type::size() == rhs.size();
    }
};

/** @brief basic_slice specialization for dynamic slice
 * 
 * @tparam T slice type
 * 
 */
template <typename T>
struct basic_slice<T>
{
    using self_type = basic_slice<T>;
    using value_type = T;
    using size_type = size_t;


    /** @brief default constructor for dynamic slice
     * 
     * @code
     * auto s1 = basic_slice<ptrdiff_t>{};
     * @endcode
     * 
     */
    constexpr basic_slice()
        : first_{}, last_{}, step_{1}, size_{-1}
    {
    }


    /** @brief construct a dynamic slice with first, last and step
     * 
     * @code
     * auto s1 = basic_slice<ptrdiff_t>{0,10,3};
     * auto s2 = basic_slice<ptrdiff_t>{0,10};
     * @endcode
     * 
     * @param f first or starting index
     * @param l last or ending index
     * @param s step which has default value of 1
     * 
     */
    constexpr basic_slice(value_type f, value_type l, value_type s = 1)
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
                size_ = 0l;
            }

        }
    }



    /** @brief construct a dynamic slice with single index
     * 
     * @code
     * auto s1 = basic_slice<ptrdiff_t>{3};
     * auto s2 = basic_slice<ptrdiff_t>{-3};
     * @endcode
     * 
     * @param n index of extent 
     * 
     */
    constexpr basic_slice(value_type n)
        : basic_slice(n, n, 1)
    {
    }

    /** Copy Constructor */
    template<typename U>
    constexpr basic_slice(basic_slice<U> const &other)
        : first_(other.first_), last_(other.last_), step_(other.step_), size_(other.size_)
    {
    }
    
    /** Move Constructor */
    template<typename U>
    constexpr basic_slice( basic_slice<U>&& other) 
        : first_(std::move(other.first_))
        , last_(std::move(other.last_))
        , step_(std::move(other.step_))
        , size_(std::move(other.size_))
        {}

    template<typename U>
    basic_slice &operator=(basic_slice<U> const &other)
    {
        auto temp = self_type(other);
        swap(*this,temp);
        return *this;
    }


    template<typename U>
    basic_slice &operator=(basic_slice<U>&& other)
    {
        auto temp = self_type(std::move(other));
        swap(*this,temp);
        return *this;
    }

    /** @brief returns the starting of slice */ 
    TENSOR_AUTO_CONSTEXPR_RETURN first() const noexcept { return first_; }
    
    /** @brief returns the ending of slice */ 
    TENSOR_AUTO_CONSTEXPR_RETURN last() const noexcept { return last_; }
    
    /** @brief returns the step of slice */ 
    TENSOR_AUTO_CONSTEXPR_RETURN step() const noexcept { return step_; }
    
    /** @brief returns the size of slice */ 
    TENSOR_CONSTEXPR_RETURN(size_type) size() const noexcept { return size_ == -1 ? 0 : size_; }
    
    /** @brief returns true if slice is empty or false */ 
    TENSOR_AUTO_CONSTEXPR_RETURN empty() const noexcept{ return size_ == -1;}

    ~basic_slice() = default;

    // /** @brief returns the relative address of next element 
    //  *
    //  * @param idx index of element
    //  *  
    //  */  
    // TENSOR_CONSTEXPR_RETURN(value_type) operator[](size_type idx) const
    // {
    //     return first_ + idx * step_;
    // }

    // /** @brief caluates the next slice
    //  *
    //  * @param rhs of type basic_slice<T>
    //  *  
    //  */ 
    // TENSOR_AUTO_CONSTEXPR_RETURN operator()(basic_slice<T> const &rhs) const
    // {
    //     auto const &lhs = *this;
    //     return basic_slice<T>{
    //         rhs.first() * lhs.step() + lhs.first(),
    //         rhs.last() * lhs.step() + lhs.first(),
    //         lhs.step() * rhs.step()};
    // }

    /** @brief prints the slice */
    friend std::ostream& operator<<(std::ostream& os, basic_slice const& rhs){
        os<<"slice( "<<rhs.first()<<", "<<rhs.last()<<", "<<rhs.step()<<" )";
        return os;
    } 

    template<typename U, ptrdiff_t... Args> 
    TENSOR_AUTO_CONSTEXPR_RETURN operator==( basic_slice<U,Args...> const& rhs ){
        return this->first() == rhs.first() && this->last() == rhs.last() && this->step() == rhs.step() && this->size() == rhs.size();
    }

    friend auto swap(basic_slice& lhs, basic_slice& rhs){
        std::swap( lhs.first_, rhs.first_ );
        std::swap( lhs.last_, rhs.last_ );
        std::swap( lhs.step_, rhs.step_ );
        std::swap( lhs.size_, rhs.size_ );
    }

private:
    value_type first_, last_, step_, size_;
};

/** @brief type alias for basic_slice<ptrdiff_t,Args...> */
template <ptrdiff_t... Args>
using slice = basic_slice<ptrdiff_t, Args...>;

} // namespace boost::numeric::ublas::span

#endif