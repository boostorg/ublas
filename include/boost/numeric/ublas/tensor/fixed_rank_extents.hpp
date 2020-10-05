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


#ifndef _BOOST_NUMERIC_UBLAS_TENSOR_FIXED_RANK_EXTENTS_HPP_
#define _BOOST_NUMERIC_UBLAS_TENSOR_FIXED_RANK_EXTENTS_HPP_

#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <array>
#include <boost/numeric/ublas/tensor/type_traits.hpp>
#include <boost/numeric/ublas/tensor/extents_functions.hpp>

namespace boost::numeric::ublas {

/** @brief Template class for storing tensor extents for compile time.
 *
 * @code basic_static_extents<1,2,3,4> t @endcode
 * @tparam E parameter pack of extents
 *
 */
template <class ExtentsType, std::size_t N>
class basic_fixed_rank_extents
{

public:

    static constexpr std::size_t const _size = N;

    using base_type       = std::array<ExtentsType,_size>;
    using value_type      = typename base_type::value_type;
    using size_type       = typename base_type::size_type;
    using reference       = typename base_type::reference;
    using const_reference = typename base_type::const_reference;
    using const_pointer   = typename base_type::const_pointer;
    using const_iterator  = typename base_type::const_iterator;
    using const_reverse_iterator = typename base_type::const_reverse_iterator;

    static_assert( std::numeric_limits<value_type>::is_integer, "Static error in basic_fixed_rank_extents: type must be of type integer.");
    static_assert(!std::numeric_limits<value_type>::is_signed,  "Static error in basic_fixed_rank_extents: type must be of type unsigned integer.");

    //@returns the rank of basic_static_extents
    [[nodiscard]] 
    static constexpr size_type size() noexcept { return _size; }

    [[nodiscard]] inline
    constexpr const_reference at(size_type k) const{ 
        return _base.at(k);
    }

    [[nodiscard]] inline
    constexpr reference at(size_type k){ 
        return _base.at(k);
    }

    [[nodiscard]] inline
    constexpr const_reference operator[](size_type k) const noexcept{ 
        return _base[k];
    }

    [[nodiscard]] inline
    constexpr reference operator[](size_type k) noexcept{ 
        return _base[k];
    }

    constexpr basic_fixed_rank_extents() = default;
    
    constexpr basic_fixed_rank_extents(basic_fixed_rank_extents const& other)
        : _base(other._base)
    {}
    
    constexpr basic_fixed_rank_extents(basic_fixed_rank_extents && other) noexcept
        : _base( std::move(other._base) )
    {}
    
    constexpr basic_fixed_rank_extents& operator=(basic_fixed_rank_extents const& other)
        noexcept(std::is_nothrow_swappable_v<base_type>)
    {
        basic_fixed_rank_extents temp(other);
        swap(*this,temp);
        return *this;
    }
    
    constexpr basic_fixed_rank_extents& operator=(basic_fixed_rank_extents && other)
        noexcept(std::is_nothrow_swappable_v<base_type>)
    {
        swap(*this,other);
        return *this;
    }

    ~basic_fixed_rank_extents() = default;
    
    constexpr basic_fixed_rank_extents(std::initializer_list<value_type> li){
        if( li.size() > _size ){
            throw std::out_of_range("boost::numeric::ublas::basic_fixed_rank_extents(std::initializer_list<value_type>): "
                "number of elements in std::initializer_list is greater than the size"
            );
        }
        
        std::copy(li.begin(), li.end(), _base.begin());

        if ( !is_valid(*this) ){
            throw std::length_error("Error in basic_fixed_rank_extents::basic_fixed_rank_extents() : "
                "shape tuple is not a valid permutation: has zero elements."
            );
        }
    }
    
    constexpr basic_fixed_rank_extents(const_iterator begin, const_iterator end){
        if( std::distance(begin,end) < 0 || static_cast<std::size_t>(std::distance(begin,end)) > _size){
            throw std::out_of_range("boost::numeric::ublas::basic_fixed_rank_extents(): initializer list size is greater than the rank");
        }
        
        std::copy(begin, end, _base.begin());

        if ( !is_valid(*this) ){
            throw std::length_error("Error in basic_fixed_rank_extents::basic_fixed_rank_extents(const_iterator,const_iterator) : "
                "shape tuple is not a valid permutation: has zero elements."
            );
        }
    }
    
    inline
    constexpr void fill( value_type value ){
        _base.fill(value);
    }
    
    template<typename OtherExtents>
    constexpr basic_fixed_rank_extents(OtherExtents const& e){
        static_assert( is_extents_v<OtherExtents>, "boost::numeric::ublas::basic_fixed_rank_extents(OtherExtents const&) : " 
            "OtherExtents should be a valid tensor extents"
        );

        if constexpr( is_static_rank_v< OtherExtents > ){
            static_assert( OtherExtents::_size == _size, 
                "basic_fixed_rank_extents::basic_fixed_rank_extents(OtherExtents const&) : "
                "unequal rank found, rank should be equal"
            );
        }else{
            if( e.size() != size() ){
                throw std::length_error("Error in basic_fixed_rank_extents::basic_fixed_rank_extents(OtherExtents const&) : "
                    "unequal rank found, rank should be equal"
                );
            }
        }

        std::copy_n(e.begin(),_size, _base.begin());
    }
    
    constexpr basic_fixed_rank_extents(base_type const& data)
        : _base(data)
    {
        if ( !is_valid(*this) ){
            throw std::length_error("Error in basic_fixed_rank_extents::basic_fixed_rank_extents(base_type const&) : "
                "shape tuple is not a valid permutation: has zero elements."
            );
        }
    }
    
    constexpr basic_fixed_rank_extents(base_type&& data)
        : _base(std::move(data))
    {
        if ( !is_valid(*this) ){
            throw std::length_error("Error in basic_fixed_rank_extents::basic_fixed_rank_extents(base_type &&) : "
                "shape tuple is not a valid permutation: has zero elements."
            );
        }
    }

    /** @brief Returns the std::vector containing extents */
    [[nodiscard]] inline
    constexpr base_type const& base() const noexcept{
        return _base;
    }

    /** @brief Checks if extents is empty or not
     *
     * @returns true if rank is 0 else false
     *
     */
    [[nodiscard]] inline
    constexpr bool empty() const noexcept { return _size == size_type{0}; }

    friend void swap(basic_fixed_rank_extents& lhs, basic_fixed_rank_extents& rhs) 
        noexcept(std::is_nothrow_swappable_v<base_type>)
    {
        std::swap(lhs._base   , rhs._base   );
    }

    [[nodiscard]] inline
    constexpr const_pointer data() const noexcept
    {
        return _base.data();
    }

    [[nodiscard]] inline
    constexpr const_iterator
    begin() const noexcept
    {
        return _base.begin();
    }

    [[nodiscard]] inline
    constexpr const_iterator
    end() const noexcept
    {
        return _base.end();
    }
    
    [[nodiscard]] inline
    constexpr const_reference back () const
    {
        return _base.back();
    }

    [[nodiscard]] inline
    constexpr const_reverse_iterator
    rbegin() const noexcept
    {
        return _base.rbegin();
    }

    [[nodiscard]] inline
    constexpr const_reverse_iterator
    rend() const noexcept
    {
        return _base.rend();
    }

private:
    base_type _base{};
};

} // namespace boost::numeric::ublass



#endif
