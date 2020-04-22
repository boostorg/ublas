//
//  Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google
//


#ifndef BOOST_UBLAS_FIXED_RANK_TENSOR_IMPL_HPP
#define BOOST_UBLAS_FIXED_RANK_TENSOR_IMPL_HPP

#include <boost/numeric/ublas/tensor/tensor.hpp>

namespace boost::numeric::ublas {

    template< typename T, std::size_t R, typename F = first_order >
    struct fixed_rank_tensor:
            public basic_tensor< fixed_rank_tensor< T, R, F > >
    {
        using self_type                 = fixed_rank_tensor< T, R, F >;
        using super_type                = basic_tensor< self_type >;

        template<class derived_type>
        using tensor_expression_type    = typename super_type::template tensor_expression_type<derived_type>;

        template<class derived_type>
        using matrix_expression_type    = typename super_type::template matrix_expression_type<derived_type>;

        template<class derived_type>
        using vector_expression_type    = typename super_type::template vector_expression_type<derived_type>;

        using array_type                = typename tensor_traits<self_type>::container_type;
        using layout_type               = typename tensor_traits<self_type>::layout_type;

        using size_type                 = typename array_type::size_type;
        using difference_type           = typename array_type::difference_type;
        using value_type                = typename array_type::value_type;

        using reference                 = typename array_type::reference;
        using const_reference           = typename array_type::const_reference;

        using pointer                   = typename array_type::pointer;
        using const_pointer             = typename array_type::const_pointer;

        using iterator                  = typename array_type::iterator;
        using const_iterator            = typename array_type::const_iterator;

        using reverse_iterator          = typename array_type::reverse_iterator;
        using const_reverse_iterator    = typename array_type::const_reverse_iterator;

        using tensor_temporary_type     = self_type;
        using storage_category          = dense_tag;

        using extents_type              = typename tensor_traits<self_type>::extents_type;
        using strides_type              = strides_t<extents_type,layout_type>;

        using matrix_type               = typename super_type::matrix_type;
        using vector_type               = typename super_type::vector_type;

        constexpr fixed_rank_tensor() = default;

        constexpr fixed_rank_tensor( extents_type const& e, value_type const& i )
            : super_type(e,i)
        {}

        constexpr fixed_rank_tensor( extents_type const& e, array_type const& a )
            : super_type(e,a)
        {}

        constexpr fixed_rank_tensor( extents_type const& e )
            : super_type(e)
        {}

        fixed_rank_tensor( std::initializer_list< size_type > li )
            : super_type( std::move(li) )
        {}

        constexpr fixed_rank_tensor( matrix_type const& v )
            : super_type( extents_type{v.size1(), v.size2()} )
        {
            static_assert( R == 2, "Error in boost::numeric::ublas::tensor(const matrix &v)"
                                            " : rank of extents are not correct, please check!");

            std::copy(v.data().begin(), v.data().end(), super_type::begin());
        }

        constexpr fixed_rank_tensor( matrix_type && v )
            : super_type( extents_type{v.size1(), v.size2()} )
        {
            
            static_assert( R == 2, "Error in boost::numeric::ublas::tensor(matrix &&v)"
                                            " : rank of extents are not correct, please check!");

            std::move(v.data().begin(), v.data().end(),super_type::begin());
        }

        constexpr fixed_rank_tensor (const vector_type &v)
            : super_type( extents_type{ v.size(), typename extents_type::value_type{1} } )
        {
            static_assert( R == 2, "Error in boost::numeric::ublas::tensor(const vector &v)"
                                            " : rank of extents are not correct, please check!");

            std::copy(v.data().begin(), v.data().end(), super_type::begin());
            
        }

        constexpr fixed_rank_tensor (vector_type &&v)
            : super_type( extents_type{ v.size(), typename extents_type::value_type{1} } )
        {
            
            static_assert( R == 2, "Error in boost::numeric::ublas::tensor(vector &&v)"
                                            " : rank of extents are not correct, please check!");

            std::move(v.data().begin(), v.data().end(),super_type::begin());
            
        }

        template<class OtherLayout>
        constexpr fixed_rank_tensor (const fixed_rank_tensor<value_type, R, OtherLayout> &expr)
            : super_type(static_cast<super_type const&>(expr))
        {}

        template<class derived_type>
        constexpr fixed_rank_tensor (const tensor_expression_type<derived_type> &expr)
            : super_type(expr)
        {}

        template<class derived_type>
        constexpr fixed_rank_tensor& operator= (const tensor_expression_type<derived_type> &expr)
        {
            fixed_rank_tensor temp( expr );
            swap(*this,temp);
            return *this;
        }

        template<class derived_type>
        constexpr fixed_rank_tensor (const matrix_expression_type<derived_type> &expr)
            : fixed_rank_tensor( matrix_type(expr) )
        {}

        template<class derived_type>
        constexpr fixed_rank_tensor (const vector_expression_type<derived_type> &expr)
            : fixed_rank_tensor( vector_type(expr) )
        {}

        template<class OtherTensor>
        constexpr fixed_rank_tensor (const basic_tensor<OtherTensor> & other)
            : super_type(other.extents())
        {
            static_assert( is_valid_tensor_v<OtherTensor>,
                "boost::numeric::ublas::fixed_rank_tensor( const OtherTensor& ) : The OtherTensor should have a tensor type"
            );

            static_assert( std::is_same_v<value_type, typename OtherTensor::value_type>,
                "boost::numeric::ublas::fixed_rank_tensor( const OtherTensor& ) : The tensor should have same value type"
            );

            if ( other.extents().size() != R ){
                throw std::runtime_error("boost::numeric::ublas::static_tensor( const OtherTensor& ) : The tensor should have equal number of extents");
            }
        
            std::copy(other.begin(), other.end(), super_type::begin());
        }

        template<class OtherTensor>
        constexpr fixed_rank_tensor (basic_tensor<OtherTensor> && other)
            : super_type(other.extents())
        {
            static_assert( is_valid_tensor_v<OtherTensor>,
                "boost::numeric::ublas::fixed_rank_tensor( OtherTensor&& ) : The OtherTensor should have a tensor type"
            );

            static_assert( std::is_same_v<value_type, typename OtherTensor::value_type>,
                "boost::numeric::ublas::fixed_rank_tensor( OtherTensor&& ) : The tensor should have same value type"
            );

            if ( other.extents().size() != R ){
                throw std::runtime_error("boost::numeric::ublas::static_tensor( OtherTensor&& ) : The tensor should have equal number of extents");
            }
            
            std::move(other.begin(), other.end(), super_type::begin());
        }
        
        inline constexpr fixed_rank_tensor& operator=(const_reference v) noexcept{
            std::fill(super_type::begin(), super_type::end(), v);
            return *this;
        }
    };

    template<typename V, typename E>
    fixed_rank_tensor( E const&, V const& ) -> fixed_rank_tensor<V, E::_size, first_order>;
    
    template<typename E>
    fixed_rank_tensor( E const& ) -> fixed_rank_tensor<float, E::_size, first_order>;

} // boost::numeric::ublas


namespace boost::numeric::ublas{
    
    template<typename T, std::size_t R, typename F>
    struct tensor_traits< fixed_rank_tensor<T, R, F> > {
        using container_type= std::vector< T >;
        using extents_type  = dynamic_extents<R>;
        using layout_type   = F;
        using container_tag = dynamic_tensor_tag;
    };

    template<typename T, std::size_t R, typename F>
    struct is_valid_tensor< fixed_rank_tensor<T, R, F> > : std::true_type{};
    
    template<typename T, std::size_t R, typename F, typename NewValue>
    struct tensor_rebind< fixed_rank_tensor<T, R, F>, NewValue > {
        using type = fixed_rank_tensor< NewValue, R, F >;
    };
    
    template<typename T, std::size_t R, typename F, typename NewValue, typename NewExtents>
    struct tensor_rebind< fixed_rank_tensor<T, R, F>, NewValue, NewExtents > {
        using type = fixed_rank_tensor< NewValue, NewExtents::_size, F >;
    };
    
    template<typename T, std::size_t R, typename F, typename NewValue, typename NewExtents, typename NewLayout>
    struct tensor_rebind< fixed_rank_tensor<T, R, F>, NewValue, NewExtents, NewLayout > {
        using type = fixed_rank_tensor< NewValue, NewExtents::_size, NewLayout >;
    };

    template<typename T, std::size_t R, typename F>
    struct is_static< fixed_rank_tensor<T, R, F> > : std::false_type{};
    
    template<typename T, std::size_t R, typename F>
    struct is_static_rank< fixed_rank_tensor<T, R, F> > : std::true_type{};
    
    template<typename T, std::size_t R, typename F>
    struct is_dynamic< fixed_rank_tensor<T, R, F> > : std::true_type{};
    
    template<typename T, std::size_t R, typename F>
    struct is_dynamic_rank< fixed_rank_tensor<T, R, F> > : std::false_type{};

    template<typename V, typename F, typename T, std::size_t R>
    struct result_tensor< V, basic_fixed_rank_extents<T,R>, F >{
        using type = fixed_rank_tensor< V, R, F >;
    };

} // namespace boost::numeric::ublas::detail



#endif
