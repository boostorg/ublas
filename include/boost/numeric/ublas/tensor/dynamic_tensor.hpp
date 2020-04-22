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


#ifndef BOOST_UBLAS_DYNAMIC_TENSOR_IMPL_HPP
#define BOOST_UBLAS_DYNAMIC_TENSOR_IMPL_HPP

#include <boost/numeric/ublas/tensor/tensor.hpp>

namespace boost::numeric::ublas {

    template< typename T = float, typename F = first_order >
    struct dynamic_tensor:
            public basic_tensor< dynamic_tensor< T, F > >
    {
        using self_type                 = dynamic_tensor< T, F >;
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

        constexpr dynamic_tensor() = default;

        dynamic_tensor( std::initializer_list< size_type > li )
            : super_type( std::move(li) )
        {}

        constexpr dynamic_tensor( extents_type const& e, value_type const& i )
            : super_type(e,i)
        {}

        constexpr dynamic_tensor( extents_type const& e, array_type const& a )
            : super_type(e,a)
        {}

        constexpr dynamic_tensor( extents_type const& e )
            : super_type(e)
        {}

        constexpr dynamic_tensor( matrix_type const& v )
            : super_type( extents_type{v.size1(), v.size2()} )
        {
            std::copy(v.data().begin(), v.data().end(),super_type::data_.begin());
        }

        constexpr dynamic_tensor( matrix_type && v )
            : super_type( extents_type{v.size1(), v.size2()} )
        {
            std::move(v.data().begin(), v.data().end(),super_type::begin());
        }

        constexpr dynamic_tensor (const vector_type &v)
            : super_type( extents_type{ v.size(), typename extents_type::value_type{1} } )
        {
            std::copy(v.data().begin(), v.data().end(),super_type::data_.begin());
        }

        constexpr dynamic_tensor (vector_type &&v)
            : super_type( extents_type{ v.size(), typename extents_type::value_type{1} } )
        {
            std::move(v.data().begin(), v.data().end(),super_type::begin());
        }

        template<class OtherLayout>
        constexpr dynamic_tensor (const dynamic_tensor<value_type, OtherLayout> &expr)
            : super_type(expr)
        {

        }

        template<class derived_type>
        constexpr dynamic_tensor (const tensor_expression_type<derived_type> &expr)
            : super_type(expr)
        {}

        template<class derived_type>
        constexpr dynamic_tensor& operator= (const tensor_expression_type<derived_type> &expr)
        {
            dynamic_tensor temp( expr );
            swap(*this,temp);
            return *this;
        }

        template<class derived_type>
        constexpr dynamic_tensor (const matrix_expression_type<derived_type> &expr)
            : dynamic_tensor( matrix_type(expr) )
        {}

        template<class derived_type>
        constexpr dynamic_tensor (const vector_expression_type<derived_type> &expr)
            : dynamic_tensor( vector_type(expr) )
        {}


        template<class OtherTensor>
        constexpr dynamic_tensor (const basic_tensor<OtherTensor> & other)
            : super_type(other.extents())
        {
            static_assert( is_valid_tensor_v<OtherTensor>,
                "boost::numeric::ublas::dynamic_tensor( const OtherTensor& ) : The OtherTensor should have a tensor type"
            );

            static_assert( std::is_same_v<value_type, typename OtherTensor::value_type>,
                "boost::numeric::ublas::dynamic_tensor( const OtherTensor& ) : The tensor should have same value type"
            );

            std::copy(other.begin(), other.end(), super_type::begin());
        }

        template<class OtherTensor>
        constexpr dynamic_tensor (basic_tensor<OtherTensor> && other)
            : super_type(other.extents())
        {
            static_assert( is_valid_tensor_v<OtherTensor>,
                "boost::numeric::ublas::dynamic_tensor( OtherTensor&& ) : The OtherTensor should have a tensor type"
            );

            static_assert( std::is_same_v<value_type, typename OtherTensor::value_type>,
                "boost::numeric::ublas::dynamic_tensor( OtherTensor&& ) : The tensor should have same value type"
            );

            std::move(other.begin(), other.end(), super_type::begin());

        }

        inline constexpr dynamic_tensor& operator=(const_reference v) noexcept{
            std::fill(super_type::begin(), super_type::end(), v);
            return *this;
        }

    };

    template<typename V>
    dynamic_tensor(dynamic_extents<> const&, V const&) -> dynamic_tensor<V,first_order>;

    dynamic_tensor(dynamic_extents<> const&) -> dynamic_tensor<float,first_order>;

    template<typename T>
    dynamic_tensor(basic_tensor<T> const&) -> dynamic_tensor< typename T::value_type, typename T::layout_type >;

    template<typename T>
    dynamic_tensor(basic_tensor<T> &&) -> dynamic_tensor< typename T::value_type, typename T::layout_type >;

} // boost::numeric::ublas


namespace boost::numeric::ublas{

    template<typename T, typename F>
    struct tensor_traits< dynamic_tensor<T,F> > {
        using container_type= std::vector< T >;
        using extents_type  = dynamic_extents<>;
        using layout_type   = F;
        using container_tag = dynamic_tensor_tag;
    };

    template<typename T, typename F>
    struct is_valid_tensor< dynamic_tensor<T,F> > : std::true_type{};

    template<typename T, typename F, typename NewValue>
    struct tensor_rebind< dynamic_tensor<T, F>, NewValue > {
        using type = dynamic_tensor< NewValue, F >;
    };

    template<typename T, typename F, typename NewValue, typename NewLayout>
    struct tensor_rebind< dynamic_tensor<T, F>, NewValue, NewLayout > {
        using type = dynamic_tensor< NewValue, NewLayout >;
    };

    template<typename T, typename F>
    struct is_static< dynamic_tensor<T, F> > : std::false_type{};

    template<typename T, typename F>
    struct is_static_rank< dynamic_tensor<T, F> > : std::false_type{};

    template<typename T, typename F>
    struct is_dynamic< dynamic_tensor<T, F> > : std::true_type{};

    template<typename T, typename F>
    struct is_dynamic_rank< dynamic_tensor<T, F> > : std::true_type{};

    template<typename V, typename F>
    struct result_tensor< V, dynamic_extents<>, F >{
        using type = dynamic_tensor< V, F >;
    };

} // namespace boost::numeric::ublas::detail



#endif
