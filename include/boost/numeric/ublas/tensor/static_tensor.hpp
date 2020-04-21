//
// 	Copyright (c) 2018-2020, Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2019-2020, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google
//


#ifndef BOOST_UBLAS_STATIC_TENSOR_IMPL_HPP
#define BOOST_UBLAS_STATIC_TENSOR_IMPL_HPP

#include <boost/numeric/ublas/tensor/tensor.hpp>

namespace boost::numeric::ublas {

    template< typename T, typename E, typename F = first_order >
    struct static_tensor:
            public basic_tensor< static_tensor< T, E, F > >
    {

        static_assert( detail::is_static_v<E>,
                "boost::numeric::static_tensor : extents type should be of static_extents, please check");

        using self_type                 = static_tensor< T, E, F >;
        using super_type				= basic_tensor< self_type >;

        template<class derived_type>
        using tensor_expression_type 	= typename super_type::template tensor_expression_type<derived_type>;

        template<class derived_type>
        using matrix_expression_type 	= typename super_type::template matrix_expression_type<derived_type>;

        template<class derived_type>
        using vector_expression_type 	= typename super_type::template vector_expression_type<derived_type>;

        using array_type  				= typename detail::tensor_traits<self_type>::array_type;
        using layout_type 				= typename detail::tensor_traits<self_type>::layout_type;

        using size_type       			= typename array_type::size_type;
        using difference_type 			= typename array_type::difference_type;
        using value_type      			= typename array_type::value_type;

        using reference       			= typename array_type::reference;
        using const_reference 			= typename array_type::const_reference;

        using pointer         			= typename array_type::pointer;
        using const_pointer   			= typename array_type::const_pointer;

        using iterator        			= typename array_type::iterator;
        using const_iterator  			= typename array_type::const_iterator;

        using reverse_iterator        	= typename array_type::reverse_iterator;
        using const_reverse_iterator  	= typename array_type::const_reverse_iterator;

        using tensor_temporary_type     = self_type;
        using storage_category          = dense_tag;

        using extents_type              = typename detail::tensor_traits<self_type>::extents_type;
        using strides_type              = typename detail::tensor_traits<self_type>::strides_type;

        using matrix_type               = typename super_type::matrix_type;
        using vector_type               = typename super_type::vector_type;

        constexpr static_tensor() = default;

        constexpr static_tensor( array_type const& a ){
            std::copy(a.begin(), a.end(), super_type::begin());
        }

        constexpr static_tensor( value_type const& i ){
            std::fill(super_type::begin(), super_type::end(), i);
        }

        constexpr static_tensor( matrix_type const& v )
        {
            static_assert(extents_type::_size == 2
                , "Error in boost::numeric::ublas::tensor(const matrix_type &v)"
                                        " : invalid extents size");
            
            if( !( super_type::size(0) == v.size1() || super_type::size(1) == v.size2() ) ){
                throw std::runtime_error("Error in boost::numeric::ublas::tensor(const matrix_type &v)"
                                        " : rank of extents are not correct, please check!");
            }
            std::copy(v.data().begin(), v.data().end(),super_type::data_.begin());
        }

        constexpr static_tensor( matrix_type && v )
        {
            static_assert(extents_type::_size == 2
                , "Error in boost::numeric::ublas::tensor(const matrix_type &v)"
                                        " : invalid extents size");
            
            auto const sz = v.size1() * v.size2();
            
            if( !( super_type::size(0) == v.size1() || super_type::size(1) == v.size2() ) ){
                throw std::runtime_error("Error in boost::numeric::ublas::tensor(matrix_type &&v)"
                                        " : rank of extents are not correct, please check!");
            }

            for(auto i = size_type{}; i < sz; ++i){
                super_type::data_[i] = std::move(v.data()[i]);
            }
        }

        constexpr static_tensor (const vector_type &v){
            
            static_assert(extents_type::_size == 2
                , "Error in boost::numeric::ublas::tensor(const vector_type &v)"
                                        " : invalid extents size");
            
            if( !( super_type::size(0) == v.size() || super_type::size(1) == 1 ) ){
                throw std::runtime_error("Error in boost::numeric::ublas::tensor(const vector_type &v)"
                                        " : rank of extents are not correct, please check!");
            }

            std::copy(v.data().begin(), v.data().end(),super_type::data_.begin());
            
        }

        constexpr static_tensor (vector_type &&v){
            
            static_assert(extents_type::_size == 2
                , "Error in boost::numeric::ublas::tensor(const vector_type &&v)"
                                        " : invalid extents size");
            
            auto const sz = v.size();
            
            if( !( super_type::size(0) == v.size() || super_type::size(1) == 1 ) ){
                throw std::runtime_error("Error in boost::numeric::ublas::tensor(vector_type &&v)"
                                        " : rank of extents are not correct, please check!");
            }

            for(auto i = size_type{}; i < sz; ++i){
                super_type::data_[i] = std::move(v.data()[i]);
            }
            
        }

        template<class OtherLayout>
        constexpr static_tensor (const static_tensor<value_type, extents_type, OtherLayout> &expr)
            : super_type(static_cast<super_type const&>(expr))
        {}

        template<class OtherTensor>
        constexpr static_tensor (const basic_tensor<OtherTensor> & other)
        {
            static_assert( detail::is_tensor_v<OtherTensor>,
                "boost::numeric::ublas::static_tensor( const OtherTensor& ) : The OtherTensor should have a tensor type"
            );

            static_assert( std::is_same_v<value_type, typename OtherTensor::value_type>,
                "boost::numeric::ublas::static_tensor( const OtherTensor& ) : The tensor should have same value type"
            );

            if ( other.extents() == super_type::extents() ){
                throw std::runtime_error("boost::numeric::ublas::static_tensor( const OtherTensor& ) : The tensor should have equal extents");
            }

            std::copy(other.begin(), other.end(), super_type::begin());
        }

        template<class OtherTensor>
        constexpr static_tensor (basic_tensor<OtherTensor> && other)
        {
            static_assert( detail::is_tensor_v<OtherTensor>,
                "boost::numeric::ublas::static_tensor( OtherTensor&& ) : The OtherTensor should have a tensor type"
            );

            static_assert( std::is_same_v<value_type, typename OtherTensor::value_type>,
                "boost::numeric::ublas::static_tensor( OtherTensor&& ) : The tensor should have same value type"
            );

            if ( other.extents() == super_type::extents() ){
                throw std::runtime_error("boost::numeric::ublas::static_tensor( OtherTensor&& ) : The tensor should have equal extents");
            }

            for( auto i = size_type(0); i < other.size(); ++i ){
                super_type::data_[i] = std::move(other[i]);
            }
        }

        template<class derived_type>
        constexpr static_tensor (const tensor_expression_type<derived_type> &expr)
            : super_type(expr)
        {}

        template<class derived_type>
        constexpr static_tensor& operator= (const tensor_expression_type<derived_type> &expr)
        {
            static_tensor temp( expr );
            swap(*this,temp);
            return *this;
        }

        template<class derived_type>
        constexpr static_tensor (const matrix_expression_type<derived_type> &expr)
            : static_tensor( matrix_type(expr) )
        {}

        template<class derived_type>
        constexpr static_tensor (const vector_expression_type<derived_type> &expr)
            : static_tensor( vector_type(expr) )
        {}

        inline constexpr static_tensor& operator=(const_reference v) noexcept{
            std::fill(super_type::begin(), super_type::end(), v);
            return *this;
        }

        void reshape (extents_type const& e, value_type v) = delete;
    };

} // boost::numeric::ublas


namespace boost::numeric::ublas::detail{
    
    template<typename T, typename E, typename F>
    struct tensor_traits< static_tensor<T,E,F> > {
        using array_type 	= std::array< T, static_product_v<E> >;
        using extents_type 	= E;
        using layout_type 	= F;
        using strides_type	= strides_t<extents_type,layout_type>;
        using container_tag	= static_tag;
    };

    template<typename T, typename E, typename F>
    struct is_tensor< static_tensor<T,E,F> > : std::true_type{};

    template<typename T, typename E, typename F, typename NewValue>
    struct tensor_rebind< static_tensor<T, E, F>, NewValue > {
        using type = static_tensor< NewValue, E, F >;
    };

    template<typename T, typename E, typename F, typename NewValue, typename NewExtents>
    struct tensor_rebind< static_tensor<T, E, F>, NewValue, NewExtents > {
        using type = static_tensor< NewValue, NewExtents, F >;
    };

    template<typename T, typename E, typename F, typename NewValue, typename NewExtents, typename NewLayout>
    struct tensor_rebind< static_tensor<T, E, F>, NewValue, NewExtents, NewLayout > {
        using type = static_tensor< NewValue, NewExtents, NewLayout >;
    };


    template<typename T, typename E, typename F>
    struct is_static< static_tensor<T, E, F> > : std::true_type{};
    
    template<typename T, typename E, typename F>
    struct is_static_rank< static_tensor<T, E, F> > : std::true_type{};
    
    template<typename T, typename E, typename F>
    struct is_dynamic< static_tensor<T, E, F> > : std::false_type{};
    
    template<typename T, typename E, typename F>
    struct is_dynamic_rank< static_tensor<T, E, F> > : std::false_type{};

    template<typename V, typename F, typename T, T... Es>
    struct result_tensor< V, basic_static_extents<T,Es...>, F >{
        using type = static_tensor< V, basic_static_extents<T,Es...>, F >;
    };

} // namespace boost::numeric::ublas::detail



#endif
