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


#ifndef BOOST_UBLAS_DYNAMIC_TENSOR_IMPL_HPP
#define BOOST_UBLAS_DYNAMIC_TENSOR_IMPL_HPP

#include <boost/numeric/ublas/tensor/tensor.hpp>

namespace boost::numeric::ublas {

	template< typename T = float, typename F = first_order >
	struct dynamic_tensor:
			public basic_tensor< dynamic_tensor< T, F > >
	{
		using self_type                 = dynamic_tensor< T, F >;
		using super_type				= basic_tensor< self_type >;

		template<class derived_type>
		using tensor_expression_type 	= typename super_type::template tensor_expression_type<derived_type>;

		template<class derived_type>
		using matrix_expression_type 	= typename super_type::template matrix_expression_type<derived_type>;

		template<class derived_type>
		using vector_expression_type 	= typename super_type::template vector_expression_type<derived_type>;

		using array_type  				= std::vector<T>;
		using layout_type 				= F;

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

		using extents_type              = dynamic_extents<>;
		using strides_type              = strides_t<extents_type,layout_type>;

		using matrix_type               = typename super_type::matrix_type;
		using vector_type               = typename super_type::vector_type;

		constexpr dynamic_tensor() = default;

		constexpr dynamic_tensor( array_type const& a )
			: super_type(extents_type{}, a)
		{}

		constexpr dynamic_tensor( extents_type const& e, value_type const& i )
			: super_type(e)
		{
			resize(e);
			std::fill(super_type::begin(), super_type::end(), i);
		}

		constexpr dynamic_tensor( extents_type const& e )
			: super_type(e)
		{
			resize(e);
		}

		constexpr dynamic_tensor( matrix_type const& v )
			: super_type( extents_type(v.size1(), v.size2()) )
		{
			auto const sz = v.size1() * v.size2();
			if(sz){
				resize(sz);
				std::copy(v.data().begin(), v.data().end(),super_type::data_.begin());
			}
		}

		constexpr dynamic_tensor( matrix_type && v )
			: super_type( extents_type(v.size1(), v.size2()) )
		{
			auto const sz = v.size1() * v.size2();
			if(sz){
				resize(sz);
				for(auto i = size_type{}; i < sz; ++i){
					super_type::data_[i] = std::move(v.data()[i]);
				}
			}
		}

		constexpr dynamic_tensor (const vector_type &v)
			: super_type( extents_type(v.size(), typename extents_type::value_type{1}) )
		{
			auto const sz = v.size();
			if(sz){
				resize(sz);
				std::copy(v.data().begin(), v.data().end(),super_type::data_.begin());
			}
			
		}

		constexpr dynamic_tensor (vector_type &&v)
			: super_type( extents_type(v.size(), typename extents_type::value_type{1}) )
		{
			auto const sz = v.size();
			if(sz){
				resize(sz);
				for(auto i = size_type{}; i < sz; ++i){
					super_type::data_[i] = std::move(v.data()[i]);
				}
			}
			
		}

		template<class OtherLayout>
		constexpr dynamic_tensor (const dynamic_tensor<value_type, OtherLayout> &expr)
			: super_type(static_cast<super_type const&>(expr))
		{}

		template<class derived_type>
		constexpr dynamic_tensor (const tensor_expression_type<derived_type> &expr)
			: dynamic_tensor(expr, detail::retrieve_extents(expr) )
		{}

		template<class derived_type>
		constexpr dynamic_tensor (const tensor_expression_type<derived_type> &expr, extents_type const& e)
			: super_type(expr,e)
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

	private:

		inline void resize( size_type sz ){
			super_type::data_.resize(sz);;
		}

		inline void resize( extents_type const& e ){
			super_type::data_.resize( product(e) );
		}

	};

	template<typename V>
	dynamic_tensor(dynamic_extents<> const&, V const&) -> dynamic_tensor<V,first_order>;
	
	dynamic_tensor(dynamic_extents<> const&) -> dynamic_tensor<float,first_order>;

} // boost::numeric::ublas


namespace boost::numeric::ublas::detail{
	
	template<typename T, typename F>
	struct tensor_traits< dynamic_tensor<T,F> > {
		using array_type 	= std::vector< T >;
		using extents_type 	= dynamic_extents<>;
		using layout_type 	= F;
		using strides_type	= strides_t<extents_type,layout_type>;
	};

	template<typename T, typename F>
	struct is_tensor< dynamic_tensor<T,F> > : std::true_type{};
	
    template<typename T, typename F, typename NewValue>
    struct tensor_rebind< dynamic_tensor<T, F>, NewValue > {
        using type = static_tensor< NewValue, F >;
    };
	
    template<typename T, typename F, typename NewValue, typename NewLayout>
    struct tensor_rebind< dynamic_tensor<T, F>, NewValue, NewLayout > {
        using type = static_tensor< NewValue, NewLayout >;
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
