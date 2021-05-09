//
//  Copyright (c) 2018, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2019, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//


#ifndef BOOST_UBLAS_TENSOR_FUNCTIONS_HPP
#define BOOST_UBLAS_TENSOR_FUNCTIONS_HPP

#include "multiplication.hpp"
#include "tensor_engine.hpp"
#include "detail/strides_functions.hpp"

#include "traits/type_traits_extents.hpp"
#include "traits/type_traits_strides.hpp"

#include "fixed_rank_strides.hpp"
//#include "fixed_rank_extents.hpp"

namespace boost::numeric::ublas{
    
    template<typename T> class tensor_core;

} // namespace boost::numeric::ublas


namespace boost::numeric::ublas
{
    namespace detail{

        template<typename T>
        struct is_complex : std::false_type{};

        template<typename T>
        struct is_complex< std::complex<T> > : std::true_type{};

        template<typename T>
        inline static constexpr bool is_complex_v = is_complex<T>::value;

        /// To check if the type is the std::array or not.
        /// Can be extented by providing specialization.
        /// Point to Remember: C-Style arrays are not supported.
        template<typename T>
        struct is_bounded_array : std::false_type{};

        template<typename T>
        inline static constexpr bool is_bounded_array_v = is_bounded_array<T>::value;

        template<typename T, std::size_t N>
        struct is_bounded_array<std::array<T,N>> : std::true_type{};
        
        /// Gives the extent of rank one std::array.
        /// Similar to is_bounded_array, it can also be
        /// extented using specialization.
        /// Custom Type should have similar APIs to 
        /// std::array.
        /// Point to Remember: C-Style arrays are not supported.
        template<typename T>
        struct extent_of_rank_one_array;
        
        template<typename T, std::size_t N>
        struct extent_of_rank_one_array<std::array<T,N>> 
            : std::integral_constant<std::size_t,N>
        {};

        template<typename T>
        inline static constexpr bool extent_of_rank_one_array_v = extent_of_rank_one_array<T>::value;

    } // namespace detail
    
    /** @brief Computes the m-mode tensor-times-vector product
     *
     * Implements C[i1,...,im-1,im+1,...,ip] = A[i1,i2,...,ip] * b[im]
     *
     * @note calls ublas::ttv
     *
     * @param[in] m contraction dimension with 1 <= m <= p
     * @param[in] a tensor object A with order p
     * @param[in] b vector object B
     *
     * @returns tensor object C with order p-1, the same storage format and allocator type as A
    */
    template <typename TensorEngine, typename A >
    inline decltype(auto) prod( tensor_core< TensorEngine > const &a, 
        vector<typename tensor_core< TensorEngine >::value_type, A> const &b, 
        const std::size_t m)
    {

        using tensor_type   = tensor_core< TensorEngine >;
        using extents_type  = typename tensor_type::extents_type;
        using value_type    = typename tensor_type::value_type;
        using array_type    = typename tensor_type::array_type;
        using layout_type   = typename tensor_type::layout_type;

        auto const p = a.rank();

        static_assert(  
            std::is_same_v< 
                typename tensor_core<TensorEngine>::resizable_tag, 
                storage_resizable_container_tag 
            >,
            "error in boost::numeric::ublas::prod(tensor_core const&, vector const& ): "
            "tensor container should be resizable"
        );

        static_assert(  
            is_dynamic_v<extents_type>,
            "error in boost::numeric::ublas::prod(tensor_core const&, vector const& ): "
            "extents type should be dynamic"
        );

        if (m == 0ul)
            throw std::length_error(
                "error in boost::numeric::ublas::prod(ttv): "
                "contraction mode must be greater than zero.");

        if (p < m)
            throw std::length_error(
                "error in boost::numeric::ublas::prod(ttv): rank of tensor must be "
                "greater than or equal to the modus.");

        if (a.empty())
            throw std::length_error(
                "error in boost::numeric::ublas::prod(ttv): first "
                "argument tensor should not be empty.");

        if (b.empty())
            throw std::length_error(
                "error in boost::numeric::ublas::prod(ttv): second "
                "argument vector should not be empty.");

        using extents_value_type = typename extents_type::value_type;

        auto const& na = a.extents();

        auto compute_nc = [](auto const& na){
            using size_type = typename extents_type::size_type;
            if constexpr( is_static_rank_v<extents_type> ){
                constexpr size_type sz = std::max( size(extents_type{}) -1u , size_type(2) );
                using new_extents_type = ublas::extents<sz>;
                auto ret = typename new_extents_type::base_type{};
                std::fill(ret.begin(), ret.end(),size_type(1));
                return new_extents_type( ret );
            }else{
                using extents_base_type = typename extents_type::base_type;
                auto const sz = std::max( ublas::size(na) - 1, size_type(2) );
                auto arr = extents_base_type(sz,1);
                return extents_type{ std::move(arr) } ;
            }
        };

        auto nc = compute_nc(na);
        auto nb = std::vector<extents_value_type>{b.size(), extents_value_type(1)};

        for (auto i = 0ul, j = 0ul; i < p; ++i)
            if (i != m - 1)
                nc[j++] = na.at(i);

        using c_extents_type = std::decay_t< decltype(nc) >;

        using t_engine = tensor_engine< 
            c_extents_type,  
            layout_type,
            strides<c_extents_type>,
            array_type
        >;
        
        auto c = tensor_core<t_engine>( nc, value_type{} );
        auto bb = &(b(0));
        ttv(m, p,
            c.data(), data(c.extents()), c.strides().data(),
            a.data(), data(a.extents()), a.strides().data(),
            bb, data(nb), data(nb));
        return c;
    }

    /** @brief Computes the m-mode tensor-times-matrix product
     *
     * Implements C[i1,...,im-1,j,im+1,...,ip] = A[i1,i2,...,ip] * B[j,im]
     *
     * @note calls ublas::ttm
     *
     * @param[in] a tensor object A with order p
     * @param[in] b vector object B
     * @param[in] m contraction dimension with 1 <= m <= p
     *
     * @returns tensor object C with order p, the same storage format and allocator type as A
    */
    template <typename TensorEngine, typename A>
    inline decltype(auto) prod( tensor_core< TensorEngine > const &a, 
        matrix<typename tensor_core< TensorEngine >::value_type, typename tensor_core< TensorEngine >::layout_type , A> const &b, 
        const std::size_t m)
    {

        using tensor_type   = tensor_core< TensorEngine >;
        using extents_type  = typename tensor_type::extents_type;        
        using value_type    = typename tensor_type::value_type;
        using layout_type   = typename tensor_type::layout_type;


        static_assert(  
            std::is_same_v< 
                typename tensor_core<TensorEngine>::resizable_tag, 
                storage_resizable_container_tag 
            >,
            "error in boost::numeric::ublas::prod(tensor_core const&, matrix const& ): "
            "tensor container should be resizable"
        );

        static_assert(  
            is_dynamic_v<extents_type>,
            "error in boost::numeric::ublas::prod(tensor_core const&, matrix const& ): "
            "extents type should be dynamic"
        );

        auto const p = a.rank();

        if (m == 0ul)
            throw std::length_error(
                "error in boost::numeric::ublas::prod(ttm): "
                "contraction mode must be greater than zero.");

        if (p < m || m > ublas::size(a.extents()))
            throw std::length_error(
                "error in boost::numeric::ublas::prod(ttm): rank "
                "of the tensor must be greater equal the modus.");

        if (a.empty())
            throw std::length_error(
                "error in boost::numeric::ublas::prod(ttm): first "
                "argument tensor should not be empty.");

        if (b.size1() * b.size2() == 0ul)
            throw std::length_error(
                "error in boost::numeric::ublas::prod(ttm): second "
                "argument matrix should not be empty.");

        auto nc = a.extents();
        auto nb = extents<2>{b.size1(), b.size2()};
        auto wb = basic_fixed_rank_strides<std::size_t,2,layout_type>(nb);//strides_type(nb);
//        compute_strides(nb,wb);

        nc[m-1] = nb[0];

        auto c = tensor_type(nc, value_type{});

        auto bb = &(b(0, 0));
        ttm(m, p,
            c.data(), data(c.extents()), c.strides().data(),
            a.data(), data(a.extents()), a.strides().data(),
            bb, data(nb), wb.data());

        return c;
    }

    /** @brief Computes the q-mode tensor-times-tensor product
     *
     * Implements C[i1,...,ir,j1,...,js] = sum( A[i1,...,ir+q] * B[j1,...,js+q]  )
     *
     * @note calls ublas::ttt
     *
     * na[phia[x]] = nb[phib[x]] for 1 <= x <= q
     *
     * @param[in]  a  left-hand side tensor with order r+q
     * @param[in]  b  right-hand side tensor with order s+q
     * @param[in]  phia one-based permutation tuple of length q for the first
     * input tensor a can be of type std::vector<std::size_t> or std::array<std::size_t,N>
     * @param[in]  phib one-based permutation tuple of length q for the second
     * input tensor b can be of type std::vector<std::size_t> or std::array<std::size_t,N>
     * @result     tensor with order r+s
    */
    template <typename TensorEngine1, typename TensorEngine2, typename PermuType = std::vector<size_t>,
        std::enable_if_t<
            !(  is_static_v<typename tensor_core< TensorEngine1 >::extents_type> ||
                is_static_v<typename tensor_core< TensorEngine2 >::extents_type> )
        ,int> = 0
    >
    inline decltype(auto) prod(tensor_core< TensorEngine1 > const &a, tensor_core< TensorEngine2 > const &b,
                                PermuType const &phia, PermuType const &phib)
    {
        using tensor_type       = tensor_core< TensorEngine1 >;
        using extents_type      = typename tensor_type::extents_type;       
        using value_type        = typename tensor_type::value_type;
        using layout_type       = typename tensor_type::layout_type;
        using size_type         = typename extents_type::size_type;
        using array_type        = typename tensor_type::array_type;

        static_assert( 
            std::is_same_v< 
                typename tensor_core<TensorEngine1>::resizable_tag, 
                typename tensor_core<TensorEngine2>::resizable_tag 
            > && 
            std::is_same_v< 
                typename tensor_core<TensorEngine1>::resizable_tag, 
                storage_resizable_container_tag 
            >,
            "error in boost::numeric::ublas::prod(tensor_core const&, tensor_core const&, "
            "PermuType const&, PermuType const& ): "
            "Both the tensor storage should have the same type of storage and both should be resizable"
        );

        static_assert(
            std::is_same_v<value_type, typename tensor_core< TensorEngine2 >::value_type>,
            "error in boost::numeric::ublas::prod(tensor_core< TensorEngine1 > const&, tensor_core< TensorEngine2 > const&, "
            "PermuType const&, PermuType const&): "
            "Both the tensor should have the same value_type"
        );

        auto const pa = a.rank();
        auto const pb = b.rank();

        auto const q = static_cast<size_type>(phia.size());

        if (pa == 0ul)
            throw std::runtime_error("error in ublas::prod: order of left-hand side tensor must be greater than 0.");
        if (pb == 0ul)
            throw std::runtime_error("error in ublas::prod: order of right-hand side tensor must be greater than 0.");
        if (pa < q)
            throw std::runtime_error("error in ublas::prod: number of contraction dimensions cannot be greater than the order of the left-hand side tensor.");
        if (pb < q)
            throw std::runtime_error("error in ublas::prod: number of contraction dimensions cannot be greater than the order of the right-hand side tensor.");

        if (q != phib.size())
            throw std::runtime_error("error in ublas::prod: permutation tuples must have the same length.");

        if (pa < phia.size())
            throw std::runtime_error("error in ublas::prod: permutation tuple for the left-hand side tensor cannot be greater than the corresponding order.");
        if (pb < phib.size())
            throw std::runtime_error("error in ublas::prod: permutation tuple for the right-hand side tensor cannot be greater than the corresponding order.");

        auto const &na = a.extents();
        auto const &nb = b.extents();

        for (auto i = 0ul; i < q; ++i)
            if (na.at(phia.at(i) - 1) != nb.at(phib.at(i) - 1))
                throw std::runtime_error("error in ublas::prod: permutations of the extents are not correct.");

        std::size_t const r = pa - q;
        std::size_t const s = pb - q;

        std::vector<std::size_t> phia1(pa);
        std::vector<std::size_t> phib1(pb);
        std::iota(phia1.begin(), phia1.end(), 1ul);
        std::iota(phib1.begin(), phib1.end(), 1ul);

        auto create_nc = [](auto const& na, auto const& nb, auto const& phia, auto const& phib){
            using lextents_type     = std::decay_t< decltype(na) >;
            using rextents_type     = std::decay_t< decltype(nb) >;
            using array_type        = std::decay_t< decltype(phia) >;
            using extents_base_type = typename extents<>::base_type;
            if constexpr( 
                detail::is_bounded_array_v<array_type> && 
                is_static_rank_v<lextents_type> &&
                is_static_rank_v<rextents_type>
            ){
                constexpr auto const N   = detail::extent_of_rank_one_array_v<array_type>;
                constexpr auto const sz  = size(lextents_type{}) + size(rextents_type{}) - 2 * N;
                constexpr auto const msz = std::max(size_type(sz), size_type(2));
                return extents<msz>();
            }else{
              size_type const size = ( ublas::size(na) + ublas::size(nb) ) - ( phia.size() + phib.size() );
                return extents<>( extents_base_type ( std::max(size, size_type(2)), size_type(1) ) );
            }
        };

        auto nc = create_nc(na,nb,phia,phib);

        for (auto i = 0ul; i < phia.size(); ++i)
            *std::remove(phia1.begin(), phia1.end(), phia.at(i)) = phia.at(i);

        //phia1.erase( std::remove(phia1.begin(), phia1.end(), phia.at(i)),  phia1.end() )  ;

        for (auto i = 0ul; i < r; ++i)
            nc[i] = na[phia1[i] - 1];

        for (auto i = 0ul; i < phib.size(); ++i)
            *std::remove(phib1.begin(), phib1.end(), phib.at(i)) = phib.at(i);
        //phib1.erase( std::remove(phib1.begin(), phib1.end(), phia.at(i)), phib1.end() )  ;

        for (auto i = 0ul; i < s; ++i)
            nc[r + i] = nb[phib1[i] - 1];

        // std::copy( phib.begin(), phib.end(), phib1.end()  );
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
        assert(phia1.size() == pa);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay)
        assert(phib1.size() == pb);

        using c_extents_type = std::decay_t< decltype(nc) >;

        using t_engine = tensor_engine< 
            c_extents_type,
            layout_type,
            strides<c_extents_type>,
            array_type
        >;

        auto c = tensor_core<t_engine>( nc, value_type{} );
        
        ttt(pa, pb, q,
            phia1.data(), phib1.data(),
            c.data(), data(c.extents()), c.strides().data(),
            a.data(), data(a.extents()), a.strides().data(),
            b.data(), data(b.extents()), b.strides().data());

        return c;
    }

    // template<class V, class F, class A1, class A2, std::size_t N, std::size_t M>
    // auto operator*( tensor_index<V,F,A1,N> const& lhs, tensor_index<V,F,A2,M>
    // const& rhs)

    /** @brief Computes the q-mode tensor-times-tensor product
     *
     * Implements C[i1,...,ir,j1,...,js] = sum( A[i1,...,ir+q] * B[j1,...,js+q]  )
     *
     * @note calls ublas::ttt
     *
     * na[phi[x]] = nb[phi[x]] for 1 <= x <= q
     *
     * @param[in]  a  left-hand side tensor with order r+q
     * @param[in]  b  right-hand side tensor with order s+q
     * @param[in]  phi one-based permutation tuple of length q for both input
     * tensors can be of type std::vector<std::size_t> or std::array<std::size_t,N>
     * @result     tensor with order r+s
    */
    template <typename TensorEngine1, typename TensorEngine2, typename PermuType = std::vector<size_t>>
    inline decltype(auto) prod(
      tensor_core< TensorEngine1 > const &a,
      tensor_core< TensorEngine2 > const &b,
      PermuType const &phi)
    {
        return prod(a, b, phi, phi);
    }

    /** @brief Computes the inner product of two tensors     *
     * Implements c = sum(A[i1,i2,...,ip] * B[i1,i2,...,jp])
     *
     * @note calls inner function
     *
     * @param[in] a tensor object A
     * @param[in] b tensor object B
     *
     * @returns a value type.
    */
    template <typename TensorEngine1, typename TensorEngine2>
    inline decltype(auto) inner_prod(tensor_core< TensorEngine1 > const &a, tensor_core< TensorEngine2 > const &b)
    {
        using value_type = typename tensor_core< TensorEngine1 >::value_type;
        
        static_assert(
            std::is_same_v<value_type, typename tensor_core< TensorEngine2 >::value_type>,
            "error in boost::numeric::ublas::inner_prod(tensor_core< TensorEngine1 > const&, tensor_core< TensorEngine2 > const&): "
            "Both the tensor should have the same value_type"
        );

        if (a.rank() != b.rank())
            throw std::length_error("error in boost::numeric::ublas::inner_prod: Rank of both the tensors must be the same.");

        if (a.empty() || b.empty())            
            throw std::length_error("error in boost::numeric::ublas::inner_prod: Tensors should not be empty.");

        if (a.extents() != b.extents())
            throw std::length_error("error in boost::numeric::ublas::inner_prod: Tensor extents should be the same.");
        
        return inner(a.rank(), data(a.extents()),
                    a.data(), a.strides().data(),
                    b.data(), b.strides().data(), value_type{0});
    }

    /** @brief Computes the outer product of two tensors
     *
     * Implements C[i1,...,ip,j1,...,jq] = A[i1,i2,...,ip] * B[j1,j2,...,jq]
     *
     * @note calls outer function
     *
     * @param[in] a tensor object A
     * @param[in] b tensor object B
     *
     * @returns tensor object C with the same storage format F and allocator type A1
    */
    template <typename TensorEngine1, typename TensorEngine2,
        std::enable_if_t<
            !(  is_static_v<typename tensor_core< TensorEngine1 >::extents_type> ||
                is_static_v<typename tensor_core< TensorEngine2 >::extents_type> )
            ,int> = 0
    >
    inline decltype(auto) outer_prod(tensor_core< TensorEngine1 > const &a, tensor_core< TensorEngine2 > const &b)
    {
        using tensor_type   = tensor_core< TensorEngine1 >;
        using value_type    = typename tensor_type::value_type;
        using layout_type   = typename tensor_type::layout_type;
        using array_type   = typename tensor_type::array_type;

        static_assert( 
            std::is_same_v< 
                typename tensor_core<TensorEngine1>::resizable_tag, 
                typename tensor_core<TensorEngine2>::resizable_tag 
            > && 
            std::is_same_v< 
                typename tensor_core<TensorEngine1>::resizable_tag, 
                storage_resizable_container_tag 
            >,
            "error in boost::numeric::ublas::outer_prod(tensor_core const&, tensor_core const&): "
            "Both the tensor storage should have the same type of storage and both should be resizable"
        );

        static_assert(
            std::is_same_v<value_type, typename tensor_core< TensorEngine2 >::value_type>,
            "error in boost::numeric::ublas::outer_prod(tensor_core< TensorEngine1 > const&, tensor_core< TensorEngine2 > const&): "
            "Both the tensor should have the same value_type"
        );

        if (a.empty() || b.empty())
            throw std::runtime_error(
                "error in boost::numeric::ublas::outer_prod: "
                "tensors should not be empty.");

        auto const& na = a.extents();
        auto const& nb = b.extents();

        auto create_nc = [](auto const& na, auto const& nb){
            using na_type = std::decay_t< decltype(na) >;
            using nb_type = std::decay_t< decltype(nb) >;

            if constexpr( is_static_rank_v<na_type> && is_static_rank_v<nb_type> ){
              constexpr auto na_size = size(na_type{});
              constexpr auto nb_size = size(nb_type{});
              using extents_type = extents<na_size+nb_size>;
              auto nc = typename extents_type::base_type{};
              auto nci = std::copy(ublas::begin(na),ublas::end(na), std::begin(nc));
              std::copy(ublas::begin(nb),ublas::end(nb), nci);
              return extents_type(nc);
            }else {
              using extents_type = extents<>;
              auto nc = typename extents_type::base_type(ublas::size(na)+ublas::size(nb));
              auto nci = std::copy(ublas::begin(na),ublas::end(na), std::begin(nc));
              std::copy(ublas::begin(nb),ublas::end(nb), nci);
              return extents_type(nc);
            }
        };

        auto nc  = create_nc(na,nb);

        using c_extents_type = std::decay_t< decltype(nc) >;

        using t_engine = tensor_engine< 
            c_extents_type,  
            layout_type,
            strides<c_extents_type>,
            array_type
        >;

        auto c = tensor_core<t_engine>( nc, value_type{} );

        outer(c.data(), c.rank(), data(c.extents()), c.strides().data(),
            a.data(), a.rank(), data(na), a.strides().data(),
            b.data(), b.rank(), data(nb), b.strides().data());

        return c;
    }

    /** @brief Transposes a tensor according to a permutation tuple
     *
     * Implements C[tau[i1],tau[i2]...,tau[ip]] = A[i1,i2,...,ip]
     *
     * @note calls trans function
     *
     * @param[in] a    tensor object of rank p
     * @param[in] tau  one-based permutation tuple of length p
     * @returns        a transposed tensor object with the same storage format F and allocator type A
    */
    template <typename TensorEngine,typename PermuType = std::vector<std::size_t> >
    inline decltype(auto) trans(tensor_core< TensorEngine > const &a, PermuType const &tau)
    {

        using tensor_type   = tensor_core< TensorEngine >;
        using layout_type   = typename tensor_type::layout_type;
        using array_type    = typename tensor_type::array_type;
        using extents_type  = typename tensor_type::extents_type;
        
        static_assert(
            is_dynamic_v< extents_type > ,
            "error in boost::numeric::ublas::trans(tensor_core< TensorEngine > const &a, "
            "PermuType const &tau): "
            "Tensor should have dynamic extents"
        );

        using t_engine = tensor_engine< 
            extents_type,
            layout_type,
            strides<extents_type>,
            array_type
        >;

        auto const p = a.rank();
        auto const &na = a.extents();
        typename extents_type::base_type nc;

        if constexpr( is_dynamic_rank_v<extents_type> ){
            nc.resize(p);
        }

        for (auto i = 0u; i < p; ++i)
            nc.at(tau.at(i) - 1) = na.at(i);

        auto c = tensor_core<t_engine>( extents_type( std::move(nc) ) );
        
        if (a.empty())
            return c;

        trans(a.rank(), data(a.extents()), tau.data(),
            c.data(), c.strides().data(),
            a.data(), a.strides().data());

        return c;
    }
    /**
     *
     * @brief Computes the frobenius nor of a tensor
     *
     * @note Calls accumulate on the tensor.
     *
     * implements
     * k = sqrt( sum_(i1,...,ip) A(i1,...,ip)^2 )
     *
     * @tparam V the data type of tensor
     * @tparam F the format of tensor storage
     * @tparam A the array_type of tensor
     * @param a the tensor whose norm is expected of rank p.
     * @return the frobenius norm of a tensor.
     */
    template <typename TensorEngine>
    inline decltype(auto) norm(tensor_core< TensorEngine > const &a)
    {
        using tensor_type = tensor_core< TensorEngine >;
        using value_type = typename tensor_type::value_type;
        
        static_assert(std::is_default_constructible<value_type>::value,
                    "Value type of tensor must be default construct able in order "
                    "to call boost::numeric::ublas::norm");

        if (a.empty())
        {
            throw std::runtime_error(
                "error in boost::numeric::ublas::norm: tensors should not be empty.");
        }
        return std::sqrt(accumulate(a.order(), data(a.extents()), a.data(), a.strides().data(), value_type{},
                                    [](auto const &l, auto const &r) { return l + r * r; }));
    }


    /** @brief Computes the complex conjugate component of tensor elements within a tensor expression
     *
     * @param[in] lhs tensor expression
     * @returns   unary tensor expression
    */
    template<typename TensorEngine, class D,
        std::enable_if_t< detail::is_complex_v<typename tensor_core< TensorEngine >::value_type>, int > = 0
    >
    auto conj(detail::tensor_expression< tensor_core<TensorEngine>, D > const& expr)
    {
        return detail::make_unary_tensor_expression< tensor_core<TensorEngine> > (expr(), [] (auto const& l) { return std::conj( l ); } );
    }

    /** @brief Computes the complex conjugate component of tensor elements within a tensor expression
     *
     * @param[in] expr tensor expression
     * @returns   complex tensor
    */
    template<class T, class D>
    auto conj(detail::tensor_expression<T,D> const& expr)
    {
        using old_tensor_type   = T;
        using value_type    = typename old_tensor_type::value_type;
        using layout_type   = typename old_tensor_type::layout_type;
        using array_type    = typename old_tensor_type::array_type;
        using extents_type  = typename old_tensor_type::extents_type;

        using complex_type = std::complex<value_type>;
        using storage_traits_t = storage_traits<array_type>;

        using t_engine = tensor_engine< 
            extents_type,
            layout_type,
            strides<extents_type>,
            typename storage_traits_t::template rebind<complex_type>
        >;

        using tensor_type = tensor_core<t_engine>;

        if( ublas::empty( detail::retrieve_extents( expr  ) ) )
            throw std::runtime_error("error in boost::numeric::ublas::conj: tensors should not be empty.");

        auto a = old_tensor_type( expr );
        auto c = tensor_type( a.extents() );

        std::transform( a.begin(), a.end(),  c.begin(), [](auto const& l){ return std::conj(l) ; }  );

        return c;
    }

    /** @brief Extract the real component of tensor elements within a tensor expression
     *
     * @param[in] lhs tensor expression
     * @returns   unary tensor expression
    */
    template<class T, class D>
    auto real(detail::tensor_expression<T,D> const& expr) {
        return detail::make_unary_tensor_expression<T> (expr(), [] (auto const& l) { return std::real( l ); } );
    }

    /** @brief Extract the real component of tensor elements within a tensor expression
     *
     * @param[in] lhs tensor expression
     * @returns   unary tensor expression
    */
    template<typename TensorEngine, class D,
        std::enable_if_t< detail::is_complex_v<typename tensor_core< TensorEngine >::value_type>, int > = 0
    >
    auto real(detail::tensor_expression< tensor_core< TensorEngine > ,D > const& expr)
    {
        
        using old_tensor_type   = tensor_core< TensorEngine >;
        using complex_type  = typename old_tensor_type::value_type;
        using value_type    = typename complex_type::value_type;
        using layout_type   = typename old_tensor_type::layout_type;
        using array_type    = typename old_tensor_type::array_type;
        using extents_type  = typename old_tensor_type::extents_type;
        using storage_traits_t = storage_traits<array_type>;
        
        using t_engine = tensor_engine< 
            extents_type,
            layout_type,
            strides<extents_type>,
            typename storage_traits_t::template rebind<value_type>
        >;

        using tensor_type = tensor_core<t_engine>;

        if( ublas::empty ( detail::retrieve_extents( expr  ) ) )
            throw std::runtime_error("error in boost::numeric::ublas::real: tensors should not be empty.");

        auto a = old_tensor_type( expr );
        auto c = tensor_type( a.extents() );

        std::transform( a.begin(), a.end(),  c.begin(), [](auto const& l){ return std::real(l) ; }  );

        return c;
    }


    /** @brief Extract the imaginary component of tensor elements within a tensor expression
     *
     * @param[in] lhs tensor expression
     * @returns   unary tensor expression
    */
    template<class T, class D>
    auto imag(detail::tensor_expression<T,D> const& lhs) {
        return detail::make_unary_tensor_expression<T> (lhs(), [] (auto const& l) { return std::imag( l ); } );
    }


    /** @brief Extract the imag component of tensor elements within a tensor expression
     *
     * @param[in] lhs tensor expression
     * @returns   unary tensor expression
    */
    template<typename TensorEngine, class D,
        std::enable_if_t< detail::is_complex_v<typename tensor_core< TensorEngine >::value_type>, int > = 0
    >
    auto imag(detail::tensor_expression< tensor_core< TensorEngine > ,D> const& expr)
    {
        using old_tensor_type   = tensor_core< TensorEngine >;
        using complex_type  = typename old_tensor_type::value_type;
        using value_type    = typename complex_type::value_type;
        using layout_type   = typename old_tensor_type::layout_type;
        using array_type    = typename old_tensor_type::array_type;
        using extents_type  = typename old_tensor_type::extents_type;
        using storage_traits_t = storage_traits<array_type>;
        
        using t_engine = tensor_engine<
            extents_type,
            layout_type,
            strides<extents_type>,
            typename storage_traits_t::template rebind<value_type>
        >;

        using tensor_type = tensor_core<t_engine>;

        if( ublas::empty( detail::retrieve_extents( expr  ) ) )
            throw std::runtime_error("error in boost::numeric::ublas::real: tensors should not be empty.");

        auto a = old_tensor_type( expr );
        auto c = tensor_type( a.extents() );

        std::transform( a.begin(), a.end(),  c.begin(), [](auto const& l){ return std::imag(l) ; }  );

        return c;
    }

} // namespace boost::numeric::ublas

#endif
