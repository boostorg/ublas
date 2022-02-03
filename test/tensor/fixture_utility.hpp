//
//  Copyright (c) 2021, Cem Bassoy, cem.bassoy@gmail.com
//  Copyright (c) 2021, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#ifndef BOOST_UBLAS_TEST_TENSOR_FIXTURE_UTILITY_HPP
#define BOOST_UBLAS_TEST_TENSOR_FIXTURE_UTILITY_HPP

#include <boost/numeric/ublas/tensor.hpp>
#include <type_traits>
#include "utility.hpp"
#include <cstddef>
#include <random>

namespace boost::numeric::ublas{

    template<integral ExtentsType>
    struct fixture_extents_dynamic{

        using extents_type = extents_core<ExtentsType>;
        using base_type = std::vector<extents_type>;
        
        extents_core<ExtentsType> const n{};

        extents_core<ExtentsType> const n1{1};
        extents_core<ExtentsType> const n2{2};

        extents_core<ExtentsType> const n11{1, 1};
        extents_core<ExtentsType> const n12{1, 2};
        extents_core<ExtentsType> const n21{2, 1};
        extents_core<ExtentsType> const n22{2, 2};
        extents_core<ExtentsType> const n32{3, 2};

        extents_core<ExtentsType> const n111{1, 1, 1};
        extents_core<ExtentsType> const n112{1, 1, 2};
        extents_core<ExtentsType> const n121{1, 2, 1};
        extents_core<ExtentsType> const n123{1, 2, 3};
        extents_core<ExtentsType> const n211{2, 1, 1};
        extents_core<ExtentsType> const n213{2, 1, 3};
        extents_core<ExtentsType> const n321{3, 2, 1};
        extents_core<ExtentsType> const n432{4, 3, 2};

        extents_core<ExtentsType> const n1111{1, 1, 1, 1};
        extents_core<ExtentsType> const n4231{4, 2, 3, 1};

        base_type const collection{ 
            n,
            n1,     n2,     
            n11,    n12,    n21,    n22,    n32,
            n111,   n112,   n121,   n123,   n211,   n213,   n321,   n432, 
            n1111,  n4231
        };

        template<typename T, typename L = layout::last_order, typename... Args>
        static auto make_tensor(extents_type e, Args&&... args){
            return tensor_type<T,L>(std::move(e), std::forward<Args>(args)...);
        }
    };

    template<integral ExtentsType>
    struct fixture_extents_static_rank{
        constexpr static extents_core<ExtentsType,0> n{};

        constexpr static extents_core<ExtentsType,1> n1{1};
        constexpr static extents_core<ExtentsType,1> n2{2};

        constexpr static extents_core<ExtentsType,2> n11{1, 1};
        constexpr static extents_core<ExtentsType,2> n12{1, 2};
        constexpr static extents_core<ExtentsType,2> n21{2, 1};
        constexpr static extents_core<ExtentsType,2> n22{2, 2};
        constexpr static extents_core<ExtentsType,2> n32{3, 2};

        constexpr static extents_core<ExtentsType,3> n111{1, 1, 1};
        constexpr static extents_core<ExtentsType,3> n112{1, 1, 2};
        constexpr static extents_core<ExtentsType,3> n121{1, 2, 1};
        constexpr static extents_core<ExtentsType,3> n123{1, 2, 3};
        constexpr static extents_core<ExtentsType,3> n211{2, 1, 1};
        constexpr static extents_core<ExtentsType,3> n213{2, 1, 3};
        constexpr static extents_core<ExtentsType,3> n321{3, 2, 1};
        constexpr static extents_core<ExtentsType,3> n432{4, 3, 2};

        constexpr static extents_core<ExtentsType,4> n1111{1, 1, 1, 1};
        constexpr static extents_core<ExtentsType,4> n4231{4, 2, 3, 1};
        
        constexpr static std::tuple collection{
            n,
            n1,     n2,     
            n11,    n12,    n21,    n22,    n32,
            n111,   n112,   n121,   n123,   n211,   n213,   n321,   n432, 
            n1111,  n4231
        };

        using tuple_type = decltype(collection);
        
        constexpr static auto size = std::tuple_size_v<tuple_type>;
    };

    template<integral ExtentsType>
    struct fixture_extents_static{
        using n_type        = extents_core<ExtentsType>;

        using n1_type       = extents_core<ExtentsType,1>;
        using n2_type       = extents_core<ExtentsType,2>;

        using n11_type      = extents_core<ExtentsType,1, 1>;
        using n12_type      = extents_core<ExtentsType,1, 2>;
        using n21_type      = extents_core<ExtentsType,2, 1>;
        using n22_type      = extents_core<ExtentsType,2, 2>;
        using n32_type      = extents_core<ExtentsType,3, 2>;

        using n111_type     = extents_core<ExtentsType,1, 1, 1>;
        using n112_type     = extents_core<ExtentsType,1, 1, 2>;
        using n121_type     = extents_core<ExtentsType,1, 2, 1>;
        using n123_type     = extents_core<ExtentsType,1, 2, 3>;
        using n211_type     = extents_core<ExtentsType,2, 1, 1>;
        using n213_type     = extents_core<ExtentsType,2, 1, 3>;
        using n321_type     = extents_core<ExtentsType,3, 2, 1>;
        using n432_type     = extents_core<ExtentsType,4, 3, 2>;

        using n1111_type    = extents_core<ExtentsType,1, 1, 1, 1>;
        using n4231_type    = extents_core<ExtentsType,4, 2, 3, 1>;

        constexpr static n1_type    n1{};
        constexpr static n2_type    n2{};

        constexpr static n11_type   n11{};
        constexpr static n12_type   n12{};
        constexpr static n21_type   n21{};
        constexpr static n22_type   n22{};
        constexpr static n32_type   n32{};

        constexpr static n111_type  n111{};
        constexpr static n112_type  n112{};
        constexpr static n121_type  n121{};
        constexpr static n123_type  n123{};
        constexpr static n211_type  n211{};
        constexpr static n213_type  n213{};
        constexpr static n321_type  n321{};
        constexpr static n432_type  n432{};

        constexpr static n1111_type n1111{};
        constexpr static n4231_type n4231{};


        using tuple_type = std::tuple<
            n_type,
            n1_type,     n2_type,     
            n11_type,    n12_type,    n21_type,    n22_type,    n32_type,
            n111_type,   n112_type,   n121_type,   n123_type,   n211_type,   n213_type,   n321_type,   n432_type, 
            n1111_type,  n4231_type
        >;

        tuple_type const collection{};
        
        constexpr static auto size = std::tuple_size_v<tuple_type>;

    };
    
    template<typename T, typename L = layout::last_order>
    struct fixture_tensor_dynamic{
    private:
        fixture_extents_dynamic<std::size_t> const m_fixture{};
    public:
        using tensor_type = tensor_dynamic<T,L>;
        using base_type = std::vector<tensor_type>;

        tensor_type const t2 { m_fixture.n2 };

        tensor_type const t11 { m_fixture.n11 };
        tensor_type const t12 { m_fixture.n12 };
        tensor_type const t21 { m_fixture.n21 };
        tensor_type const t22 { m_fixture.n22 };
        tensor_type const t32 { m_fixture.n32 };

        tensor_type const t111 { m_fixture.n111 };
        tensor_type const t112 { m_fixture.n112 };
        tensor_type const t121 { m_fixture.n121 };
        tensor_type const t123 { m_fixture.n123 };
        tensor_type const t211 { m_fixture.n211 };
        tensor_type const t213 { m_fixture.n213 };
        tensor_type const t321 { m_fixture.n321 };
        tensor_type const t432 { m_fixture.n432 };

        tensor_type const t1111 { m_fixture.n1111 };
        tensor_type const t4231 { m_fixture.n4231 };

        base_type const collection{ 
            t2,     
            t11,    t12,    t21,    t22,    t32,
            t111,   t112,   t121,   t123,   t211,   t213,   t321,   t432, 
            t1111,  t4231
        };
    };

    template<typename TupleType>
    using tuple_fixture_tensor_dynamic = fixture_tensor_dynamic<typename TupleType::first_type, typename TupleType::second_type>;

    template<typename T, typename L = layout::last_order>
    struct fixture_tensor_static_rank{
    private:
        constexpr static fixture_extents_static_rank<std::size_t> m_fixture{};
    public:
        template<std::size_t N>
        using tensor_type = tensor_static_rank<T,N,L>;

        tensor_type<1> const t2 { m_fixture.n2 };

        tensor_type<2> const t11 { m_fixture.n11 };
        tensor_type<2> const t12 { m_fixture.n12 };
        tensor_type<2> const t21 { m_fixture.n21 };
        tensor_type<2> const t22 { m_fixture.n22 };
        tensor_type<2> const t32 { m_fixture.n32 };

        tensor_type<3> const t111 { m_fixture.n111 };
        tensor_type<3> const t112 { m_fixture.n112 };
        tensor_type<3> const t121 { m_fixture.n121 };
        tensor_type<3> const t123 { m_fixture.n123 };
        tensor_type<3> const t211 { m_fixture.n211 };
        tensor_type<3> const t213 { m_fixture.n213 };
        tensor_type<3> const t321 { m_fixture.n321 };
        tensor_type<3> const t432 { m_fixture.n432 };

        tensor_type<4> const t1111 { m_fixture.n1111 };
        tensor_type<4> const t4231 { m_fixture.n4231 };
        
        using tuple_type = std::tuple<
            tensor_type<1> const&,
            tensor_type<2> const&, tensor_type<2> const&, tensor_type<2> const&, tensor_type<2> const&, tensor_type<2> const&,
            tensor_type<3> const&, tensor_type<3> const&, tensor_type<3> const&, tensor_type<3> const&, tensor_type<3> const&, 
            tensor_type<3> const&, tensor_type<3> const&, tensor_type<3> const&,
            tensor_type<4> const&, tensor_type<4> const&
        >;
        
        tuple_type collection{
            t2,     
            t11,    t12,    t21,    t22,    t32,
            t111,   t112,   t121,   t123,   t211,   t213,   t321,   t432, 
            t1111,  t4231
        };

        constexpr static auto size = std::tuple_size_v<tuple_type>;
    };

    template<typename TupleType>
    using tuple_fixture_tensor_static_rank = fixture_tensor_static_rank<typename TupleType::first_type, typename TupleType::second_type>;

    template<typename T, typename L = layout::last_order>
    struct fixture_tensor_static{

        template<typename E>
        using tensor_type = tensor_static<T,E,L>;
        using fixture_type = fixture_extents_static<std::size_t>;

        using t2_type       = tensor_type<typename fixture_type::n2_type>;

        using t11_type      = tensor_type<typename fixture_type::n11_type>;
        using t12_type      = tensor_type<typename fixture_type::n12_type>;
        using t21_type      = tensor_type<typename fixture_type::n21_type>;
        using t22_type      = tensor_type<typename fixture_type::n22_type>;
        using t32_type      = tensor_type<typename fixture_type::n32_type>;

        using t111_type     = tensor_type<typename fixture_type::n111_type>;
        using t112_type     = tensor_type<typename fixture_type::n112_type>;
        using t121_type     = tensor_type<typename fixture_type::n121_type>;
        using t123_type     = tensor_type<typename fixture_type::n123_type>;
        using t211_type     = tensor_type<typename fixture_type::n211_type>;
        using t213_type     = tensor_type<typename fixture_type::n213_type>;
        using t321_type     = tensor_type<typename fixture_type::n321_type>;
        using t432_type     = tensor_type<typename fixture_type::n432_type>;

        using t1111_type    = tensor_type<typename fixture_type::n1111_type>;
        using t4231_type    = tensor_type<typename fixture_type::n4231_type>;

        // t2_type const    t2{};

        t11_type const   t11{};
        t12_type const   t12{};
        t21_type const   t21{};
        t22_type const   t22{};
        t32_type const   t32{};

        t111_type const  t111{};
        t112_type const  t112{};
        t121_type const  t121{};
        t123_type const  t123{};
        t211_type const  t211{};
        t213_type const  t213{};
        t321_type const  t321{};
        t432_type const  t432{};

        t1111_type const t1111{};
        t4231_type const t4231{};


        using tuple_type = std::tuple<
            // t2_type const&,     
            t11_type const&,    t12_type const&,    t21_type const&,    t22_type const&,    t32_type const&,
            t111_type const&,   t112_type const&,   t121_type const&,   t123_type const&,   t211_type const&,   t213_type const&,   t321_type const&,   t432_type const&, 
            t1111_type const&,  t4231_type const&
        >;

        tuple_type collection{
            // t2,     
            t11,    t12,    t21,    t22,    t32,
            t111,   t112,   t121,   t123,   t211,   t213,   t321,   t432, 
            t1111,  t4231
        };
        
        constexpr static auto size = std::tuple_size_v<tuple_type>;
    };

    template<typename TupleType>
    using tuple_fixture_tensor_static = fixture_tensor_static<typename TupleType::first_type, typename TupleType::second_type>;

    namespace detail{
        template<typename T>
        struct is_tuple : std::false_type{};
        
        template<typename... Ts>
        struct is_tuple< std::tuple<Ts...> > : std::true_type{};

        template<typename T>
        inline static constexpr auto is_tuple_v = is_tuple< std::decay_t<T> >::value;

        template<typename T>
        concept HasTuple = requires{
            typename std::decay_t<T>::tuple_type;
            requires is_tuple_v< typename std::decay_t<T>::tuple_type >;
        };
        
    }

    template<detail::HasTuple FixtureType, typename FnType>
    constexpr auto for_each_fixture(FixtureType&& fixture, FnType&& fn){
        for_each_in_tuple(fixture.collection, std::forward<FnType>(fn));
    }

    template<integral ExtentsType, typename FnType>
    constexpr auto for_each_fixture(fixture_extents_dynamic<ExtentsType> const& fixture, FnType&& fn){
        std::size_t i{};
        for(auto const& el : fixture.collection){
            std::invoke(fn, i, el);
            ++i;
        }
    }

    template<typename V, typename L, typename FnType>
    constexpr auto for_each_fixture(fixture_tensor_dynamic<V,L> const& fixture, FnType&& fn){
        std::size_t i{};
        for(auto const& el : fixture.collection){
            std::invoke(fn, i, el);
            ++i;
        }
    }

} // namespace boost::numeric::ublas

#endif // BOOST_UBLAS_TEST_TENSOR_FIXTURE_UTILITY_HPP