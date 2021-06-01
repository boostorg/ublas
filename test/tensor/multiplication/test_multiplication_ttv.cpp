//
// 	Copyright (c) 2021  Cem Bassoy, cem.bassoy@gmail.com
// 	Copyright (c) 2021, Amit Singh, amitsingh19975@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#include <boost/test/unit_test.hpp>
#include "../fixture_utility.hpp"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

BOOST_AUTO_TEST_SUITE(test_multiplication_ttv, 
    *boost::unit_test::description("Validate Tensor Times Vector")
)


BOOST_TEST_DECORATOR(
    *boost::unit_test::label("ttv")
    *boost::unit_test::description("Testing ttv for dynamic tensor")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_extents_dynamic,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::fixture_extents_dynamic<std::size_t>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using layout_type = typename TestTupleType::second_type;
    using fixture_type = ublas::fixture_extents_dynamic<std::size_t>;
    using vector_t  = std::vector<value_type>;
    using inner_t = inner_type_t<value_type>;

    auto const& self = static_cast<fixture_type const&>(*this);

    ublas::for_each_fixture(self, []<typename extents_type>(auto /*id*/, extents_type const& na){
        auto const rank = ublas::size(na);
        auto const p = ublas::product(na);
        using base_t = typename extents_type::base_type;

        BOOST_TEST_CONTEXT("[TTV Dynamic Tensor] testing for rank(" << rank << ")"){
            auto a = vector_t(p, value_type{2});
            auto wa = ublas::to_strides(na,layout_type{});
            for(std::size_t m = 0ul; m < rank; ++m){
                auto b  = vector_t  (na[m], value_type{1} );
                auto nb = extents_type {na[m],1ul};
                auto wb = ublas::to_strides(nb,layout_type{});
                auto const nc_rank = std::max(std::size_t{rank-1u}, std::size_t{2});
                auto nc_base = base_t(nc_rank,1ul);

                for(std::size_t i = 0ul, j = 0ul; i < rank; ++i)
                    if(i != m)
                    nc_base[j++] = na[i];

                auto nc = extents_type (std::move(nc_base));
                auto wc = ublas::to_strides(nc,layout_type{});
                auto c  = vector_t  (ublas::product(nc), value_type{0});

                ublas::ttv(m+1, rank,
                            c.data(), nc.data(), wc.data(),
                            a.data(), na.data(), wa.data(),
                            b.data(), nb.data(), wb.data());

                auto v = value_type{ static_cast<inner_t>(na[m]) };
                BOOST_CHECK(std::equal(c.begin(),c.end(),a.begin(), [v](auto cc, auto aa){return cc == v*aa;}));
            }
        }
    });
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("ttv")
    *boost::unit_test::description("Testing ttv for static rank tensor")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_extents_static_rank,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::fixture_extents_static_rank<std::size_t>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using layout_type = typename TestTupleType::second_type;
    using fixture_type = ublas::fixture_extents_static_rank<std::size_t>;
    using vector_t  = std::vector<value_type>;
    using inner_t = inner_type_t<value_type>;

    auto const& self = static_cast<fixture_type const&>(*this);

    ublas::for_each_fixture(self, []<typename extents_type>(auto /*id*/, extents_type const& na){
        constexpr auto rank = std::tuple_size_v<extents_type>;
        auto const p = ublas::product(na);
        using extents_value_type = typename extents_type::value_type;

        if constexpr(rank > 0ul){
            constexpr auto nc_rank = std::max(std::size_t{rank - 1u}, std::size_t{2});
            using nc_extents_type = ublas::extents_core<extents_value_type, nc_rank>;
            using nb_extents_type = ublas::extents_core<extents_value_type, 2ul>;
            using base_t = typename nc_extents_type::base_type;
            BOOST_TEST_CONTEXT("[TTV Static Rank Tensor] testing for rank(" << rank << ")"){
                auto a = vector_t(p, value_type{2});
                auto wa = ublas::to_strides(na,layout_type{});
                for(std::size_t m = 0ul; m < rank; ++m){
                    auto b  = vector_t  (na[m], value_type{1});
                    auto nb = nb_extents_type {na[m],1ul};
                    auto wb = ublas::to_strides(nb,layout_type{});
                    auto nc_base = base_t();
                    std::fill(nc_base.begin(), nc_base.end(), 1ul);

                    for(std::size_t i = 0ul, j = 0ul; i < rank; ++i)
                        if(i != m) nc_base[j++] = na[i];

                    auto nc = nc_extents_type (std::move(nc_base));
                    auto wc = ublas::to_strides(nc,layout_type{});
                    auto c  = vector_t  (ublas::product(nc), value_type{0});

                    ublas::ttv(m+1, rank,
                                c.data(), nc.data(), wc.data(),
                                a.data(), na.data(), wa.data(),
                                b.data(), nb.data(), wb.data());

                    auto v = value_type{ static_cast<inner_t>(na[m]) };
                    BOOST_CHECK(std::equal(c.begin(),c.end(),a.begin(), [v](auto cc, auto aa){return cc == v*aa;}));
                }
            }
        }

    });
}

template<typename E, std::size_t N, std::size_t M>
constexpr auto generate_result_extents() noexcept{
    namespace ublas = boost::numeric::ublas;
    using value_type = typename E::value_type;
    constexpr auto sz = std::max( std::size_t{N - 1ul}, std::size_t{2} );


    constexpr auto helper = []<std::size_t... Is>(std::index_sequence<Is...>){
        constexpr auto process_arr = []{
            std::array<value_type,sz> temp;
            constexpr auto n = ublas::to_array_v<E>;
            std::fill(std::begin(temp), std::end(temp), value_type{1});
            for(std::size_t i = 0ul, j = 0ul; i < N; ++i){
                if(i != M) temp[j++] = n[i];
            }
            return temp;
        };
        constexpr auto arr = process_arr();
        return ublas::extents_core<value_type, arr[Is]...>{};
    };
    return helper(std::make_index_sequence<sz>{});
}

// FIXME: temp fix to the invalid computation of static strides,
// rempve this after the fix
template<typename L, typename T, std::size_t N>
constexpr auto get_strides(std::array<T,N> const& temp) noexcept{
    namespace ublas = boost::numeric::ublas;
    using extents_type = ublas::extents_core<T,N>;
    auto n = extents_type{temp};
    return ublas::to_strides(n, L{});
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("ttv")
    *boost::unit_test::description("Testing ttv for static tensor")
)
BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_extents_static,
    TestTupleType,
    boost::numeric::ublas::test_types,
    boost::numeric::ublas::fixture_extents_static<std::size_t>
){
    namespace ublas = boost::numeric::ublas;
    using value_type = typename TestTupleType::first_type;
    using layout_type = typename TestTupleType::second_type;
    using fixture_type = ublas::fixture_extents_static<std::size_t>;
    using inner_t = inner_type_t<value_type>;

    auto const& self = static_cast<fixture_type const&>(*this);

    ublas::for_each_fixture(self, []<typename extents_type>(auto /*id*/, extents_type const& /*na*/){
        using extents_value_type = typename extents_type::value_type;
        static constexpr auto rank = ublas::size_v<extents_type>;
        static constexpr auto p = ublas::product_v<extents_type>;

        if constexpr (rank > 0ul){

            BOOST_TEST_CONTEXT("[TTV Static Tensor] testing for rank(" << rank << ")"){
                
                auto na = ublas::to_array_v<extents_type>;
                auto a = std::array<value_type,p>();
                std::fill(std::begin(a), std::end(a), value_type{2});
                // FIXME: use strides_v after the fix
                auto wa = get_strides<layout_type>(na);

                static_for_each<rank>([&a, &wa]<typename IType>(IType /*id*/){
                    
                    constexpr auto na = ublas::to_array_v<extents_type>;
                    constexpr auto m = IType::value;

                    using nb_type = ublas::extents_core<extents_value_type, na[m], 1ul>;
                    auto b = std::array<value_type,ublas::product_v<nb_type>>();
                    std::fill(std::begin(b), std::end(b), value_type{1});
                    constexpr auto nb = ublas::to_array_v<nb_type>;
                    
                    // FIXME: use strides_v after the fix
                    auto wb = get_strides<layout_type>(nb);

                    using nc_type = decltype(generate_result_extents<extents_type, rank, m>());
                    constexpr auto nc = ublas::to_array_v<nc_type>;

                    // FIXME: use strides_v after the fix
                    auto wc = get_strides<layout_type>(nc);
                    auto c  = std::array<value_type, ublas::product_v<nc_type> >();
                    std::fill(std::begin(c), std::end(c), value_type{0});

                    ublas::ttv(m+1, rank,
                                c.data(), nc.data(), wc.data(),
                                a.data(), na.data(), wa.data(),
                                b.data(), nb.data(), wb.data());

                    auto v = value_type{ static_cast<inner_t>(na[m]) };
                    BOOST_CHECK(std::equal(c.begin(),c.end(),a.begin(), [v](auto cc, auto aa){return cc == v*aa;}));
                });
            }
        }

    });
}

BOOST_AUTO_TEST_SUITE_END()