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

BOOST_AUTO_TEST_SUITE(test_tensor_ttt, 
    *boost::unit_test::description("Validate Tensor Times Tensor")
)

constexpr auto calculate_nc_base(auto& out, auto const& na, auto const& nb, std::size_t r, std::size_t s) noexcept{
    using namespace boost::numeric::ublas;
    using namespace std;
    auto it = std::copy(begin(na), begin(na) + r, out.begin());
    std::copy(begin(nb), begin(nb) + s, it);
};

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("ttt")
    *boost::unit_test::description("Testing ttt for dynamic tensor")
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
        using extents_base_t = typename extents_type::base_type;

        if(rank <= 1ul) return;

        constexpr auto one = std::size_t{1};
        constexpr auto two = std::size_t{2};

        BOOST_TEST_CONTEXT("[TTT Dynamic Tensor] testing for rank(" << rank << ")"){

            auto wa = ublas::to_strides(na,layout_type{});
            auto a  = vector_t(p, value_type{2});
            auto pa = rank;

            auto const& nb = na;
            auto wb = ublas::to_strides(nb,layout_type{});
            auto b  = vector_t(ublas::product(nb), value_type{3});
            auto pb = ublas::size(nb);

            // the number of contractions is changed.
            for(auto q = std::size_t{0}; q <= pa; ++q) {

                auto r  = pa - q;
                auto s  = pb - q;

                auto pc = r+s > 0 ? std::max(std::size_t{r+s},two) : two;

                auto nc_base = extents_base_t(pc, 1ul);
                calculate_nc_base(nc_base, na, nb, r, s);

                auto nc = extents_type ( std::move(nc_base) );
                auto wc = ublas::to_strides(nc,layout_type{});
                auto c  = vector_t  ( ublas::product(nc), value_type{0} );

                ublas::ttt(pa,pb,q,
                            c.data(), nc.data(), wc.data(),
                            a.data(), na.data(), wa.data(),
                            b.data(), nb.data(), wb.data());


                auto acc = one;
                for(auto i = r; i < pa; ++i)
                    acc *= na[i];

                auto v = value_type{ static_cast<inner_t>(acc) }*a[0]*b[0];

                BOOST_CHECK( std::all_of(c.begin(),c.end(), [v](auto cc){return cc == v; } ) );

            }
        }
    });
}

BOOST_TEST_DECORATOR(
    *boost::unit_test::label("ttt")
    *boost::unit_test::description("Testing ttt for static rank tensor")
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
        static constexpr auto rank = ublas::size_v<extents_type>;
        auto const p = ublas::product(na);
        using extents_value_type = typename extents_type::value_type;

        if constexpr(rank > 1ul){

            static constexpr auto one = std::size_t{1};
            static constexpr auto two = std::size_t{2};
            
            auto wa = ublas::to_strides(na,layout_type{});
            auto a  = vector_t(p, value_type{2});
            static constexpr auto pa = rank;

            auto const& nb = na;
            auto wb = ublas::to_strides(nb,layout_type{});
            auto b  = vector_t(ublas::product(nb), value_type{3});
            static constexpr auto pb = ublas::size_v<decltype(nb)>;

            BOOST_TEST_CONTEXT("[TTT Static Rank Tensor] testing for rank(" << rank << ")"){

                static_for_each<pa>([&wa, &a, &na, &nb, &wb, &b]<typename IType>(IType /*id*/){
                    constexpr auto q = IType::value;
                    constexpr auto r  = pa - q;
                    constexpr auto s  = pb - q;

                    constexpr auto pc = r+s > 0 ? std::max(std::size_t{r+s},two) : two;

                    using nc_type = ublas::extents_core<extents_value_type, pc>;
                    auto nc_base = typename nc_type::base_type{};
                    std::fill(nc_base.begin(), nc_base.end(), 1ul);
                    calculate_nc_base(nc_base, na, nb, r, s);

                    auto nc = extents_type ( nc_base );
                    auto wc = ublas::to_strides(nc,layout_type{});
                    auto c  = vector_t  ( ublas::product(nc), value_type{0} );

                    ublas::ttt(pa,pb,q,
                                c.data(), nc.data(), wc.data(),
                                a.data(), na.data(), wa.data(),
                                b.data(), nb.data(), wb.data());


                    auto acc = one;
                    for(auto i = r; i < pa; ++i)
                        acc *= na[i];

                    auto v = value_type{ static_cast<inner_t>(acc) }*a[0]*b[0];

                    BOOST_CHECK( std::all_of(c.begin(),c.end(), [v](auto cc){return cc == v; } ) );
                });
            }
        }

    });
}

template<typename E1, typename E2, std::size_t r, std::size_t s>
constexpr auto compute_nc() noexcept{
    namespace ublas = boost::numeric::ublas;
    using value_type = typename E1::value_type;
    constexpr auto pc = r+s > std::size_t{0} ? std::max(std::size_t{r+s}, std::size_t{2}) : std::size_t{2};
    
    constexpr auto helper = []<std::size_t... Is>(std::index_sequence<Is...>){
        constexpr auto helper1 = []{
            std::array<value_type,pc> temp;
            constexpr auto na = ublas::to_array_v<E1>;
            constexpr auto nb = ublas::to_array_v<E2>;
            std::fill(std::begin(temp), std::end(temp), value_type{1});
            calculate_nc_base(temp, na, nb, r, s);
            return temp;
        };
        constexpr auto arr = helper1();
        return ublas::extents_core<value_type, arr[Is]...>{};
    };
    return helper(std::make_index_sequence<pc>{});
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
    *boost::unit_test::label("ttt")
    *boost::unit_test::description("Testing ttt for static tensor")
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

    ublas::for_each_fixture(self, []<typename extents_type>(auto /*id*/, extents_type const& na){
        static constexpr auto rank = ublas::size_v<extents_type>;
        static constexpr auto p = ublas::product_v<extents_type>;

        if constexpr(rank > 1ul){

            static constexpr auto one = std::size_t{1};
            
            // FIXME: remove after the fix
            auto wa = get_strides<layout_type>(ublas::to_array_v<extents_type>);
            auto a  = std::array<value_type, p>();
            std::fill(a.begin(), a.end(), value_type{1});
            static constexpr auto pa = rank;

            auto const& nb = na;
            using nb_type = decltype(nb);
            
            // FIXME: remove after the fix
            auto wb = get_strides<layout_type>(ublas::to_array_v<nb_type>);
            auto b  = std::array<value_type, ublas::product_v<nb_type>>();
            std::fill(b.begin(), b.end(), value_type{3});

            static constexpr auto pb = ublas::size_v<nb_type>;

            BOOST_TEST_CONTEXT("[TTT Static Rank Tensor] testing for rank(" << rank << ")"){

                static_for_each<pa>([&wa, &a, &nb, &wb, &b, &na]<typename IType>(IType /*id*/){
                    constexpr auto q = IType::value;
                    constexpr auto r  = pa - q;
                    constexpr auto s  = pb - q;

                    auto nc = compute_nc<extents_type, nb_type, r, s>();
                    using nc_type = decltype(nc);
                    
                    // FIXME: remove after the fix
                    auto wc = get_strides<layout_type>(ublas::to_array_v<nc_type>);
                    auto c  = std::array<value_type, ublas::product_v<nc_type>>();
                    std::fill(c.begin(), c.end(), value_type{0});

                    ublas::ttt(pa,pb,q,
                                c.data(), nc.data(), wc.data(),
                                a.data(), na.data(), wa.data(),
                                b.data(), nb.data(), wb.data());


                    auto acc = one;
                    for(auto i = r; i < pa; ++i)
                        acc *= na[i];

                    auto v = value_type{ static_cast<inner_t>(acc) }*a[0]*b[0];

                    BOOST_CHECK( std::all_of(c.begin(),c.end(), [v](auto cc){return cc == v; } ) );
                });
            }
        }

    });
}

BOOST_AUTO_TEST_SUITE_END()