//
// 	Copyright (c) 2020, Cem Bassoy, cem.bassoy@gmail.com
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Google and Fraunhofer IOSB, Ettlingen, Germany
//

#include <boost/test/unit_test.hpp>
#include <boost/mp11.hpp>

#include <boost/numeric/ublas/tensor/extents.hpp>
#include <boost/numeric/ublas/tensor/access.hpp>

#include <numeric>
#include <algorithm>


BOOST_AUTO_TEST_SUITE ( test_access_suite )


using layout_types = std::tuple<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;

//zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::first_order, boost::numeric::ublas::last_order>;

struct fixture
{

  using extents_t = boost::numeric::ublas::extents<>;
  using value_t = typename extents_t::value_type;
  using multi_index_t = std::vector<value_t>;
  using index_t = value_t;

  fixture()
  {
    static_assert(shapes.size() == multi_index.size(),"");
    static_assert(shapes.size() == indexf.size(),"");
    static_assert(shapes.size() == indexl.size(),"");
    static_assert(shapes.size() == ranks.size(),"");

    for(auto k = 0u; k < multi_index.size(); ++k){
      auto const& n = shapes[k];
      auto const r = ranks[k];
      assert( n.size() == r );
      for (auto const& i : multi_index[k]){
        assert( std::equal(i.begin(), i.end(), boost::numeric::ublas::begin(n), std::less<>{}) ) ;
      }
    }
  }


  static inline auto shapes = std::array<extents_t,15>
    {{
      {       },
      {1,1    },

      {1,2    },
      {1,4    },
      {2,1    },
      {4,1    },
      {2,3    },

      {2,3,1  },
      {1,2,3  },
      {3,1,2  },
      {3,2,4  },

      {2,3,4,1},
      {1,2,3,4},
      {3,1,2,4},
      {3,2,4,5}
    }};

  static constexpr inline auto ranks = std::array<unsigned,shapes.size()>
    { 0,2,2,2,2,2,2,3,3,3,3,4,4,4,4 };

  static inline auto multi_index = std::array<std::array<multi_index_t,3>,shapes.size()>
    {{
      {{ {       }, {       }, {       } }}, // 0 {}
      {{ {0,0    }, {0,0    }, {0,0    } }}, // 1 {1,1}

      {{ {0,0    }, {0,1    }, {0,1    } }}, // 2 {1,2}
      {{ {0,0    }, {0,2    }, {0,3    } }}, // 3 {1,4}
      {{ {0,0    }, {1,0    }, {1,0    } }}, // 4 {2,1}
      {{ {0,0    }, {2,0    }, {3,0    } }}, // 5 {4,1}
      {{ {0,0    }, {1,1    }, {1,2    } }}, // 6 {2,3}

      {{ {0,0,0  }, {1,1,0  }, {1,2,0  } }}, // 7 {2,3,1}
      {{ {0,0,0  }, {0,1,1  }, {0,1,2  } }}, // 8 {1,2,3}
      {{ {0,0,0  }, {1,0,1  }, {2,0,1  } }}, // 9 {3,1,2}
      {{ {0,0,0  }, {1,1,2  }, {2,1,3  } }}, //10 {3,2,4}

      {{ {0,0,0,0}, {1,1,2,0}, {1,2,3,0} }}, //11 {2,3,4,1}
      {{ {0,0,0,0}, {0,1,1,2}, {0,1,2,3} }}, //12 {1,2,3,4}
      {{ {0,0,0,0}, {1,0,1,2}, {2,0,1,3} }}, //13 {3,1,2,4}
      {{ {0,0,0,0}, {1,1,2,3}, {2,1,3,4} }}  //14 {3,2,4,5}
    }};

  static constexpr inline auto indexf = std::array<std::array<index_t,3>,shapes.size()>
    {{
      {{0, 0,  0}}, //  0 {}
      {{0, 0,  0}}, //  1 {1,1}
      {{0, 1,  1}}, //  2 { {0,0    }, {0,1    }, {0,1    } }, // 2 {1,2}
      {{0, 2,  3}}, //  3 { {0,0    }, {0,2    }, {0,3    } }, // 2 {1,4}
      {{0, 1,  1}}, //  4 { {0,0    }, {1,0    }, {1,0    } }, // 3 {2,1}
      {{0, 2,  3}}, //  5 { {0,0    }, {2,0    }, {3,0    } }, // 3 {4,1}
      {{0, 3,  5}}, //  6 { {0,0    }, {1,1    }, {1,2    } }, // 4 {2,3}
      {{0, 3,  5}}, //  7 { {0,0,0  }, {1,1,0  }, {1,2,0  } }, // 5 {2,3,1}
      {{0, 3,  5}}, //  8 { {0,0,0  }, {0,1,1  }, {0,1,2  } }, // 6 {1,2,3}
      {{0, 4,  5}}, //  9 { {0,0,0  }, {1,0,1  }, {2,0,1  } }, // 7 {3,1,2}
      {{0,16, 23}}, // 10 { {0,0,0  }, {1,1,2  }, {2,1,3  } }, // 8 {3,2,4}, {1,3,6}
      {{0,15, 23}}, // 11 { {0,0,0,0}, {1,1,2,0}, {1,2,3,0} }, // 9 {2,3,4,1}, {1,2,6,6}
      {{0,15, 23}}, // 12 { {0,0,0,0}, {0,1,1,2}, {0,1,2,3} }, //10 {1,2,3,4}, {1,1,2,6}
      {{0,16, 23}}, // 13 { {0,0,0,0}, {1,0,1,2}, {2,0,1,3} }, //11 {3,1,2,4}, {1,3,3,6}
      {{0,88,119}}, // 14 { {0,0,0,0}, {1,1,2,3}, {2,1,3,4} }  //12 {3,2,4,5}, {1,3,6,24}
    }};

  static constexpr inline auto indexl = std::array<std::array<index_t,3>,shapes.size()>
    {{
      {{0, 0,  0}}, //  0 {}
      {{0, 0,  0}}, //  1 {1,1}
      {{0, 1,  1}}, //  2 { {0,0    }, {0,1    }, {0,1    } }, // 2 {1,2}
      {{0, 2,  3}}, //  3 { {0,0    }, {0,2    }, {0,3    } }, // 2 {1,4}
      {{0, 1,  1}}, //  4 { {0,0    }, {1,0    }, {1,0    } }, // 3 {2,1}
      {{0, 2,  3}}, //  5 { {0,0    }, {2,0    }, {3,0    } }, // 3 {4,1}
      {{0, 4,  5}}, //  6 { {0,0    }, {1,1    }, {1,2    } }, // 4 {2,3    }, {3,1}
      {{0, 4,  5}}, //  7 { {0,0,0  }, {1,1,0  }, {1,2,0  } }, // 5 {2,3,1  }, {3,1,1}
      {{0, 4,  5}}, //  8 { {0,0,0  }, {0,1,1  }, {0,1,2  } }, // 6 {1,2,3  }, {6,3,1}
      {{0, 3,  5}}, //  9 { {0,0,0  }, {1,0,1  }, {2,0,1  } }, // 7 {3,1,2  }, {2,2,1}
      {{0,14, 23}}, // 10 { {0,0,0  }, {1,1,2  }, {2,1,3  } }, // 8 {3,2,4  }, {8,4,1}
      {{0,18, 23}}, // 11 { {0,0,0,0}, {1,1,2,0}, {1,2,3,0} }, // 9 {2,3,4,1}, {12, 4,1,1}
      {{0,18, 23}}, // 12 { {0,0,0,0}, {0,1,1,2}, {0,1,2,3} }, //10 {1,2,3,4}, {24,12,4,1}
      {{0,14, 23}}, // 13 { {0,0,0,0}, {1,0,1,2}, {2,0,1,3} }, //11 {3,1,2,4}, { 8, 8,4,1}
      {{0,73,119}}, // 14 { {0,0,0,0}, {1,1,2,3}, {2,1,3,4} }  //12 {3,2,4,5}, {40,20,5,1}
    }};

  template<typename extents_type>
  constexpr inline auto prodn(extents_type const& n)
  {
    return std::accumulate(boost::numeric::ublas::begin(n),boost::numeric::ublas::end(n),1ul, std::multiplies<>{});
  }

  //  static constexpr inline auto const& e = shapes;
  //  static constexpr inline auto const& i = multi_indices;


  //  template<unsigned k, unsigned p, unsigned r> struct x        { static inline constexpr auto value = e[k][r]*x<k,p,r+1>::value; };
  //  template<unsigned k, unsigned p>             struct x<k,p,p> { static inline constexpr auto value =    1;                 };
  //  template<unsigned k, unsigned p>             struct x<k,p,0> { static inline constexpr auto value =    1*x<k,p,1>::value; };

  //  template<unsigned k, unsigned p, unsigned r> struct y        { static inline constexpr auto value = e[k][r  ]*y<k,p,r-1>::value; };
  //  template<unsigned k, unsigned p>             struct y<k,p,p> { static inline constexpr auto value =         1*y<k,p,p-1>::value; };
  //  template<unsigned k, unsigned p>             struct y<k,p,0> { static inline constexpr auto value = e[k][p-1];                 };


  //  template<unsigned k, unsigned p> static inline constexpr auto wf = x<k,p,0>::value;
  //  template<unsigned k, unsigned p> static inline constexpr auto wl = y<k,p,p>::value;

  //  template<unsigned r, unsigned k, unsigned kk> struct zf         { static inline constexpr auto value = i[k][kk][r]*wf<k,r> + zf<r-1,k,kk>::value; };
  //  template<unsigned k, unsigned kk>             struct zf<0,k,kk> { static inline constexpr auto value = i[k][kk][0]*wf<k,0>; };

  //  template<unsigned k, unsigned kk>  static inline constexpr auto c2 = zf<2,k,kk>::value;
  //  template<unsigned k, unsigned kk>  static inline constexpr auto c3 = zf<3,k,kk>::value;
  //  template<unsigned k, unsigned kk>  static inline constexpr auto c4 = zf<4,k,kk>::value;



};



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_compute_single_index, layout_t,  layout_types, fixture )
{
  namespace ub = boost::numeric::ublas;
  namespace mp = boost::mp11;


  constexpr auto is_first_order = std::is_same_v<layout_t,ub::first_order>;
  constexpr auto const& index = is_first_order ? indexf : indexl;

  mp::mp_for_each<mp::mp_iota_c<std::size(index)>>( [&]( auto I ) {
    auto const& n    = std::get<I>(shapes);
    auto const& w    = ub::to_strides(n,layout_t{});
    auto const& i    = std::get<I>(multi_index);
    auto const& jref = std::get<I>(index);
    mp::mp_for_each<mp::mp_iota_c<std::size(i)>>( [&]( auto K ) {
      auto const& ii = std::get<K>(i);
      auto const  j  = ub::detail::compute_single_index(ii.begin(), ii.end() ,w.begin());
      BOOST_CHECK(j < prodn(n));
      BOOST_CHECK_EQUAL(j,jref[K]);
    });
  });
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_compute_single_index_static_rank, layout_t,  layout_types, fixture )
{
  namespace ub = boost::numeric::ublas;
  namespace mp = boost::mp11;

  constexpr auto is_first_order = std::is_same_v<layout_t,ub::first_order>;
  constexpr auto const& index = is_first_order ? indexf : indexl;

  mp::mp_for_each<mp::mp_iota_c<std::size(index)>>( [&]( auto I ) {
    auto const& n    = std::get<I>(shapes);
    auto const& w    = ub::to_strides(n,layout_t{});
    auto const& i    = std::get<I>(multi_index);
    auto const& jref = std::get<I>(index);
    mp::mp_for_each<mp::mp_iota_c<std::size(i)>>( [&]( auto K ) {
      constexpr auto r = std::get<I>(ranks);
      auto const& ii = std::get<K>(i);
      auto const  j  = ub::detail::compute_single_index<r>(ii.begin(), ii.end() , w.begin());
      BOOST_CHECK(j < prodn(n));
      BOOST_CHECK_EQUAL(j,jref[K]);
    });
  });
}


BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_compute_multi_index, layout_t,  layout_types, fixture )
{
  namespace ub = boost::numeric::ublas;
  namespace mp = boost::mp11;

  constexpr auto is_first_order = std::is_same_v<layout_t,ub::first_order>;
  constexpr auto const& index = is_first_order ? indexf : indexl;

  for(auto k = 0u; k < index.size(); ++k){
    auto const& n = shapes[k];
    auto const& w    = ub::to_strides(n,layout_t{});
    auto const& iref = multi_index[k];
    auto const& jref = index[k];
    for(auto kk = 0u; kk < iref.size(); ++kk){
      auto const  jj = jref[kk];
      auto const& ii = iref[kk];
      auto i = multi_index_t(w.size());
      ub::detail::compute_multi_index(jj, w.begin(), w.end(), i.begin(), layout_t{});
//      std::cout << "j= " << jj << std::endl;
//      std::cout << "i= [ "; for(auto iii : i) std::cout << iii << " "; std::cout << "];" << std::endl;
//      std::cout << "ii_ref = [ "; for(auto iii : ii) std::cout << iii << " "; std::cout << "];" << std::endl;
//      std::cout << "n= [ "; for(auto iii : n) std::cout << iii << " "; std::cout << "];" << std::endl;
//      std::cout << "w= [ "; for(auto iii : w) std::cout << iii << " "; std::cout << "];" << std::endl;
//      std::cout << std::endl;
      BOOST_CHECK ( std::equal(i.begin(),i.end(),ii.begin()) ) ;
    }
  }
}

BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_compute_multi_index_static_rank, layout_t,  layout_types, fixture )
{
    namespace ub = boost::numeric::ublas;
    namespace mp = boost::mp11;

    constexpr auto is_first_order = std::is_same_v<layout_t,ub::first_order>;
    constexpr auto const& index = is_first_order ? indexf : indexl;


    mp::mp_for_each<mp::mp_iota_c<std::size(index)>>( [&]( auto I ) {
        auto const& n    = std::get<I>(shapes);
        auto const& iref = std::get<I>(multi_index);
        auto const& jref = std::get<I>(index);
        auto const& w    = ub::to_strides(n,layout_t{});        
        mp::mp_for_each<mp::mp_iota_c<std::size(iref)>>( [&]( auto K ) {
            auto const  jj = std::get<K>(jref);
            auto const& ii = std::get<K>(iref);
            auto         i = multi_index_t(w.size());
            constexpr auto r = std::get<I>(ranks);
            ub::detail::compute_multi_index<r>(jj, w.begin(), w.end(), i.begin(), layout_t{});
            BOOST_CHECK ( std::equal(i.begin(),i.end(),ii.begin()) ) ;
        });
    });
}




BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_compute_single_index_subtensor, layout_t,  layout_types, fixture )
{
    namespace ub = boost::numeric::ublas;

    // subtensor the whole index-domain of a tensor

    constexpr auto is_first_order = std::is_same_v<layout_t,ub::layout::first_order>;
    constexpr auto const& index = is_first_order ? indexf : indexl;

    // subtensor the whole index-domain of a tensor
    for(auto k = 0u; k < index.size(); ++k){
        auto const& n = shapes[k];
        auto const& w    = ub::to_strides(n,layout_t{});
        auto const& jref = index[k];
        for(auto kk = 0u; kk < jref.size(); ++kk){
            auto const jj = jref[kk];
            auto const j  = ub::detail::compute_single_index(jj,w.begin(),w.end(),w.begin());
            BOOST_CHECK_EQUAL ( j, jj ) ;
        }
    }
}



BOOST_AUTO_TEST_SUITE_END()
