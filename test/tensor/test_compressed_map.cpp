//  Copyright (c) 2018-2019 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <boost/numeric/ublas/tensor/storage.hpp>
#include <boost/test/unit_test.hpp>
#include "utility.hpp"
#include <vector>
#include <complex>

BOOST_AUTO_TEST_SUITE(test_compressed_map)

template<typename T>
using comp_map = boost::numeric::ublas::storage::sparse_tensor::compressed_map<T>;
using type_list = list<int,long,float,double,std::complex<float>>;

struct fixture {
  fixture() : sizes {1,2,3,4,5}{}
  
  std::vector<size_t> sizes;

};

BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_compressed_map_ctor, value_type, type_list,fixture) {
  using namespace boost::numeric;
  auto s1 = comp_map<value_type>{};
  BOOST_CHECK(s1.empty());
  BOOST_CHECK_EQUAL(s1.size(), 0);
  BOOST_CHECK_EQUAL(s1.container_size(), 0);

  auto s2 = comp_map<value_type>(0);
  BOOST_CHECK(s2.empty());
  BOOST_CHECK_EQUAL(s2.size(), 0);
  BOOST_CHECK_EQUAL(s2.container_size(), 0);

  for(auto const& s : sizes){
    auto s3 = comp_map<value_type>(s,value_type{0});
    BOOST_CHECK(s3.empty());
    BOOST_CHECK_EQUAL(s3.size(), s);
    BOOST_CHECK_EQUAL(s3.container_size(), size_t{0});
    
    auto s4 = comp_map<value_type>(s,value_type{1});
    BOOST_CHECK(!s4.empty());
    BOOST_CHECK_EQUAL(s4.size(), s);
    BOOST_CHECK_EQUAL(s4.container_size(), s);
  }

  std::vector<value_type> data;
  
  for(auto i = 0; i < 10; i++){
    data.push_back(value_type(i));
  }

  auto s5 = comp_map<value_type>(data);
  for(auto i = 0; i < 10; i++){
    BOOST_CHECK_EQUAL(s5[i], value_type(i));
  }

  auto s6 = comp_map<value_type>(std::move(data));
  for(auto i = 0; i < 10; i++){
    BOOST_CHECK_EQUAL(s6[i], value_type(i));
  }

}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_compressed_map_prune, value_type, type_list,fixture) {
  using namespace boost::numeric;
  std::vector<value_type> data(10,value_type{1});
  auto s = comp_map<value_type>{data};
  
  for(auto i = 0; i < 5; i++){
    s[i] = 0;
  }

  BOOST_CHECK(!s.empty());
  BOOST_CHECK_EQUAL(s.size(), 10);
  BOOST_CHECK_EQUAL(s.container_size(), 10);
  
  s.prune();
  BOOST_CHECK(!s.empty());
  BOOST_CHECK_EQUAL(s.size(), 10);
  BOOST_CHECK_EQUAL(s.container_size(), 5);

}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_compressed_map_contains, value_type, type_list,fixture) {
  using namespace boost::numeric;
  using key_type = typename comp_map<value_type>::key_type;
  
  std::vector<value_type> data;
  
  for(auto i = 0; i < 10; i++){
    data.push_back(value_type(i));
  }

  auto s = comp_map<value_type>{data};

  BOOST_CHECK(!s.contains(key_type{0}));
  for(auto i = 1; i < 10; i++){
    BOOST_CHECK(s.contains(key_type{i}));
  }

  for(auto i = 10; i < 20; i++){
    BOOST_CHECK(!s.contains(key_type{i}));
  }

}

BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_compressed_map_access, value_type, type_list,fixture) {
  using namespace boost::numeric;
  using key_type = typename comp_map<value_type>::key_type;
  
  std::vector<value_type> data;
  
  for(auto i = 0; i < 10; i++){
    data.push_back(value_type(i));
  }

  auto s = comp_map<value_type>{data};

  for(auto i = 0; i < 10; i++){
    BOOST_CHECK_EQUAL(s.at(key_type(i)),value_type(i));
  }

  for(auto i = 0; i < 10; i++){
    BOOST_CHECK_EQUAL(s[key_type(i)],value_type(i));
  }

}

BOOST_AUTO_TEST_SUITE_END()
