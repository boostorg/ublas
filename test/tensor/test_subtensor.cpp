//  Copyright (c) 2018 Cem Bassoy
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
//  The authors gratefully acknowledge the support of
//  Fraunhofer and Google in producing this work
//  which started as a Google Summer of Code project.
//


#include <random>
#include <boost/test/unit_test.hpp>

#include "utility.hpp"

#include <boost/numeric/ublas/tensor/tensor.hpp>
#include <boost/numeric/ublas/tensor/tags.hpp>
#include <boost/numeric/ublas/tensor/span.hpp>



BOOST_AUTO_TEST_SUITE ( subtensor_testsuite )

using test_types = zip<int,float,std::complex<float>>::with_t<boost::numeric::ublas::layout::first_order, boost::numeric::ublas::layout::last_order>;



struct fixture
{
  using extents_type = boost::numeric::ublas::extents<>;

  fixture() : extents {
          extents_type{1,1}, // 1
          extents_type{1,2}, // 2
          extents_type{2,1}, // 3
          extents_type{2,3}, // 4
          extents_type{2,3,1}, // 5
          extents_type{4,1,3}, // 6
          extents_type{1,2,3}, // 7
          extents_type{4,2,3}, // 8
          extents_type{4,2,3,5}} // 9
  {
  }
  std::vector<extents_type> extents;
};


BOOST_FIXTURE_TEST_CASE_TEMPLATE( subtensor_ctor1_test, value,  test_types, fixture )
{

  namespace ublas      = boost::numeric::ublas;
  using value_type     = typename value::first_type;
  using layout_type    = typename value::second_type;
  using tensor_type    = ublas::tensor_dynamic<value_type, layout_type>;
  using subtensor_type = typename tensor_type::subtensor_type;


  auto check = [](auto const& e) {
    auto t = tensor_type(e);
    auto s = subtensor_type(t);
    auto ss = subtensor_type(s);
	  BOOST_CHECK_EQUAL (  s.size() , t.size() );
	  BOOST_CHECK_EQUAL (  ss.size() , t.size() );
    BOOST_CHECK_EQUAL (  s.rank() , t.rank() );
    BOOST_CHECK_EQUAL (  ss.rank() , t.rank() );
    if(ublas::empty(e)) {
      BOOST_CHECK_EQUAL ( s.empty(), t.empty() );
      BOOST_CHECK_EQUAL ( ss.empty(), t.empty() );
      BOOST_CHECK_EQUAL ( s. data(), t. data() );
      BOOST_CHECK_EQUAL ( ss. data(), t. data() );
    }
    else{
      BOOST_CHECK_EQUAL ( !s.empty(), !t.empty() );
      BOOST_CHECK_EQUAL ( !ss.empty(), !t.empty() );
      BOOST_CHECK_EQUAL (  s. data(),  t. data() );
      BOOST_CHECK_EQUAL (  ss. data(),  t. data() );
    }
  };

  for(auto const& e : extents)
    check(e);

}



BOOST_AUTO_TEST_CASE_TEMPLATE( subtensor_ctor2_test, value,  test_types )
{

  namespace ublas = boost::numeric::ublas;
  using value_type  = typename value::first_type;
  using layout_type = typename value::second_type;
  using tensor_type = ublas::tensor_dynamic<value_type, layout_type>;
  using subtensor_type = typename tensor_type::subtensor_type;
  using span  = ublas::span<>;


  {
    auto A    = tensor_type{};
    auto Asub = subtensor_type( A );

    BOOST_CHECK( Asub.strides() == A.strides() );
    BOOST_CHECK( Asub.extents() == A.extents() );
    BOOST_CHECK( Asub.data() == A.data() );

	  auto Asubsub = subtensor_type( Asub );

    BOOST_CHECK( Asubsub.strides() == A.strides() );
    BOOST_CHECK( Asubsub.extents() == A.extents() );
    BOOST_CHECK( Asubsub.data() == A.data() );
  }



  {
    auto A    = tensor_type{1,1};
    auto Asub = subtensor_type( A, span(0), span(0)  );

    BOOST_CHECK( Asub.strides() == A.strides() );
    BOOST_CHECK( Asub.extents() == A.extents() );
    BOOST_CHECK( Asub.data() == A.data() );

    auto Asubsub = subtensor_type( Asub, span(0), span(0));

    BOOST_CHECK( Asubsub.strides() == A.strides() );
    BOOST_CHECK( Asubsub.extents() == A.extents() );
    BOOST_CHECK( Asubsub.data() == A.data() );

  }


  {
    auto A    = tensor_type{1,2};
    auto Asub = subtensor_type( A, span(0), span{}  );

    BOOST_CHECK( Asub.strides() == A.strides() );
    BOOST_CHECK( Asub.extents() == A.extents() );
    BOOST_CHECK( Asub.data() == A.data() );

    auto Asubsub = subtensor_type( Asub, span(0), span());

    BOOST_CHECK( Asubsub.strides() == A.strides() );
    BOOST_CHECK( Asubsub.extents() == A.extents() );
    BOOST_CHECK( Asubsub.data() == A.data() );

  }

  {
    auto A    = tensor_type{1,2};
    auto Asub = subtensor_type( A, span(), span() );

    BOOST_CHECK_EQUAL( Asub.strides().at(0), A.strides().at(0) );
    BOOST_CHECK_EQUAL( Asub.strides().at(1), A.strides().at(1) );

    BOOST_CHECK_EQUAL( Asub.extents().at(0) , 1 );
    BOOST_CHECK_EQUAL( Asub.extents().at(1) , 2 );

    BOOST_CHECK_EQUAL( Asub.data() , A.data()+
                                     Asub.spans().at(0).first()*A.strides().at(0) +
                                     Asub.spans().at(1).first()*A.strides().at(1) );

    auto Asubsub = subtensor_type( Asub, span(0), span(1) );

    BOOST_CHECK_EQUAL( Asubsub.strides().at(0), A.strides().at(0) );
    BOOST_CHECK_EQUAL( Asubsub.strides().at(1), A.strides().at(1) );

    BOOST_CHECK_EQUAL( Asubsub.extents().at(0) , 1 );
    BOOST_CHECK_EQUAL( Asubsub.extents().at(1) , 1 );

    BOOST_CHECK_EQUAL( Asubsub.data() , Asub.data() +
                                      Asubsub.spans().at(0).first()*Asub.strides().at(0) +
                                      Asubsub.spans().at(1).first()*Asub.strides().at(1) );

  }

  {
    auto A    = tensor_type{2,3};
    auto Asub = subtensor_type( A, span(), span(0,2,2) );
    auto B    = tensor_type(Asub.extents());

    BOOST_CHECK_EQUAL( Asub.strides().at(0), A.strides().at(0) );
    BOOST_CHECK_EQUAL( Asub.strides().at(1), A.strides().at(1) * 2 );

    BOOST_CHECK_EQUAL( Asub.extents().at(0) , 2 );
    BOOST_CHECK_EQUAL( Asub.extents().at(1) , 2 );

    BOOST_CHECK_EQUAL( B.strides().at(0), Asub.span_strides().at(0) );
    BOOST_CHECK_EQUAL( B.strides().at(1), Asub.span_strides().at(1) );

    BOOST_CHECK_EQUAL( Asub.data() , A.data()+
                                     Asub.spans().at(0).first()*A.strides().at(0) +
                                     Asub.spans().at(1).first()*A.strides().at(1) );

    auto Asubsub = subtensor_type( Asub, span(1), span() );
    auto C       = tensor_type(Asubsub.extents());

    BOOST_CHECK_EQUAL( Asubsub.strides().at(0), Asub.strides().at(0) );
    BOOST_CHECK_EQUAL( Asubsub.strides().at(1), Asub.strides().at(1) );

    BOOST_CHECK_EQUAL( Asubsub.extents().at(0) , 1 );
    BOOST_CHECK_EQUAL( Asubsub.extents().at(1) , 2 );

    BOOST_CHECK_EQUAL( C.strides().at(0), Asubsub.span_strides().at(0) );
    BOOST_CHECK_EQUAL( C.strides().at(1), Asubsub.span_strides().at(1));

    BOOST_CHECK_EQUAL( Asubsub.data() , Asub.data()+
                                     Asubsub.spans().at(0).first()*Asub.strides().at(0) +
                                     Asubsub.spans().at(1).first()*Asub.strides().at(1) );

  }

  {
    auto A    = tensor_type{4,3};
    auto Asub = subtensor_type( A, span(0,3,3), span(0,2,ublas::max)  );
    auto B    = tensor_type(Asub.extents());

    BOOST_CHECK_EQUAL( Asub.strides().at(0), A.strides().at(0) * 3 );
    BOOST_CHECK_EQUAL( Asub.strides().at(1), A.strides().at(1) * 2 );

    BOOST_CHECK_EQUAL( Asub.extents().at(0) , 2 );
    BOOST_CHECK_EQUAL( Asub.extents().at(1) , 2 );

    BOOST_CHECK_EQUAL( B.strides().at(0), Asub.span_strides().at(0) );
    BOOST_CHECK_EQUAL( B.strides().at(1), Asub.span_strides().at(1) );

    BOOST_CHECK_EQUAL( Asub.data() , A.data()+
                                     Asub.spans().at(0).first()*A.strides().at(0) +
                                     Asub.spans().at(1).first()*A.strides().at(1) );

    auto Asubsub = subtensor_type( Asub, span(1), span(1,ublas::max)  );
    auto C    = tensor_type(Asubsub.extents());

    BOOST_CHECK_EQUAL( Asubsub.strides().at(0), Asub.strides().at(0) );
    BOOST_CHECK_EQUAL( Asubsub.strides().at(1), Asub.strides().at(1) );

    BOOST_CHECK_EQUAL( Asubsub.extents().at(0) , 1 );
    BOOST_CHECK_EQUAL( Asubsub.extents().at(1) , 1 );

    BOOST_CHECK_EQUAL( C.strides().at(0), Asubsub.span_strides().at(0) );
    BOOST_CHECK_EQUAL( C.strides().at(1), Asubsub.span_strides().at(1) );

    BOOST_CHECK_EQUAL( Asubsub.data() , Asub.data()+
                                     Asubsub.spans().at(0).first()*Asub.strides().at(0) +
                                     Asubsub.spans().at(1).first()*Asub.strides().at(1) );
  }

  {
    auto A    = tensor_type{4,3,5};
    auto Asub = subtensor_type( A, span(1,2), span(1,ublas::max), span(2,4)  );

    auto B    = tensor_type(Asub.extents());

    BOOST_CHECK_EQUAL( Asub.strides().at(0), A.strides().at(0) );
    BOOST_CHECK_EQUAL( Asub.strides().at(1), A.strides().at(1) );
    BOOST_CHECK_EQUAL( Asub.strides().at(2), A.strides().at(2) );

    BOOST_CHECK_EQUAL( Asub.extents().at(0) , 2 );
    BOOST_CHECK_EQUAL( Asub.extents().at(1) , 2 );
    BOOST_CHECK_EQUAL( Asub.extents().at(2) , 3 );

    BOOST_CHECK_EQUAL( B.strides().at(0), Asub.span_strides().at(0) );
    BOOST_CHECK_EQUAL( B.strides().at(1), Asub.span_strides().at(1) );
    BOOST_CHECK_EQUAL( B.strides().at(2), Asub.span_strides().at(2) );

    BOOST_CHECK_EQUAL( Asub.data() , A.data()+
                                     Asub.spans().at(0).first()*A.strides().at(0) +
                                     Asub.spans().at(1).first()*A.strides().at(1)+
                                     Asub.spans().at(2).first()*A.strides().at(2));

    auto Asubsub = subtensor_type( Asub, span(1), span(), span(0,2,2)  );

    auto C    = tensor_type(Asubsub.extents());

    BOOST_CHECK_EQUAL( Asubsub.strides().at(0), Asub.strides().at(0) );
    BOOST_CHECK_EQUAL( Asubsub.strides().at(1), Asub.strides().at(1) );
    BOOST_CHECK_EQUAL( Asubsub.strides().at(2), Asub.strides().at(2) * 2);

    BOOST_CHECK_EQUAL( Asubsub.extents().at(0) , 1 );
    BOOST_CHECK_EQUAL( Asubsub.extents().at(1) , 2 );
    BOOST_CHECK_EQUAL( Asubsub.extents().at(2) , 2 );

    BOOST_CHECK_EQUAL( C.strides().at(0), Asubsub.span_strides().at(0) );
    BOOST_CHECK_EQUAL( C.strides().at(1), Asubsub.span_strides().at(1) );
    BOOST_CHECK_EQUAL( C.strides().at(2), Asubsub.span_strides().at(2) );

    BOOST_CHECK_EQUAL( Asubsub.data() , Asub.data()+
                                     Asubsub.spans().at(0).first()*Asub.strides().at(0) +
                                     Asubsub.spans().at(1).first()*Asub.strides().at(1)+
                                     Asubsub.spans().at(2).first()*Asub.strides().at(2));
  }

}

BOOST_AUTO_TEST_CASE_TEMPLATE( subtensor_ctor3_test, value,  test_types )
{

  namespace ublas = boost::numeric::ublas;
  using value_type  = typename value::first_type;
  using layout_type = typename value::second_type;
  using tensor_type = ublas::tensor_dynamic<value_type, layout_type>;
  using subtensor_type = typename tensor_type::subtensor_type;
  using span  = ublas::span<>;


  {
    auto A    = tensor_type{};
    auto Asub = subtensor_type( A );

    BOOST_CHECK( Asub.strides() == A.strides() );
    BOOST_CHECK( Asub.extents() == A.extents() );
    BOOST_CHECK( Asub.data() == A.data() );

  	auto Asubsub = subtensor_type( Asub );

    BOOST_CHECK( Asubsub.strides() == A.strides() );
    BOOST_CHECK( Asubsub.extents() == A.extents() );
    BOOST_CHECK( Asubsub.data() == A.data() );
  }



  {
    auto A    = tensor_type{1,1};
    auto Asub = A( span(0), span(0)  );

    BOOST_CHECK( Asub.strides() == A.strides() );
    BOOST_CHECK( Asub.extents() == A.extents() );
    BOOST_CHECK( Asub.data() == A.data() );

    auto Asubsub = Asub( span(0), span(0));

    BOOST_CHECK( Asubsub.strides() == A.strides() );
    BOOST_CHECK( Asubsub.extents() == A.extents() );
    BOOST_CHECK( Asubsub.data() == A.data() );

  }


  {
    auto A    = tensor_type{1,2};
    auto Asub = A( span(0), span{}  );

    BOOST_CHECK( Asub.strides() == A.strides() );
    BOOST_CHECK( Asub.extents() == A.extents() );
    BOOST_CHECK( Asub.data() == A.data() );

    auto Asubsub =  Asub( span(0), span());

    BOOST_CHECK( Asubsub.strides() == A.strides() );
    BOOST_CHECK( Asubsub.extents() == A.extents() );
    BOOST_CHECK( Asubsub.data() == A.data() );

  }

  {
    auto A    = tensor_type{1,2};
    auto Asub = A(span(), span() );

    BOOST_CHECK_EQUAL( Asub.strides().at(0), A.strides().at(0) );
    BOOST_CHECK_EQUAL( Asub.strides().at(1), A.strides().at(1) );


    BOOST_CHECK_EQUAL( Asub.extents().at(0) , 1 );
    BOOST_CHECK_EQUAL( Asub.extents().at(1) , 2 );

    BOOST_CHECK_EQUAL( Asub.data() , A.data()+
                                     Asub.spans().at(0).first()*A.strides().at(0) +
                                     Asub.spans().at(1).first()*A.strides().at(1) );

    auto Asubsub =  Asub( span(0), span(1) );

    BOOST_CHECK_EQUAL( Asubsub.strides().at(0), A.strides().at(0) );
    BOOST_CHECK_EQUAL( Asubsub.strides().at(1), A.strides().at(1) );

    BOOST_CHECK_EQUAL( Asubsub.extents().at(0) , 1 );
    BOOST_CHECK_EQUAL( Asubsub.extents().at(1) , 1 );

    BOOST_CHECK_EQUAL( Asubsub.data() , Asub.data() +
                                      Asubsub.spans().at(0).first()*Asub.strides().at(0) +
                                      Asubsub.spans().at(1).first()*Asub.strides().at(1) );

  }

  {
    auto A    = tensor_type{2,3};
    auto Asub =  A(span(), span(0,2,2) );
    auto B    = tensor_type(Asub.extents());

    BOOST_CHECK_EQUAL( Asub.strides().at(0), A.strides().at(0) );
    BOOST_CHECK_EQUAL( Asub.strides().at(1), A.strides().at(1) * 2 );

    BOOST_CHECK_EQUAL( Asub.extents().at(0) , 2 );
    BOOST_CHECK_EQUAL( Asub.extents().at(1) , 2 );

    BOOST_CHECK_EQUAL( B.strides().at(0), Asub.span_strides().at(0) );
    BOOST_CHECK_EQUAL( B.strides().at(1), Asub.span_strides().at(1) );

    BOOST_CHECK_EQUAL( Asub.data() , A.data()+
                                     Asub.spans().at(0).first()*A.strides().at(0) +
                                     Asub.spans().at(1).first()*A.strides().at(1) );

    auto Asubsub = Asub( span(1), span() );
    auto C       = tensor_type(Asubsub.extents());

    BOOST_CHECK_EQUAL( Asubsub.strides().at(0), Asub.strides().at(0) );
    BOOST_CHECK_EQUAL( Asubsub.strides().at(1), Asub.strides().at(1) );

    BOOST_CHECK_EQUAL( Asubsub.extents().at(0) , 1 );
    BOOST_CHECK_EQUAL( Asubsub.extents().at(1) , 2 );

    BOOST_CHECK_EQUAL( C.strides().at(0), Asubsub.span_strides().at(0) );
    BOOST_CHECK_EQUAL( C.strides().at(1), Asubsub.span_strides().at(1));

    BOOST_CHECK_EQUAL( Asubsub.data() , Asub.data()+
                                     Asubsub.spans().at(0).first()*Asub.strides().at(0) +
                                     Asubsub.spans().at(1).first()*Asub.strides().at(1) );

  }

  {
    auto A    = tensor_type{4,3};
    auto Asub =  A( span(0,3,3), span(0,2,ublas::max)  );
    auto B    = tensor_type(Asub.extents());

    BOOST_CHECK_EQUAL( Asub.strides().at(0), A.strides().at(0) * 3 );
    BOOST_CHECK_EQUAL( Asub.strides().at(1), A.strides().at(1) * 2 );

    BOOST_CHECK_EQUAL( Asub.extents().at(0) , 2 );
    BOOST_CHECK_EQUAL( Asub.extents().at(1) , 2 );

    BOOST_CHECK_EQUAL( B.strides().at(0), Asub.span_strides().at(0) );
    BOOST_CHECK_EQUAL( B.strides().at(1), Asub.span_strides().at(1) );

    BOOST_CHECK_EQUAL( Asub.data() , A.data()+
                                     Asub.spans().at(0).first()*A.strides().at(0) +
                                     Asub.spans().at(1).first()*A.strides().at(1) );

    auto Asubsub = Asub( span(1), span(1,ublas::max)  );
    auto C    = tensor_type(Asubsub.extents());

    BOOST_CHECK_EQUAL( Asubsub.strides().at(0), Asub.strides().at(0) );
    BOOST_CHECK_EQUAL( Asubsub.strides().at(1), Asub.strides().at(1) );

    BOOST_CHECK_EQUAL( Asubsub.extents().at(0) , 1 );
    BOOST_CHECK_EQUAL( Asubsub.extents().at(1) , 1 );

    BOOST_CHECK_EQUAL( C.strides().at(0), Asubsub.span_strides().at(0) );
    BOOST_CHECK_EQUAL( C.strides().at(1), Asubsub.span_strides().at(1) );

    BOOST_CHECK_EQUAL( Asubsub.data() , Asub.data()+
                                     Asubsub.spans().at(0).first()*Asub.strides().at(0) +
                                     Asubsub.spans().at(1).first()*Asub.strides().at(1) );
  }

  {
    auto A    = tensor_type{4,3,5};
    auto Asub = A( span(1,2), span(1,ublas::max), span(2,4)  );

    auto B    = tensor_type(Asub.extents());

    BOOST_CHECK_EQUAL( Asub.strides().at(0), A.strides().at(0) );
    BOOST_CHECK_EQUAL( Asub.strides().at(1), A.strides().at(1) );
    BOOST_CHECK_EQUAL( Asub.strides().at(2), A.strides().at(2) );

    BOOST_CHECK_EQUAL( Asub.extents().at(0) , 2 );
    BOOST_CHECK_EQUAL( Asub.extents().at(1) , 2 );
    BOOST_CHECK_EQUAL( Asub.extents().at(2) , 3 );

    BOOST_CHECK_EQUAL( B.strides().at(0), Asub.span_strides().at(0) );
    BOOST_CHECK_EQUAL( B.strides().at(1), Asub.span_strides().at(1) );
    BOOST_CHECK_EQUAL( B.strides().at(2), Asub.span_strides().at(2) );

    BOOST_CHECK_EQUAL( Asub.data() , A.data()+
                                     Asub.spans().at(0).first()*A.strides().at(0) +
                                     Asub.spans().at(1).first()*A.strides().at(1)+
                                     Asub.spans().at(2).first()*A.strides().at(2));

    auto Asubsub = Asub( span(1), span(), span(0,2,2)  );

    auto C    = tensor_type(Asubsub.extents());

    BOOST_CHECK_EQUAL( Asubsub.strides().at(0), Asub.strides().at(0) );
    BOOST_CHECK_EQUAL( Asubsub.strides().at(1), Asub.strides().at(1) );
    BOOST_CHECK_EQUAL( Asubsub.strides().at(2), Asub.strides().at(2) * 2);

    BOOST_CHECK_EQUAL( Asubsub.extents().at(0) , 1 );
    BOOST_CHECK_EQUAL( Asubsub.extents().at(1) , 2 );
    BOOST_CHECK_EQUAL( Asubsub.extents().at(2) , 2 );

    BOOST_CHECK_EQUAL( C.strides().at(0), Asubsub.span_strides().at(0) );
    BOOST_CHECK_EQUAL( C.strides().at(1), Asubsub.span_strides().at(1) );
    BOOST_CHECK_EQUAL( C.strides().at(2), Asubsub.span_strides().at(2) );

    BOOST_CHECK_EQUAL( Asubsub.data() , Asub.data()+
                                     Asubsub.spans().at(0).first()*Asub.strides().at(0) +
                                     Asubsub.spans().at(1).first()*Asub.strides().at(1)+
                                     Asubsub.spans().at(2).first()*Asub.strides().at(2));
  }

}


BOOST_FIXTURE_TEST_CASE_TEMPLATE(subtensor_copy_ctor_test, value,  test_types, fixture )
{
  namespace ublas = boost::numeric::ublas;
  using value_type  = typename value::first_type;
  using layout_type = typename value::second_type;
  using tensor_type = ublas::tensor_dynamic<value_type, layout_type>;
  using subtensor_type = typename tensor_type::subtensor_type;

  auto check = [](auto const& e)
  {

    auto A    = tensor_type{e};
    value_type i{};
    for(auto & a : A)
      a = i+=value_type{1};

    auto Asub = subtensor_type( A );
    auto Bsub = subtensor_type( A );

    BOOST_CHECK( Asub.span_strides() == A.strides() );
    BOOST_CHECK( Asub.strides()      == A.strides() );
    BOOST_CHECK( Asub.extents()      == A.extents() );
    BOOST_CHECK( Asub.data()         == A.data() );

    BOOST_CHECK( Bsub.span_strides() == A.strides() );
    BOOST_CHECK( Bsub.strides()      == A.strides() );
    BOOST_CHECK( Bsub.extents()      == A.extents() );
    BOOST_CHECK( Bsub.data()         == A.data()    );

    BOOST_CHECK_EQUAL (  Bsub.size() , A.size() );
    BOOST_CHECK_EQUAL (  Bsub.rank() , A.rank() );

    if(ublas::empty(e)) {
      BOOST_CHECK       ( Bsub.empty()    );
      BOOST_CHECK_EQUAL ( Bsub.data() , nullptr);
    }
    else{
      BOOST_CHECK       ( !Bsub.empty()    );
      BOOST_CHECK_NE    (  Bsub.data() , nullptr);
    }

    for(auto i = 0ul; i < Asub.size(); ++i)
      BOOST_CHECK_EQUAL( Asub[i], Bsub[i]  );

  };

  for(auto const& e : extents)
    check(e);

}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_subtensor_read_write_single_index_access, value,  test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
  using tensor_type    = ublas::tensor_dynamic<value_type, layout_type>;
  using subtensor_type = typename tensor_type::subtensor_type;


	for(auto const& e : extents) {
		auto t = tensor_type{e};
		auto v = value_type {};
		auto s = subtensor_type(t);
    for(auto i = 0ul; i < s.size(); ++i, v+=value_type{1}){
			s[i] = v;
			BOOST_CHECK_EQUAL( t[i], v );

			s(i) = v;
			BOOST_CHECK_EQUAL( t(i), v );
		}
	}
}



BOOST_FIXTURE_TEST_CASE_TEMPLATE( test_subtensor_read_write_multi_index_access_at, value,  test_types, fixture)
{
  namespace ublas = boost::numeric::ublas;
	using value_type  = typename value::first_type;
	using layout_type = typename value::second_type;
  using tensor_type    = ublas::tensor_dynamic<value_type, layout_type>;
  using subtensor_type = typename tensor_type::subtensor_type;

	auto check1 = [](const auto& t)
    {
        auto v = value_type{};
        for(auto k = 0ul; k < t.size(); ++k){
            BOOST_CHECK_EQUAL(t[k], v);
            v+=value_type{1};
        }
    };

    auto check2 = [](const auto& t)
    {
      std::array<unsigned,2> k = {0,0};
        auto r = std::is_same<layout_type,ublas::layout::first_order>::value ? 1 : 0;
        auto q = std::is_same<layout_type,ublas::layout::last_order >::value ? 1 : 0;
        auto v = value_type{};
        for(k[r] = 0ul; k[r] < t.size(r); ++k[r]){
            for(k[q] = 0ul; k[q] < t.size(q); ++k[q]){
                BOOST_CHECK_EQUAL(t.at(k[0],k[1]), v);
                v+=value_type{1};
            }
        }
    };

    auto check3 = [](const auto& t)
    {
        std::array<unsigned,3> k = {0,0,0};
        using op_type = std::conditional_t<std::is_same_v<layout_type,ublas::layout::first_order>, std::minus<>, std::plus<>>;
        auto r = std::is_same_v<layout_type,ublas::layout::first_order> ? 2 : 0;
        auto o = op_type{};
        auto v = value_type{};
        for(k[r] = 0ul; k[r] < t.size(r); ++k[r]){
            for(k[o(r,1)] = 0ul; k[o(r,1)] < t.size(o(r,1)); ++k[o(r,1)]){
                for(k[o(r,2)] = 0ul; k[o(r,2)] < t.size(o(r,2)); ++k[o(r,2)]){
                    BOOST_CHECK_EQUAL(t.at(k[0],k[1],k[2]), v);
                    v+=value_type{1};
                }
            }
        }
    };

    auto check4 = [](const auto& t)
    {
        std::array<unsigned,4> k = {0,0,0,0};
        using op_type = std::conditional_t<std::is_same_v<layout_type,ublas::layout::first_order>, std::minus<>, std::plus<>>;
        auto r = std::is_same_v<layout_type,ublas::layout::first_order> ? 3 : 0;
        auto o = op_type{};
        auto v = value_type{};
        for(k[r] = 0ul; k[r] < t.size(r); ++k[r]){
            for(k[o(r,1)] = 0ul; k[o(r,1)] < t.size(o(r,1)); ++k[o(r,1)]){
                for(k[o(r,2)] = 0ul; k[o(r,2)] < t.size(o(r,2)); ++k[o(r,2)]){
                    for(k[o(r,3)] = 0ul; k[o(r,3)] < t.size(o(r,3)); ++k[o(r,3)]){
                        BOOST_CHECK_EQUAL(t.at(k[0],k[1],k[2],k[3]), v);
                        v+=value_type{1};
                    }
                }
            }
        }
    };

	auto check = [check1,check2,check3,check4](auto const& e) {
		auto t = tensor_type{e};
		auto v = value_type {};
		for(auto i = 0ul; i < t.size(); ++i){
			t[i] = v;
			v+=value_type{1};
		}

    auto s = subtensor_type(t);

	  if     (t.rank() == 1) check1(s);
	  else if(t.rank() == 2) check2(s);
	  else if(t.rank() == 3) check3(s);
	  else if(t.rank() == 4) check4(s);

	};

	for(auto const& e : extents)
		check(e);
}


BOOST_AUTO_TEST_SUITE_END()
