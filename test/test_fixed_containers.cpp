#undef BOOST_UBLAS_NO_EXCEPTIONS
#include "common/testhelper.hpp"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <string>
#include <sstream>
#include <complex>
#include <iomanip>
#include "utils.hpp"

using namespace boost::numeric::ublas;

using std::cout;
using std::endl;

template < class T >
bool test_vector( std::string type_name)
{
    std::stringstream stream;
    stream << "Testing for: " << type_name;
    BOOST_UBLAS_DEBUG_TRACE( stream.str() );

    bool pass = true;

    {
    typedef fixed_vector<T, 3> vec3;

    vec3 v1;

    pass &=(sizeof( vec3 )  == v1.size()*sizeof( T ) ) ;

    vector<T> v( 3, 0 ) ;

    pass &= compare( v1, v );

    v1 <<= 10.0, 10, 33;
    v  <<= 10.0, 10, 33;

    //cout << std::setprecision(20) << v1 << '\n' << v;

    pass &= compare( v1, v );


    vec3 v2;

    v2( 0 ) = 10.0; v2( 1 ) = 10; v2( 2 ) = 33;
    pass &= compare( v, v2 );


    v2 += v;

    pass &= compare( v2, 2*v );


    v1 = 2*v1 + v - 6*v2;
    pass &= compare( v1, (3-2*6)*v );


    vec3 v3{ (T)-90.0, (T)-90.0, (T)-297.0 };
    pass &= compare( v3, v1 );

    vec3 v4 =  { (T)-90.0, (T)-90.0, (T)-297.0 };
    pass &= compare( v4, v1 );

    vec3 v5( (T)-90.0, (T)-90.0, (T)-297.0 );
    pass &= compare( v5, v1 );

    vec3 v6((T) 5.0, (T)8.0, (T)9.0);

    matrix<T> M = outer_prod( v6, v6), L( 3, 3);

    L <<= 25, 40, 45, 40, 64, 72, 45, 72, 81;

    pass &= compare( M, L );

    L  <<= 1, 2, 3, 4, 5, 6, 7, 8, 9;
    v6 <<= 4, 5, 6;
    vec3 v7 ( (T)32.0, (T)77.0, (T)122.0 );

    pass &= compare( v7, prod(L, v6) );

    vec3 v8;
    noalias( v8 ) = prod(L, v6);

    pass &= compare( v7, v8 );

    }


    {
        const std::size_t N = 33;
        typedef fixed_vector<T, N> vec33;

        vec33 v1;
        vector<T> v( N );

        for ( std::size_t i = 0; i!= v1.size(); i++)
        {
            v1( i ) = 3.14159*i*i;
            v ( i ) = 3.14159*i*i;
        }

        pass &= compare( v1, v );


        auto ip = inner_prod( v, v);
        auto ip1 = inner_prod( v1, v1);

        pass &= (  ip == ip1 ) ;

        T c = 0;
        for (auto i = v1.begin(); i != v1.end(); i++)
        {
            *i = c;
            c = c + 1;
        }

        c = 0;
        for (auto i = v.begin(); i != v.end(); i++)
        {
            *i = c;
            c = c + 1;
        }

        pass &= compare( v1, v );

         try {
            T a;
            v1( 100 );
            a=v1( 100 );
            (void) a ;

        } catch ( bad_index &e) {
            std::cout << " Caught: " << e.what() << endl;
            pass &= true;
        }


    }
    return pass;
}

BOOST_UBLAS_TEST_DEF (test_vector) {

    BOOST_UBLAS_DEBUG_TRACE( "Starting fixed container tests" );

    BOOST_UBLAS_TEST_CHECK(  test_vector< double >( "double") );
    BOOST_UBLAS_TEST_CHECK(  test_vector< float >( "float") );
    BOOST_UBLAS_TEST_CHECK(  test_vector< int >( "int") );

    BOOST_UBLAS_TEST_CHECK(  test_vector< std::complex<double> >( "std::complex<double>") );
    BOOST_UBLAS_TEST_CHECK(  test_vector< std::complex<float> >( "std::complex<float>") );
    BOOST_UBLAS_TEST_CHECK(  test_vector< std::complex<int> >( "std::complex<int>") );
}

int main () {

#ifdef BOOST_UBLAS_NO_EXCEPTIONS
    std::cout << "DEFINED       SDFSDF SDF SDF " << std::endl;
#endif

    BOOST_UBLAS_TEST_BEGIN();

    BOOST_UBLAS_TEST_DO( test_vector );

    BOOST_UBLAS_TEST_END();

    return EXIT_SUCCESS;

}
