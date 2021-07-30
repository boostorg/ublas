#include <boost/numeric/ublas/tensor.hpp>

using namespace boost::numeric::ublas;

int main() {
	const auto ts = tensor_dynamic<>();
	auto sts = ts(span(1,2), span(2,3));
	// auto sts_sts = sts(span(1,2), span(2,3));
	// auto sts_sts_sts = sts_sts(span(1,2), span(2,3));
}
