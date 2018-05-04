Changes
=====
While the tensor template class and its auxiliary classes inherits the implementation style of the matrix and vector template class, they deviate in some points. This document shall give an overview of the changes.


## Implementation [Tensor]

* The current implementation of the tensor template class __only compiles with compilers supporting C++17__ mostly because `constexpr if` is used.
* Function `max_size()` in vector and matrix class is removed. It can still be queried through the underlying storage array. Note that `max_size()` is depcretated since C++17 for the std::vector.
* Functions like `find_element()` are put outside the tensor template class as free functions. 
* Iterator structures are removed.
* Member functions `data()` return a pointer instead of an `array_type` instance.
* Data access functions are implemented with `operator[]` and function `at()`. 
* Function `operator()` will be used to select/project sections of tensors.
* Functions `insert_element()` and `erase_element()` will not be used.
* Function `resize()` will be renamed to `reshape()` using extents.
* Proxy shortcuts for tensor expressions will not be used. It does not make user code more readible or convenient.



## Implementation [Unbounded Array]
* Included move copy constructor.

## Unit-Testing [Tensor]

* The boost unit-test framework is dynamically linked.
* Unit-Tests are not executed when compiled. 
* Renamed unit test folder according to the tested template classes and functions. 
* Utilized some of the latest features of the boost unit-test framework version 1.67 such as fixtures and templates.
* Pragmas not included for unit testing.

## Documentation [Tensor]
