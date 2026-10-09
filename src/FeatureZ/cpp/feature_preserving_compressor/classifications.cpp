#include "classifications.hpp"

#include <iostream>

classification_list_py get_classifications_py(
    const std::vector<py::array>& f,
    const std::vector<py::object>& functions,
    const std::vector<py::object>& param_dicts
){
    classification_list_py classifications_py;
    classifications_py.reserve(functions.size());

    py::tuple f_tuple(f.size());
    for(size_t i = 0; i < f.size(); ++i){
        f_tuple[i] = f[i];
    }

    for (size_t i = 0; i < functions.size(); ++i) {
        const py::object& function = functions[i];
        py::dict kwargs = param_dicts[i].cast<py::dict>();
        classifications_py.push_back( function(*f_tuple, **kwargs).cast<py::array>() );
    }

    return classifications_py;
}

// not combined with the previous function due to lifetime issues.
classification_list classifications_from_py(const classification_list_py& classifications_py ){
    classification_list classifications;
    classifications.reserve(classifications_py.size());

    for( const auto& array : classifications_py ){

        const py::dtype dtype = array.dtype();

        if( dtype.is( py::dtype::of<int32_t>() ) ){
            classifications.push_back( static_cast<const int32_t*>(array.data()) );
        } else if( dtype.is( py::dtype::of<float>() ) ){
            classifications.push_back( reinterpret_cast<const int32_t*>(array.data()) );
        } else {
            throw py::value_error("Classification dtype must be int32 or float32");
        }

    }

    return classifications;
}

bool classifications_match(
    const classification_list& first,
    const classification_list& second,
    ssize_t num_points
){

    for(size_t i = 0; i < first.size(); ++i){
        if( !std::equal(first[i], first[i]+num_points, second[i]) ){
            return false;
        }
    }

    return true;
}