#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

#include "sf/sf_cp_extraction.hpp"
#include "sf/sf_local_order_classification.hpp"

namespace py = pybind11;

namespace {

constexpr int array_flags = py::array::f_style | py::array::forcecast;

}

py::tuple get_sf_cp_py(py::array f) {
    const py::dtype dtype = f.dtype();

    if( dtype.is( py::dtype::of<float>() ) ){
        return get_sf_cp_points<float>(
            f.cast<py::array_t<float, array_flags>>()
        );
    }

    if( dtype.is( py::dtype::of<double>() ) ){
        return get_sf_cp_points<double>(
            f.cast<py::array_t<double, array_flags>>()
        );
    }

    throw py::type_error("f must be float32 or float64");
}

py::array_t<int32_t> get_sf_local_order_classification_py(py::array f) {
    const py::dtype dtype = f.dtype();

    if( dtype.is( py::dtype::of<float>() ) ){
        return get_sf_local_order_classification<float>(
            f.cast<py::array_t<float, array_flags>>()
        );
    }

    if( dtype.is( py::dtype::of<double>() ) ){
        return get_sf_local_order_classification<double>(
            f.cast<py::array_t<double, array_flags>>()
        );
    }

    throw py::type_error("f must be float32 or float64");
}

void bind_scalar_field_topology(py::module_& m) {
    m.def(
        "get_sf_cp",
        &get_sf_cp_py,
        py::arg("f"),
        "Extract scalar-field critical points from a 3D scalar array."
    );

    m.def(
        "get_sf_local_order_classification",
        &get_sf_local_order_classification_py,
        py::arg("f"),
        "Return a 3D int32 array whose lower 14 bits encode the local-order comparison mask."
    );
}

PYBIND11_MODULE(tensor_topology, m) {
    m.doc() = "Scalar-field topology pybind module.";
    bind_scalar_field_topology(m);
}
