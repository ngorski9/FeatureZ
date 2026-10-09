#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

#include "vf_tf/point_extraction.hpp"

namespace py = pybind11;

namespace {

constexpr int array_flags = py::array::f_style | py::array::forcecast;

inline void ensure_matching_dtype(const py::array& first, const py::array& second, const char* message) {
    if( !first.dtype().is(second.dtype()) ){
        throw py::value_error(message);
    }
}

inline void ensure_matching_dtype(
    const py::array& first,
    const py::array& second,
    const py::array& third,
    const char* message
) {
    if( !first.dtype().is(second.dtype()) || !first.dtype().is(third.dtype()) ){
        throw py::value_error(message);
    }
}

inline void ensure_matching_dtype(
    const py::array& first,
    const py::array& second,
    const py::array& third,
    const py::array& fourth,
    const char* message
) {
    if( !first.dtype().is(second.dtype()) || !first.dtype().is(third.dtype()) || !first.dtype().is(fourth.dtype()) ){
        throw py::value_error(message);
    }
}

}

py::tuple get_vf_cp_py(py::array u, py::array v) {
    ensure_matching_dtype(u, v, "u and v must have the same dtype");

    const py::dtype dtype = u.dtype();

    if( dtype.is( py::dtype::of<float>() ) ){
        return get_vf_cp_points<float>(
            u.cast<py::array_t<float, array_flags>>(),
            v.cast<py::array_t<float, array_flags>>()
        );
    }

    if( dtype.is( py::dtype::of<double>() ) ){
        return get_vf_cp_points<double>(
            u.cast<py::array_t<double, array_flags>>(),
            v.cast<py::array_t<double, array_flags>>()
        );
    }

    throw py::type_error("u and v must be float32 or float64");
}

py::array classify_vf_cp_py(py::array u, py::array v) {
    ensure_matching_dtype(u, v, "u and v must have the same dtype");

    const py::dtype dtype = u.dtype();

    if( dtype.is( py::dtype::of<float>() ) ){
        return classify_vf_cp<float>(
            u.cast<py::array_t<float, array_flags>>(),
            v.cast<py::array_t<float, array_flags>>()
        );
    }

    if( dtype.is( py::dtype::of<double>() ) ){
        return classify_vf_cp<double>(
            u.cast<py::array_t<double, array_flags>>(),
            v.cast<py::array_t<double, array_flags>>()
        );
    }

    throw py::type_error("u and v must be float32 or float64");
}

py::tuple get_tf_dp_py(
    py::array A,
    py::array B,
    py::array C,
    py::array D,
    bool symmetric
) {
    if( symmetric ){
        ensure_matching_dtype(A, B, D, "A, B, and D must have the same dtype");

        const py::dtype dtype = A.dtype();

        if( dtype.is( py::dtype::of<float>() ) ){
            return get_symmetric_dp_points<float>(
                A.cast<py::array_t<float, array_flags>>(),
                B.cast<py::array_t<float, array_flags>>(),
                D.cast<py::array_t<float, array_flags>>()
            );
        }

        if( dtype.is( py::dtype::of<double>() ) ){
            return get_symmetric_dp_points<double>(
                A.cast<py::array_t<double, array_flags>>(),
                B.cast<py::array_t<double, array_flags>>(),
                D.cast<py::array_t<double, array_flags>>()
            );
        }

        throw py::type_error("A, B, and D must be float32 or float64");
    }

    ensure_matching_dtype(A, B, C, D, "A, B, C, and D must have the same dtype");

    const py::dtype dtype = A.dtype();

    if( dtype.is( py::dtype::of<float>() ) ){
        return get_asymmetric_dp_points<float>(
            A.cast<py::array_t<float, array_flags>>(),
            B.cast<py::array_t<float, array_flags>>(),
            C.cast<py::array_t<float, array_flags>>(),
            D.cast<py::array_t<float, array_flags>>()
        );
    }

    if( dtype.is( py::dtype::of<double>() ) ){
        return get_asymmetric_dp_points<double>(
            A.cast<py::array_t<double, array_flags>>(),
            B.cast<py::array_t<double, array_flags>>(),
            C.cast<py::array_t<double, array_flags>>(),
            D.cast<py::array_t<double, array_flags>>()
        );
    }

    throw py::type_error("A, B, C, and D must be float32 or float64");
}

py::array classify_tf_dp_py(
    py::array A,
    py::array B,
    py::array C,
    py::array D,
    bool symmetric
) {
    if( symmetric ){
        ensure_matching_dtype(A, B, D, "A, B, and D must have the same dtype");

        const py::dtype dtype = A.dtype();

        if( dtype.is( py::dtype::of<float>() ) ){
            return classify_symmetric_tf_dp<float>(
                A.cast<py::array_t<float, array_flags>>(),
                B.cast<py::array_t<float, array_flags>>(),
                D.cast<py::array_t<float, array_flags>>()
            );
        }

        if( dtype.is( py::dtype::of<double>() ) ){
            return classify_symmetric_tf_dp<double>(
                A.cast<py::array_t<double, array_flags>>(),
                B.cast<py::array_t<double, array_flags>>(),
                D.cast<py::array_t<double, array_flags>>()
            );
        }

        throw py::type_error("A, B, and D must be float32 or float64");
    }

    ensure_matching_dtype(A, B, C, D, "A, B, C, and D must have the same dtype");

    const py::dtype dtype = A.dtype();

    if( dtype.is( py::dtype::of<float>() ) ){
        return classify_asymmetric_dp<float>(
            A.cast<py::array_t<float, array_flags>>(),
            B.cast<py::array_t<float, array_flags>>(),
            C.cast<py::array_t<float, array_flags>>(),
            D.cast<py::array_t<float, array_flags>>()
        );
    }

    if( dtype.is( py::dtype::of<double>() ) ){
        return classify_asymmetric_dp<double>(
            A.cast<py::array_t<double, array_flags>>(),
            B.cast<py::array_t<double, array_flags>>(),
            C.cast<py::array_t<double, array_flags>>(),
            D.cast<py::array_t<double, array_flags>>()
        );
    }

    throw py::type_error("A, B, C, and D must be float32 or float64");
}

void bind_vector_tensor_topology(py::module_& m) {
    m.def(
        "get_vf_cp",
        &get_vf_cp_py,
        py::arg("u"),
        py::arg("v"),
        "Extract vector field critical points from u/v components."
    );

    m.def(
        "classify_vf_cp",
        &classify_vf_cp_py,
        py::arg("u"),
        py::arg("v"),
        "Classify vector field critical points from u/v components."
    );

    m.def(
        "get_tf_dp",
        &get_tf_dp_py,
        py::arg("A"),
        py::arg("B"),
        py::arg("C"),
        py::arg("D"),
        py::arg("symmetric"),
        "Extract tensor field degenerate points from tensor components."
    );

    m.def(
        "classify_tf_dp",
        &classify_tf_dp_py,
        py::arg("A"),
        py::arg("B"),
        py::arg("C"),
        py::arg("D"),
        py::arg("symmetric"),
        "Classify tensor field degenerate points from tensor components."
    );
}

PYBIND11_MODULE(vector_tensor_topology, m) {
    m.doc() = "Vector/tensor topology pybind module.";
    bind_vector_tensor_topology(m);
}
