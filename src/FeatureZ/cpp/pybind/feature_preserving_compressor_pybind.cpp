#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "feature_preserving_compressor/dispatch.hpp"

namespace py = pybind11;

PYBIND11_MODULE(feature_preserving_compressor, m) {
    m.doc() = "Feature-preserving compressor pybind module.";

    m.def(
        "_feature_preserving_compress",
        &feature_preserving_compress_impl,
        py::arg("data_type"),
        py::arg("ground_truth_data"),
        py::arg("decompressed_data"),
        py::arg("feature_classification_functions"),
        py::arg("feature_classification_functions_param_dicts"),
        py::arg("compression_strategy"),
        py::arg("xi"),
        "Compress with feature preserving compressor. Returns a set of bytes that must be added to the compressed output for compression."
    );

    m.def(
        "_feature_preserving_decompress",
        &feature_preserving_decompress_impl,
        py::arg("data_type"),
        py::arg("decompressed_data"),
        py::arg("compressed_bytes"),
        py::arg("xi"),
        "Decompress with feature preserving compressor. Returns output as a numpy array."
    );
}
