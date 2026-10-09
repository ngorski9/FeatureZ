#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "featurez/main.hpp"

namespace py = pybind11;

PYBIND11_MODULE(featurez, m) {
    m.doc() = "Feature-preserving compressor 2 pybind module.";

    m.def(
        "_featurez_compress",
        &featurez_compress_impl,
        py::arg("ground_truth_data"),
        py::arg("decompressed_data"),
        py::arg("upper_bounds"),
        py::arg("lower_bounds"),
        py::arg("xi"),
        py::arg("compression_feature"),
        py::arg("linear_quantization") = false,
        py::arg("ablation_ublb") = false,
        py::arg("ablation_star") = false,
        "Return correction bytes and the modified scalar field as a NumPy array."
    );

    m.def(
        "_featurez_decompress",
        &featurez_decompress_impl,
        py::arg("decompressed_data"),
        py::arg("compressed_bytes"),
        py::arg("xi"),
        "Decompress with featurez. Returns output as a numpy array."
    );

    m.def(
        "_contour_tree_ublb",
        &contour_tree_ublb_impl,
        py::arg("join_tree_points"),
        py::arg("join_tree_segmentation"),
        py::arg("split_tree_points"),
        py::arg("split_tree_segmentation"),
        py::arg("scalar_field"),
        py::arg("epsilon"),
        "Given raw join and split trees, apply simplification of epsilon and compute the upper and lower bounds for the contour tree induced segmentation"
    );

    m.def(
        "_contours_ublb",
        &contours_ublb_impl,
        py::arg("scalar_field"),
        py::arg("contour_values"),
        "Given a scalar field and contour values, compute per-point lower and upper contour bounds."
    );
}
