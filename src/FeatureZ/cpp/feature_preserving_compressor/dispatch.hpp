#pragma once

#include "feature_preserving_compressor/compress_scalar.hpp"
#include "common/pybind_conversions.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <string>
#include <vector>
#include <utility>
#include <cstdint>
#include <sys/types.h>

namespace py = pybind11;

py::bytes feature_preserving_compress_impl(
    const std::string& data_type,
    const py::dict& gt_data,
    const py::dict& de_data,
    const py::list& functions,
    const py::list& param_dicts,
    const py::list& compression_strategy_input,
    const double xi
);

py::dict feature_preserving_decompress_impl(
    const std::string& data_type,
    const py::dict& de_data,
    const py::bytes& compressed_bytes,
    const double xi
);
