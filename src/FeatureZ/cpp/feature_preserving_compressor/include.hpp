#pragma once

#include <sys/types.h>
#include <limits>
#include <vector>
#include <array>
#include <unordered_map>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "feature_preserving_compressor/region.hpp"

namespace py = pybind11;

constexpr int32_t MAX_ITERATIONS = 10;

constexpr int32_t LOSSLESS_SYMBOL = std::numeric_limits<int32_t>::max();
constexpr double INF = std::numeric_limits<double>::infinity();

template<typename T>
inline bool contains(const std::vector<T>& v, const T& val) {
    return std::find(v.begin(), v.end(), val) != v.end();
}

// note: lifetime depends on the lifetime of the pointer. Can cause potential issues in Python if not careful.
template <class T>
py::array_t<T> make_py_array(T* ptr, ssize_t size_x, ssize_t size_y, ssize_t size_z, py::handle base = py::none())
{
    const ssize_t item = static_cast<ssize_t>(sizeof(T));
    std::array<ssize_t, 3> shape   = {size_x, size_y, size_z};
    std::array<ssize_t, 3> strides = {item, item * size_x, item * size_x * size_y};

    return py::array_t<T>( shape, strides, ptr, base );
}