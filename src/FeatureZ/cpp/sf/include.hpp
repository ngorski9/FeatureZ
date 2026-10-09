#pragma once

#include <vector>
#include <sys/types.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <array>
#include <cstring>
#include <stdexcept>

namespace py = pybind11;

template<typename T>
using link3d = std::array<T,14>;

template <typename T>
inline T get_nan() {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "get_nan<T>() only supports float or double");
    return std::numeric_limits<T>::quiet_NaN();
}