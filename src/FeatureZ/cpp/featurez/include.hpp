#pragma once

#include <sys/types.h>
#include <limits>
#include <vector>
#include <array>
#include <unordered_map>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

constexpr int32_t MAX_ITERATIONS = 10;

constexpr int32_t LOSSLESS_SYMBOL = std::numeric_limits<int32_t>::max();
constexpr double INF = std::numeric_limits<double>::infinity();

constexpr int SKIP_HOT_LOOP = -1;
constexpr int CRITICAL_POINTS = 0;
constexpr int LOCAL_ORDER = 1;
constexpr int MORSE_SMALE_SEGMENTATION = 2;
constexpr int CONTOUR_TREE = 3;

