#pragma once

#include <pybind11/pybind11.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

inline std::unordered_map<std::string, py::object> dict_to_string_object_map(const py::dict& d) {
    std::unordered_map<std::string, py::object> out;
    out.reserve(static_cast<std::size_t>(py::len(d)));
    for (auto item : d) {
        const std::string key = py::cast<std::string>(py::str(item.first));
        out.emplace(key, py::reinterpret_borrow<py::object>(item.second));
    }
    return out;
}

inline std::vector<py::object> list_to_object_vector(const py::list& l) {
    std::vector<py::object> out;
    out.reserve(static_cast<std::size_t>(py::len(l)));
    for (py::handle item : l) {
        out.emplace_back(py::reinterpret_borrow<py::object>(item));
    }
    return out;
}

