#pragma once

#include <algorithm>

#include "include.hpp"

template<typename T>
py::tuple contours_ublb_impl_template(
    const py::array_t<T, py::array::f_style | py::array::forcecast>& sf_py,
    const py::list& contour_values_py
){
    if( sf_py.ndim() != 2 && sf_py.ndim() != 3 ){
        throw py::value_error("contours_ublb requires a 2D or 3D scalar field.");
    }

    std::vector<T> contour_values;
    contour_values.reserve(py::len(contour_values_py));

    for( const py::handle& value : contour_values_py ){
        contour_values.push_back(py::cast<T>(value));
    }

    std::sort(contour_values.begin(), contour_values.end());

    py::array_t<T, py::array::f_style> ub_py =
        (sf_py.ndim() == 3)
            ? py::array_t<T, py::array::f_style>({
                sf_py.shape(0),
                sf_py.shape(1),
                sf_py.shape(2)
            })
            : py::array_t<T, py::array::f_style>({
                sf_py.shape(0),
                sf_py.shape(1)
            });
    py::array_t<T, py::array::f_style> lb_py =
        (sf_py.ndim() == 3)
            ? py::array_t<T, py::array::f_style>({
                sf_py.shape(0),
                sf_py.shape(1),
                sf_py.shape(2)
            })
            : py::array_t<T, py::array::f_style>({
                sf_py.shape(0),
                sf_py.shape(1)
            });

    const T* sf = sf_py.data();
    T* ub = ub_py.mutable_data();
    T* lb = lb_py.mutable_data();

    const size_t num_points = sf_py.size();
    const T neg_inf = -std::numeric_limits<T>::infinity();
    const T pos_inf = std::numeric_limits<T>::infinity();

    for( size_t i = 0; i < num_points; ++i ){
        const T value = sf[i];
        auto lower_it = std::lower_bound(contour_values.begin(), contour_values.end(), value);
        auto upper_it = std::upper_bound(contour_values.begin(), contour_values.end(), value);

        if( contour_values.empty() ){
            lb[i] = neg_inf;
            ub[i] = pos_inf;
        } else if( lower_it != contour_values.end() && *lower_it == value ){
            lb[i] = value;
            ub[i] = value;
        } else if( upper_it == contour_values.begin() ){
            lb[i] = neg_inf;
            ub[i] = *upper_it;
        } else if( upper_it == contour_values.end() ){
            lb[i] = contour_values.back();
            ub[i] = pos_inf;
        } else {
            lb[i] = *(upper_it - 1);
            ub[i] = *upper_it;
        }
    }

    return py::make_tuple(ub_py, lb_py);
}
