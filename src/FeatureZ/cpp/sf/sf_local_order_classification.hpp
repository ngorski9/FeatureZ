#pragma once

#include <cstdint>

#include "sf/include.hpp"
#include "sf/sf_3d.hpp"
#include "sf_cp_lut.hpp"

template<typename T>
int32_t classify_local_order_bits_3d(const SF3D<T>& sf, size_t x, size_t y, size_t z){
    const link3d<T> link = sf.get_link_3d(x, y, z);
    const T value = sf.get(x, y, z);

    int32_t bits = 0;

    for( size_t i = 0; i < num_neighbors; ++i ){
        const T neighbor = link[i];

        if( neighbor > value || (neighbor == value && link_sos_3d[i]) ){
            bits |= (int32_t(1) << i);
        }
    }

    return bits;
}

template<typename T>
py::array_t<int32_t> classify_sf_local_order_3d(const SF3D<T>& sf){
    const size_t size_x = sf.get_size_x();
    const size_t size_y = sf.get_size_y();
    const size_t size_z = sf.get_size_z();

    py::array_t<int32_t> out(
        py::array::ShapeContainer{
            static_cast<py::ssize_t>(size_x),
            static_cast<py::ssize_t>(size_y),
            static_cast<py::ssize_t>(size_z)
        }
    );
    auto out_view = out.template mutable_unchecked<3>();

    for( size_t x = 0; x < size_x; ++x ){
        for( size_t y = 0; y < size_y; ++y ){
            for( size_t z = 0; z < size_z; ++z ){
                out_view(x, y, z) = classify_local_order_bits_3d(sf, x, y, z);
            }
        }
    }

    return out;
}

template<typename T>
py::array_t<int32_t> get_sf_local_order_classification(
    const py::array_t<T, py::array::f_style | py::array::forcecast>& f_py
){
    if( f_py.ndim() != 3 ){
        throw py::value_error("Local-order classification requires a 3D array.");
    }

    const size_t size_x = f_py.shape(0);
    const size_t size_y = f_py.shape(1);
    const size_t size_z = f_py.shape(2);

    py::buffer_info f_info = f_py.request();
    T* f = static_cast<T*>(f_info.ptr);

    const SF3D<T> sf(f, size_x, size_y, size_z);
    return classify_sf_local_order_3d(sf);
}
