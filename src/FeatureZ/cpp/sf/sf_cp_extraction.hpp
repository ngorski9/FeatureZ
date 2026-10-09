#pragma once

#include <iostream>

#include "sf/include.hpp"
#include "sf/sf_3d.hpp"
#include "sf_cp_lut.hpp"

template <typename T>
int8_t classify_link_3d_interior(const link3d<T>& link, T value){

    int32_t link_idx = 0;
    
    for( int i = 0; i < 14; ++i ){

        T neighbor = link[i];
        if( neighbor > value || (neighbor == value && link_sos_3d[i]) ){
            link_idx |= (1 << i);
        }

    }

    return interior_lut_3d[link_idx];
}

template <typename T>
int8_t classify_link_3d_boundary(const link3d<T>& link, T value){

    int32_t lower_link_idx = 0;
    int32_t upper_link_idx = 0;
    int32_t lower_link_complement_idx = 0;
    int32_t upper_link_complement_idx = 0;

    for( int i = 0; i < 14; ++i ){

        T neighbor = link[i];
        
        if( !isnan(neighbor) ){

            if( neighbor > value || (neighbor == value && link_sos_3d[i]) ){
                upper_link_idx |= (1 << i);
                lower_link_complement_idx |= (1 << i);
            } else {
                lower_link_idx |= (1 << i);
                upper_link_complement_idx |= (1 << i);
            }

        } else {
            lower_link_complement_idx |= (1 << i);
            upper_link_complement_idx |= (1 << i);
        }

    }

    int8_t upper_cc = connected_component_lut_3d[upper_link_idx];
    int8_t lower_cc = connected_component_lut_3d[lower_link_idx];

    if( upper_cc == 1 && lower_cc == 1 ){

        // look for holes
        int8_t upper_cc_complement = connected_component_lut_3d[upper_link_complement_idx];
        int8_t lower_cc_complement = connected_component_lut_3d[lower_link_complement_idx];

        if( upper_cc_complement == 1 && lower_cc_complement == 1 ){
            return -1;
        } else if( upper_cc_complement == 2 ){
            return 1;
        } else if( lower_cc_complement == 2 ){
            return 2;
        } else {
            return 4;
        }

    } else if( lower_cc == 0 ){
        return 0;
    } else if( lower_cc == 2 && upper_cc == 1 ){
        return 1;
    } else if( lower_cc == 1 && upper_cc == 2 ){
        return 2;
    } else if( upper_cc == 0 ){
        return 3;
    } else {
        return 4;
    }

}

template<typename T>
int8_t classify_cp(const SF3D<T>& sf, size_t x, size_t y, size_t z){

    link3d<T> link = sf.get_link_3d(x, y, z);
    T value = sf.get(x, y, z);

    if( x == 0 || y == 0 || z == 0 || x == sf.get_size_x()-1 || y == sf.get_size_y()-1 || z == sf.get_size_z()-1 ){
        return classify_link_3d_boundary(link, value);
    } else {
        return classify_link_3d_interior(link, value);
    }
}

template<typename T>
py::tuple extract_sf_cp_3d(const SF3D<T>& sf){
    std::vector<std::array<size_t, 3>> critical_coords;
    py::list critical_types;

    const size_t size_x = sf.get_size_x();
    const size_t size_y = sf.get_size_y();
    const size_t size_z = sf.get_size_z();

    for( size_t i = 0; i < size_x; ++i ){
        for( size_t j = 0; j < size_y; ++j ){
            for( size_t k = 0; k < size_z; ++k ){

                int8_t ct = classify_cp(sf, i, j, k);

                if( ct != -1 ){
                    critical_coords.push_back( {i,j,k} );
                    critical_types.append(ct);
                }

            }
        }
    }

    py::array_t<size_t> coords(
        py::array::ShapeContainer{
            static_cast<py::ssize_t>(critical_coords.size()),
            static_cast<py::ssize_t>(3)
        }
    );
    auto coords_view = coords.template mutable_unchecked<2>();

    for( size_t i = 0; i < critical_coords.size(); ++i ){
        coords_view(i, 0) = critical_coords[i][0];
        coords_view(i, 1) = critical_coords[i][1];
        coords_view(i, 2) = critical_coords[i][2];
    }

    return py::make_tuple(critical_types, coords);
}

template<typename T>
py::tuple get_sf_cp_points(
    const py::array_t<T, py::array::f_style | py::array::forcecast>& f_py
){
    if( f_py.ndim() != 3 ){
        throw py::value_error("Scalar-field critical point extraction requires a 3D array.");
    }

    size_t size_x = f_py.shape(0);
    size_t size_y = f_py.shape(1);
    size_t size_z = f_py.shape(2);

    py::buffer_info f_info = f_py.request();
    T* f = static_cast<T*>(f_info.ptr);

    const SF3D<T> sf(f, size_x, size_y, size_z);
    return extract_sf_cp_3d(sf);
}
