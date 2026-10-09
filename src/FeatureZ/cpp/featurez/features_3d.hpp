#pragma once

#include <bitset>

#include "include.hpp"
#include "sf/sf_3d.hpp"
#include "sf/sf_cp_lut.hpp"
#include "sf/sf_cp_extraction.hpp"

template<typename T>
int16_t local_order(SF3D<T>& gt_sf, SF3D<T>& re_sf, size_t x, size_t y, size_t z){

    link3d<T> gt_link = gt_sf.get_link_3d(x, y, z);
    link3d<T> re_link = re_sf.get_link_3d(x, y, z);

    T gt_val = gt_sf.get(x, y, z);
    T re_val = re_sf.get(x, y, z);

    int32_t link_classification = 0;

    for( size_t i = 0; i < num_neighbors; ++i ){

        bool gt_higher = (gt_link[i] > gt_val) || (gt_link[i] == gt_val && link_sos_3d[i]);
        bool re_higher = (re_link[i] > re_val) || (re_link[i] == re_val && link_sos_3d[i]);

        if( gt_higher != re_higher && !std::isnan(gt_link[i])){
            link_classification |= (1 << i);
        }

    }

    return link_classification;

}

template <typename T>
std::array<size_t, num_neighbors+1> order_link(link3d<T> link, T value) {
    std::array<size_t, num_neighbors + 1> ordered_indices;

    auto get_value = [&](size_t idx) -> T {
        return idx < num_neighbors ? link[idx] : value;
    };

    auto tie_rank = [&](size_t idx) -> size_t {
        if( idx < num_neighbors / 2 ){
            return idx;
        }

        if( idx == num_neighbors ){
            return num_neighbors / 2;
        }

        return idx + 1;
    };

    auto comes_before = [&](size_t lhs, size_t rhs) -> bool {
        const T lhs_value = get_value(lhs);
        const T rhs_value = get_value(rhs);

        const bool lhs_nan = std::isnan(lhs_value);
        const bool rhs_nan = std::isnan(rhs_value);

        if( lhs_nan || rhs_nan ){
            if( lhs_nan != rhs_nan ){
                return !lhs_nan;
            }

            return tie_rank(lhs) < tie_rank(rhs);
        }

        if( lhs_value != rhs_value ){
            return lhs_value < rhs_value;
        }

        return tie_rank(lhs) < tie_rank(rhs);
    };

    for( size_t i = 0; i < num_neighbors + 1; ++i ){
        ordered_indices[i] = i;
    }

    for( size_t i = 1; i < num_neighbors + 1; ++i ){
        const size_t insert_idx = ordered_indices[i];
        size_t insert_position = i;

        while( insert_position > 0 && comes_before(insert_idx, ordered_indices[insert_position - 1]) ){
            ordered_indices[insert_position] = ordered_indices[insert_position - 1];
            --insert_position;
        }

        ordered_indices[insert_position] = insert_idx;
    }

    return ordered_indices;
}

template<typename T>
int16_t link_order(SF3D<T>& gt_sf, SF3D<T>& re_sf, size_t x, size_t y, size_t z){

    link3d<T> gt_link = gt_sf.get_link_3d(x,y,z);
    link3d<T> re_link = re_sf.get_link_3d(x,y,z);

    std::array<size_t, num_neighbors+1> gt_ordered_indices = order_link<T>(gt_link, gt_sf.get(x,y,z));
    std::array<size_t, num_neighbors+1> re_ordered_indices = order_link<T>(re_link, re_sf.get(x,y,z));

    int16_t link_classification = 0;

    for( size_t i = 0; i < num_neighbors + 1; ++i ){
        if (gt_ordered_indices[i] != re_ordered_indices[i]){
            if( gt_ordered_indices[i] < num_neighbors && !std::isnan(gt_link[gt_ordered_indices[i]])){
                link_classification |= (1 << (gt_ordered_indices[i]));
            }

            if( re_ordered_indices[i] < num_neighbors && !std::isnan(re_link[re_ordered_indices[i]])){
                link_classification |= (1 << (re_ordered_indices[i]));
            }
        }
    }

    return link_classification;
}

template<typename T>
int16_t critical_points(SF3D<T>& gt_sf, SF3D<T>& re_sf, size_t x, size_t y, size_t z){

    int8_t gt_ct = classify_cp(gt_sf, x, y, z);
    int8_t re_ct = classify_cp(re_sf, x, y, z);

    if( gt_ct == re_ct ){
        return 0;
    } else {
        return local_order(gt_sf, re_sf, x, y, z);
    }

}

template<typename T>
int16_t contour_tree(SF3D<T>& gt_sf, SF3D<T>& re_sf, size_t x, size_t y, size_t z){

    int8_t gt_ct = classify_cp(gt_sf, x, y, z);
    int8_t re_ct = classify_cp(re_sf, x, y, z);
    
    if( gt_ct != re_ct && (gt_ct == 0 || gt_ct == 3 || re_ct == 0 || re_ct == 3) ){
        return local_order(gt_sf, re_sf, x, y, z);
    } else {
        return 0;
    }

}

template<typename T>
std::pair<bool,bool> is_min_or_max(link3d<T> gt_link, T value){
    bool is_min = true;
    bool is_max = true;

    for( size_t i = 0; i < num_neighbors; ++i ){
        if( gt_link[i] > value || ( gt_link[i] == value && link_sos_3d[i] ) ){
            is_max = false;
        }

        if( gt_link[i] < value || ( gt_link[i] == value && !link_sos_3d[i] )){
            is_min = false;
        }

        if( !is_min && !is_max ){
            return {false,false};
        }
    }
    return {is_min,is_max};
}

template<typename T>
int16_t MS_segmentation(SF3D<T>& gt_sf, SF3D<T>& re_sf, size_t x, size_t y, size_t z){

    int32_t link_classification = 0;

    // int8_t gt_ct = classify_cp(gt_sf, x, y, z);
    // int8_t re_ct = classify_cp(re_sf, x, y, z);

    // if( gt_ct == 1 || gt_ct == 2 || gt_ct == 4 ){
    //     return link_order(gt_sf, re_sf, x, y, z);
    // }

    // if( gt_ct != re_ct ){
    //     link_classification |= local_order(gt_sf, re_sf, x, y, z);
    // }

    // if( gt_ct != re_ct && (gt_ct == 0 || gt_ct == 3 || re_ct == 0 || re_ct == 3) ){
    //     link_classification |= local_order(gt_sf, re_sf, x, y, z);
    // }

    link3d<T> gt_link = gt_sf.get_link_3d(x, y, z);
    link3d<T> re_link = re_sf.get_link_3d(x, y, z);

    T gt_center = gt_sf.get(x,y,z);
    T re_center = re_sf.get(x,y,z);

    std::pair<bool,bool> gt_is_min_or_max = is_min_or_max(gt_link, gt_center);
    std::pair<bool,bool> re_is_min_or_max = is_min_or_max(re_link, re_center);

    if( gt_is_min_or_max.first != re_is_min_or_max.first || gt_is_min_or_max.second != re_is_min_or_max.second ){
        link_classification |= local_order(gt_sf, re_sf, x, y, z);
    }

    T gt_max_value = -INF;
    T gt_min_value = INF;

    T re_max_value = -INF;
    T re_min_value = INF;

    size_t gt_max_index = -1;
    size_t gt_min_index = -1;

    size_t re_max_index = -1;
    size_t re_min_index = -1;

    for( size_t i = 0; i < num_neighbors; ++i ){

        T gt_value = gt_link[i];

        if( gt_value >= gt_max_value && !gt_is_min_or_max.second ){
            gt_max_value = gt_value;
            gt_max_index = i;
        }

        if( gt_value < gt_min_value && !gt_is_min_or_max.first ){
            gt_min_value = gt_value;
            gt_min_index = i;
        }

        T re_value = re_link[i];

        if( re_value >= re_max_value && !gt_is_min_or_max.second ){
            re_max_value = re_value;
            re_max_index = i;
        }

        if( re_value < re_min_value && !gt_is_min_or_max.first ){
            re_min_value = re_value;
            re_min_index = i;
        }

    }

    if( gt_min_index != re_min_index && !gt_is_min_or_max.first ){
        link_classification |= (1 << gt_min_index);
        link_classification |= (1 << re_min_index);
    }

    if( gt_max_index != re_max_index && !gt_is_min_or_max.second ){
        link_classification |= (1 << gt_max_index);
        link_classification |= (1 << re_max_index);
    }

    return link_classification;

}
