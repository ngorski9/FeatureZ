#pragma once

#include <iomanip>

#include "featurez/include.hpp"
#include "featurez/features_3d.hpp"

#include "common/quantization.hpp"
#include "common/huffman.hpp"
#include "sf/sf_3d.hpp"
#include "sf/sf_cp_extraction.hpp"

// template<typename T>
// inline int32_t classify_link(SF3D<T>& sf, size_t x, size_t y, size_t z, int compression_feature){
//     if( compression_feature == CRITICAL_POINTS ){
//         return classify_cp(sf, x, y, z);
//     } else if(compression_feature == LOCAL_ORDER) {
//         return local_order(sf, x, y, z);
//     } else if(compression_feature == IS_EXTREMUM) {
//         return is_extremum(sf, x, y, z);
//     } else if(compression_feature == MORSE_SMALE_SEGMENTATION){
//         return MorseSmaleSegmentation(sf, x, y, z);  
//     } else {
//         throw std::runtime_error("Unknown compression feature " + std::to_string(compression_feature));
//     }
// }

template<typename T, size_t N>
inline int16_t get_link_errors(SF3D<T>& gt_sf, SF3D<T>& re_sf, size_t x, size_t y, size_t z, std::array<int, N> compression_features){
    int16_t errors = 0;

    #pragma unroll
    for( size_t i = 0; i < N; ++i ){
        int compression_feature = compression_features[i]; 
     
        if( compression_feature == SKIP_HOT_LOOP ){
            continue;
        } else if( compression_feature == CRITICAL_POINTS ){
            errors |= critical_points(gt_sf, re_sf, x, y, z);
        } else if( compression_feature == LOCAL_ORDER ){
            errors |= local_order(gt_sf, re_sf, x, y, z);
        } else if( compression_feature == MORSE_SMALE_SEGMENTATION ){
            errors |= MS_segmentation(gt_sf, re_sf, x, y, z);
        } else if( compression_feature == CONTOUR_TREE ){
            errors |= contour_tree(gt_sf, re_sf, x, y, z);
        } else {
            throw std::runtime_error("Unknown compression feature " + std::to_string(compression_feature));
        }
    }

    return errors;
}

template<typename T, bool linear_quantization>
void initial_quantization(
    SF3D<T>& gt_sf,
    SF3D<T>& de_sf,
    T* re_sf_ptr,
    T* ub,
    T* lb,
    T xi,
    std::vector<int32_t>& quantizations,
    std::vector<int8_t>& precisions,
    quantization_constants<T>& constants,
    size_t num_points
){
    
    for( size_t i = 0; i < num_points; ++i ){

        T gt = gt_sf.get(i);
        T de = de_sf.get(i);
        ublb<T> bounds = {
            std::min(ub[i], gt + xi),
            std::max(lb[i], gt - xi)
        };

        int8_t precision = 0;

        if constexpr (linear_quantization){

            point_quantization<T> pq = quantize_for_precision_or_lossless<T,false>(gt, de, bounds, 0, 0, constants);

            if( pq.quantization_number == LOSSLESS_SYMBOL ){
                precisions[i] = MAX_PRECISION + 1;
            } else {
                precisions[i] = 1;
            }

            quantizations[i] = pq.quantization_number;
            re_sf_ptr[i] = pq.value;

        } else {
            point_quantization<T> pq = quantize_point_lowest_number<T,true>(gt, de, bounds, constants[precision]);

            while( !pq.success && precision < MAX_PRECISION ){
                ++precision;
                pq = quantize_point_lowest_number<T,true>(gt, de, bounds, constants[precision]);
            }

            if( !pq.success ){
                precisions[i] = MAX_PRECISION + 1;
                quantizations[i] = LOSSLESS_SYMBOL;
                re_sf_ptr[i] = gt_sf.get(i);
            } else {
                precisions[i] = precision;
                quantizations[i] = pq.quantization_number;
                re_sf_ptr[i] = pq.value;
            }
        }

        // debug only
        // if( !( lb[i] <= re_sf_ptr[i] && re_sf_ptr[i] <= ub[i]) ){
        //     std::cout << "bad initial quantization " << (int)i << std::endl;
        //     std::cout << lb[i] << " <= " << re_sf_ptr[i] << " <= " << ub[i] << std::endl;
        //     std::cout << "(gt " << gt_sf.get(i) << ")" << std::endl;
        //     std::cout << "de: " << de << std::endl;
        //     std::cout << "precision: " << (int)precision << std::endl;
        //     std::cout << "success: " << (int)pq.success << std::endl;
        //     exit(0);
        // }

    }

}

template<typename T, bool linear_quantization>
inline void hot_loop_interior_quantize(
    SF3D<T>& gt_sf,
    SF3D<T>& de_sf,
    SF3D<T>& re_sf,
    T* ub,
    T* lb,
    T xi,
    std::vector<int8_t>& precisions,
    std::vector<int32_t>& quantizations,
    size_t index,
    quantization_constants<T>& constants
){
    ++precisions[index];
    if( precisions[index] > MAX_PRECISION ){
        re_sf.set(index, gt_sf.get(index));
        quantizations[index] = LOSSLESS_SYMBOL;
    } else {
        T gt = gt_sf.get(index);
        T de = de_sf.get(index);
        int32_t precision = precisions[index];

        if( ub ){

            if constexpr ( linear_quantization ){

                ublb<T> bounds = {ub[index], lb[index]};
                point_quantization<T> pq = quantize_for_precision_or_lossless<T, true>(gt, de, bounds, precision, 0, constants);

                if( pq.quantization_number == LOSSLESS_SYMBOL ){
                    precisions[index] = MAX_PRECISION + 1;
                }

                quantizations[index] = pq.quantization_number;
                re_sf.set(index, pq.value);

            } else {

                ublb<T> bounds = {
                    std::min(ub[index], gt + xi),
                    std::max(lb[index], gt - xi)
                };

                point_quantization<T> pq = quantize_point_most_accurate<T,true>(gt, de, bounds, constants[precision]);

                while( !pq.success && precision < MAX_PRECISION ){
                    ++precision;

                    pq = quantize_point_most_accurate<T,true>(gt, de, bounds, constants[precision]);                            
                }

                if( !pq.success ){
                    quantizations[index] = LOSSLESS_SYMBOL;
                    precisions[index] = MAX_PRECISION + 1;
                    re_sf.set(index, gt_sf.get(index));
                } else {
                    quantizations[index] = pq.quantization_number;
                    precisions[index] = precision;
                    re_sf.set(index, pq.value);
                }

            }

        } else {
            point_quantization<T> pq = quantize_point_most_accurate<T,false>(gt, de, {100,-100}, constants[precision]);
            quantizations[index] = pq.quantization_number;
            re_sf.set(index, pq.value);
        }

    }
}

// returns true if edited and false otherwise
template<typename T, size_t N, bool linear_quantization>
inline bool hot_loop_interior(
    const Point& p,
    SF3D<T>& gt_sf,
    SF3D<T>& de_sf,
    SF3D<T>& re_sf,
    T* ub,
    T* lb,
    T xi,
    std::vector<int8_t>& precisions,
    std::vector<int32_t>& quantizations,
    std::array<int, N> compression_features,
    quantization_constants<T>& constants
){
    int16_t link_errors = get_link_errors(gt_sf, re_sf, p.x, p.y, p.z, compression_features);

    if( link_errors == 0 ){
        return false;
    }

    std::array<size_t, num_neighbors + 1> star_indices;

    for( size_t i = 0; i < num_neighbors; ++i ){
        Point neighbor = p + neighbors[i];

        if( gt_sf.point_is_valid(neighbor) ){
            star_indices[i] = gt_sf.get_index(neighbor);
        } else {
            star_indices[i] = -1; // yes I know this will underflow.
        }
    }

    star_indices[num_neighbors] = gt_sf.get_index(p);

    while( link_errors != 0 ){
        int32_t star_errors = link_errors | (1 << num_neighbors); // always tighten the main point

        int8_t min_precision = MAX_PRECISION + 1;

        for( size_t i = 0; i < num_neighbors + 1; ++i ){

            if( star_errors & (1 << i) ){
                int8_t precision = precisions[star_indices[i]];

                if( precision < min_precision ){
                    min_precision = precision;
                }
            }
        }

        for( size_t i = 0; i < num_neighbors + 1; ++i ){

            if( !( star_errors & (1 << i) ) ){
                continue;
            }

            size_t tighten_idx = star_indices[i];

            if( precisions[tighten_idx] != min_precision ){
                continue;
            }

            hot_loop_interior_quantize<T, linear_quantization>(gt_sf, de_sf, re_sf, ub, lb, xi, precisions, quantizations, tighten_idx, constants);

        }

        link_errors = get_link_errors(gt_sf, re_sf, p.x, p.y, p.z, compression_features);        
    }

    return true;
}

template<typename T, size_t N, bool linear_quantization>
void hot_loop(SF3D<T>& gt_sf, 
              SF3D<T>& de_sf, 
              SF3D<T>& re_sf, 
              T* ub,
              T* lb,
              T xi,
              std::vector<int8_t>& precisions, 
              std::vector<int32_t>& quantizations,
              std::array<int, N> compression_features,
              quantization_constants<T>& constants
            ){
    
    std::vector<bool> dirty(gt_sf.get_num_points(), true);

    std::vector<Point> dirty_points_sparse_repr; // used for the back half of the algo
    std::unordered_set<int> dirty_points_sparse_repr_set;
    int num_dirty = gt_sf.get_num_points();
    bool sparse = false;

    while( num_dirty > 0 && !sparse ){

        if( num_dirty < gt_sf.get_num_points() / 5 ){
            sparse = true;
        }

        for( size_t z = 0; z < gt_sf.get_size_z(); ++z ){
            for( size_t y = 0; y < gt_sf.get_size_y(); ++y ){
                for( size_t x = 0; x < gt_sf.get_size_x(); ++x ){

                    int current_index = gt_sf.get_index(x, y, z);

                    if( !dirty[current_index] ){
                        continue;
                    }

                    Point next_point = {x,y,z};

                    bool did_edit = hot_loop_interior<T, N, linear_quantization>(next_point, gt_sf, de_sf, re_sf, ub, lb, xi, precisions, 
                                                            quantizations, compression_features, constants);
                                                            
                    if( did_edit ){

                        for(int i = 0; i < num_redo_neighbors; ++i){

                            if( sparse && dirty_points_sparse_repr_set.insert(current_index).second ){
                                dirty_points_sparse_repr.push_back(next_point);
                            }

                            Point possible_check_point = next_point + redo_neighbors[i];
                            int possible_check_index = gt_sf.get_index(possible_check_point);

                            if( gt_sf.point_is_valid(possible_check_point) && !dirty[possible_check_index]){

                                if( sparse ){
                                    dirty_points_sparse_repr.push_back( possible_check_point );
                                }

                                dirty[possible_check_index] = true;
                                ++num_dirty;

                            }
                        }

                    } else {
                        dirty[current_index] = false;
                        --num_dirty;
                    }

                }
            }
        }

        std::cout << "(dense) num dirty " << num_dirty << std::endl;

    } // end while (num_dirty > 0 && !sparse)

    // sparse pass
    while( num_dirty > 0 ){

        std::vector<Point> next_sparse_pass_vector;
        std::unordered_set<int> next_sparse_pass_set;

        for( auto next_point : dirty_points_sparse_repr ){

            int current_index = gt_sf.get_index(next_point);
            bool did_edit = hot_loop_interior<T, N, linear_quantization>(next_point, gt_sf, de_sf, re_sf, ub, lb, xi, precisions, 
                                                 quantizations, compression_features, constants);

            if( did_edit ){
                
                if( next_sparse_pass_set.insert(current_index).second ){
                    next_sparse_pass_vector.push_back(next_point);
                }

                for(int i = 0; i < num_redo_neighbors; ++i){
                    Point possible_check_point = next_point + redo_neighbors[i];
                    int possible_check_index = gt_sf.get_index(possible_check_point);

                    if( gt_sf.point_is_valid(possible_check_point) && next_sparse_pass_set.insert(possible_check_index).second ){
                        next_sparse_pass_vector.push_back(possible_check_point);
                    }
                }
            }

        }

        dirty_points_sparse_repr = next_sparse_pass_vector;
        num_dirty = next_sparse_pass_vector.size();

        std::cout << "(sparse) num dirty " << num_dirty << std::endl;

    }

}

template<typename T>
py::bytes compression_output_to_bytes(
    SF3D<T>& sf,
    const std::vector<int32_t>& quantization_numbers
){
    std::vector<std::uint8_t> huffman_bytes = huffman_encode(quantization_numbers);

    std::vector<T> lossless_storage;

    for( int i = 0; i < sf.get_num_points(); ++i ){
        if( quantization_numbers[i] == LOSSLESS_SYMBOL ){
            lossless_storage.push_back(sf.get(i));
        }
    }

    const std::uint64_t num_huffman_bytes = static_cast<std::uint64_t>(huffman_bytes.size());
    const std::size_t num_lossless_bytes = lossless_storage.size() * sizeof(T);

    const std::size_t total = sizeof(num_huffman_bytes) + huffman_bytes.size() + num_lossless_bytes;

    std::string out;
    out.resize(total);

    std::size_t off = 0;

    // num huffman bytes
    std::memcpy(out.data() + off, &num_huffman_bytes, sizeof(num_huffman_bytes));
    off += sizeof(num_huffman_bytes);

    // huffman bytes
    if (!huffman_bytes.empty()) {
        std::memcpy(out.data() + off, huffman_bytes.data(), huffman_bytes.size());
        off += huffman_bytes.size();
    }

    // 3) B (raw object bytes)
    if (!lossless_storage.empty()) {
        std::memcpy(out.data() + off, lossless_storage.data(), num_lossless_bytes);
        off += num_lossless_bytes;
    }

    return py::bytes(out);
}

template<typename T, size_t N>
py::tuple featurez_compress_impl_template(
    const py::array_t<T, py::array::f_style | py::array::forcecast>& gt_sf_py,
    const py::array_t<T, py::array::f_style | py::array::forcecast>& de_sf_py,
    const std::optional<py::array_t<T, py::array::f_style | py::array::forcecast>>& ub_py,
    const std::optional<py::array_t<T, py::array::f_style | py::array::forcecast>>& lb_py,
    T xi,
    std::array<int, N> compression_features,
    bool linear_quantization,
    bool ablation_ublb,
    bool ablation_star
){

    size_t size_x = gt_sf_py.shape(0);
    size_t size_y = gt_sf_py.shape(1);
    size_t size_z = gt_sf_py.shape(2);
    size_t num_points = size_x * size_y * size_z;

    std::vector<py::array> gt_f = {gt_sf_py};

    py::buffer_info gt_info = gt_sf_py.request();
    py::buffer_info de_info = de_sf_py.request();

    T* gt_sf_ptr = static_cast<T*>(gt_info.ptr);
    T* de_sf_ptr = static_cast<T*>(de_info.ptr);

    SF3D<T> gt_sf = SF3D<T>(gt_sf_ptr, size_x, size_y, size_z);
    SF3D<T> de_sf = SF3D<T>(de_sf_ptr, size_x, size_y, size_z);

    py::array_t<T, py::array::f_style> modified_sf({size_x, size_y, size_z});
    T* re_sf_ptr = modified_sf.mutable_data();

    std::vector<int8_t> precisions(num_points, 0);
    std::vector<int32_t> quantizations(num_points, 0);
    quantization_constants<T> constants = get_quantization_constants(xi);

    T* ub = NULL;
    T* lb = NULL;

    if( ub_py.has_value() && !ablation_ublb ){

        py::buffer_info ub_info = (*ub_py).request();
        py::buffer_info lb_info = (*lb_py).request();

        ub = static_cast<T*>(ub_info.ptr);
        lb = static_cast<T*>(lb_info.ptr);

        if( linear_quantization ){
            initial_quantization<T, true>(gt_sf, de_sf, re_sf_ptr, ub, lb, xi, quantizations, precisions, constants, num_points);
        } else {
            initial_quantization<T, false>(gt_sf, de_sf, re_sf_ptr, ub, lb, xi, quantizations, precisions, constants, num_points);
        }

        // debug only
        // for( size_t i = 0; i < num_points; ++i ){
        //     T value = re_sf_ptr[i];

        //     if( value > ub[i] || value < lb[i] ){
        //         std::cout << (int)i << " initial bad ublb" << std::endl;
        //         std::cout << std::setprecision(10) << "should be: " << lb[i] << " <= " << value << " <= " << ub[i] << std::endl;
        //         std::cout << std::setprecision(10) << "(gt " << gt_sf_ptr[i] << ")" << std::endl;
        //         exit(0);
        //     }
        // }

    } else {
        memcpy(re_sf_ptr, de_sf_ptr, num_points * sizeof(T));
    }


    SF3D<T> re_sf = SF3D<T>(re_sf_ptr, size_x, size_y, size_z);

    bool do_hot_loop = false;

    for(size_t i = 0; i < N; ++i){
        if(compression_features[i] != SKIP_HOT_LOOP){
            do_hot_loop = true;
        }
    }

    if( do_hot_loop && !ablation_star ){
        if( linear_quantization ){
            hot_loop<T, N, true>(gt_sf, de_sf, re_sf, ub, lb, xi, precisions, quantizations, compression_features, constants);
        } else {
            hot_loop<T, N, false>(gt_sf, de_sf, re_sf, ub, lb, xi, precisions, quantizations, compression_features, constants);
        }

    }


    // debug only
    // for(size_t i = 0; i < gt_sf.get_size_x(); ++i){
    //     for(size_t j = 0; j < gt_sf.get_size_y(); ++j){
    //         for(size_t k = 0; k < gt_sf.get_size_z(); ++k){
    //             int16_t link_errors = get_link_errors(gt_sf, re_sf, i, j, k, compression_features);

    //             if(link_errors){
    //                 std::cout << i << " " << j << " " << k << " bad class" << std::endl;
    //                 exit(0);
    //             }

    //             if( ub ){
    //                 size_t index = re_sf.get_index(i, j, k);
    //                 T value = re_sf.get(index);

    //                 if( value > ub[index] || value < lb[index] ){
    //                     std::cout << i << " " << j << " " << k << " final bad ublb" << std::endl;
    //                     std::cout << "should be: " << lb[index] << " <= " << value << " <= " << ub[index] << std::endl;
    //                     exit(0);
    //                 }
    //             }

    //         }
    //     }
    // }

    std::cout << "converting to bytes" << std::endl;

    py::bytes compression_output =  compression_output_to_bytes(gt_sf, quantizations);

    return py::make_tuple(compression_output, modified_sf);

}
