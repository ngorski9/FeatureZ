#pragma once

#include <utility>
#include <vector>
#include <sys/types.h>
#include <unordered_map>
#include <optional>
#include <limits>
#include <stdexcept>
#include <cstring>
#include <iostream>

#include "feature_preserving_compressor/include.hpp"
#include "feature_preserving_compressor/region.hpp"
#include "feature_preserving_compressor/classifications.hpp"
#include "common/huffman.hpp"
#include "common/quantization.hpp"

template<typename T>
using sf_bounds = std::vector<ublb<T>>;

struct compression_strategy_per_function{
    bool region_based_ublb = false;
    bool preserve_mins_and_maxes = false;
    bool region_based_tightening = false;
    bool most_accurate_quantization = false;
};

using compression_strategy_list = std::vector<compression_strategy_per_function>;

inline bool uses_region_based_tightening(const compression_strategy_list& strategies){
    for( const auto& strategy : strategies ){
        if( strategy.region_based_tightening ){
            return true;
        }
    }
    return false;
}

template<typename T>
sf_bounds<T> get_initial_sf_bounds(
    const T* sf,
    T xi,
    const optional_partition_list<T>& partitions,
    const compression_strategy_list& compression_strategies,
    ssize_t num_points,
    bool any_region_based_ublb,
    bool any_preserve_mins_and_maxes
){

    sf_bounds<T> b;
    b.reserve(num_points);

    for( ssize_t i = 0; i < num_points; ++i ){
        b.push_back({xi, -xi});
    }

    if( any_region_based_ublb || any_preserve_mins_and_maxes ){

        for( size_t i = 0; i < partitions.size(); ++i ){
            const auto& strategy = compression_strategies[i];
            if( !strategy.region_based_ublb && !strategy.preserve_mins_and_maxes ){
                continue;
            }
            if( !partitions[i].has_value() ){
                continue;
            }

            const auto& partition = partitions[i].value();
            for( const auto& [partition_idx, next_region] : partition ){

                if( strategy.region_based_ublb ){
                    
                    if( strategy.preserve_mins_and_maxes ){

                        for( int i = 0; i < next_region.size(); ++i ){
                            ssize_t point = next_region[i];
                            T value = sf[point];

                            if( value == next_region.max_value || value == next_region.min_value ){

                                b[point].ub = 0;
                                b[point].lb = 0;

                            } else {

                                b[point].ub = std::min(b[point].ub, next_region.max_value - value );
                                b[point].lb = std::max(b[point].lb, next_region.min_value - value );

                            }
                        }

                    } else {

                        for( int i = 0; i < next_region.size(); ++i ){
                            ssize_t point = next_region[i];
                            T value = sf[point];
                            b[point].ub = std::min(b[point].ub, next_region.max_value - value );
                            b[point].lb = std::max(b[point].lb, next_region.min_value - value );
                        }

                    }

                }

                // if( preserve_mins_and_maxes ){
                //     b[next_region.min_index].ub = 0;
                //     b[next_region.min_index].lb = 0;
                //     b[next_region.max_index].ub = 0;
                //     b[next_region.max_index].lb = 0;
                // }


            }
        }

    }

    return b;

}

template<typename T>
struct sf_quantization_result{
    std::vector<int32_t> quantization_numbers;
    std::vector<T> sf;
};

template<typename T, bool most_accurate>
sf_quantization_result<T> quantize_sf_most_accurate_template(
    const T* gt_sf,
    const T* de_sf,
    const T xi,
    const sf_bounds<T>& bounds,
    const ssize_t num_points
){

    sf_quantization_result<T> q;

    q.quantization_numbers.reserve(num_points);
    q.sf.reserve(num_points);

    std::array<quantization_constants_for_precision<T>, MAX_PRECISION+1> quantization_constants;
    for( int i = 0; i <= MAX_PRECISION; ++i ){
        quantization_constants[i] = { get_inverse_interval_width<T>(xi, i), get_quantization_stride(i), get_un_quantize_constant(xi) };
    }

    for( ssize_t i = 0; i < num_points; ++i ){

        int32_t precision = 0;
        point_quantization<T> quantization;
        T gt = gt_sf[i];
        T de = de_sf[i];

        do{

            quantization = quantize_point<T, most_accurate, true>(gt, de, bounds[i], quantization_constants[precision]);
            ++precision;

        } while( !quantization.success && precision <= MAX_PRECISION );

        // if( quantization.success && (quantization.value - gt > bounds[i].ub || quantization.value - gt < bounds[i].lb) ){
        //     std::cout << "violated" << std::endl;
        //     exit(1);
        // }

        if( quantization.success ){
            q.quantization_numbers.push_back( quantization.quantization_number );
            q.sf.push_back( quantization.value );
        } else {
            q.quantization_numbers.push_back(LOSSLESS_SYMBOL);
            q.sf.push_back( gt );
        }

    }

    return q;

}

template<typename T>
sf_quantization_result<T> quantize_sf(
    const T* gt_sf,
    const T* de_sf,
    const T xi,
    const sf_bounds<T>& bounds,
    const ssize_t num_points,
    bool most_accurate
){
    if( most_accurate ){
        return quantize_sf_most_accurate_template<T, true>(gt_sf, de_sf, xi, bounds, num_points);
    } else {
        return quantize_sf_most_accurate_template<T, false>(gt_sf, de_sf, xi, bounds, num_points);
    }
}

template<typename T>
void tighten_mismatched_sf_partitions(
    const T* sf,
    sf_bounds<T>& bounds,
    const partition<T>& partition_,
    std::vector<int32_t>& mismatch_classifications,
    ssize_t size_x,
    ssize_t size_y,
    ssize_t size_z,
    int iteration,
    int grow_amount
){

    for( auto class_ : mismatch_classifications ){

        // define region
        
        Region<T> new_region;
        const Region<T>* tighten_region;

        if( grow_amount > 0 ){
            new_region = partition_.at(class_).grow(grow_amount, size_x, size_y, size_z);
            tighten_region = &new_region;
        } else {
            tighten_region = &partition_.at(class_);
        }

        if( iteration <= MAX_ITERATIONS ){

            for( ssize_t j = 0; j < tighten_region->size(); ++j ){
                ssize_t point = (*tighten_region)[j];
                bounds[point].lb /= 2;
                bounds[point].ub /= 2;
            }

        } else {

            for( ssize_t j = 0; j < tighten_region->size(); ++ j ){
                ssize_t point = (*tighten_region)[j];
                bounds[point].lb = 0;
                bounds[point].ub = 0;
            }

        }



    } // end for gt_classification_value (iterating through different partitions)

}

template<typename T>
void tighten_sf_by_partitions(
    const T* sf,
    Region<T>& master_tighten_region,
    classification gt_classification,
    classification re_classification,
    ssize_t size_x,
    ssize_t size_y,
    ssize_t size_z,
    ssize_t num_points,
    int grow_amount
){
    std::vector<int32_t> re_mismatch_classes;

    for( ssize_t j = 0; j < num_points; ++j ){
        if( gt_classification[j] != re_classification[j] ){
            if( !contains<int32_t>(re_mismatch_classes, re_classification[j]) ){
                re_mismatch_classes.push_back(re_classification[j]);
            }
        }
    }

    if( re_mismatch_classes.empty() ){
        return;
    }

    partition<T> re_partition = partition_from_classification<T>(re_classification, sf, num_points);

    for( auto class_ : re_mismatch_classes ){
        Region<T> grown_region;
        const Region<T>* selected_region;

        if( grow_amount > 0 ){
            grown_region = re_partition.at(class_).grow(grow_amount, size_x, size_y, size_z);
            selected_region = &grown_region;
        } else {
            selected_region = &re_partition.at(class_);
        }

        for( ssize_t i = 0; i < selected_region->size(); ++i ){
            master_tighten_region.add_point((*selected_region)[i]);
        }
    }
}

template<typename T>
void tighten_sf_by_mismatch(
    Region<T>& master_tighten_region,
    classification gt_classification,
    classification re_classification,
    ssize_t size_x,
    ssize_t size_y,
    ssize_t size_z,
    ssize_t num_points,
    int grow_amount
){
    Region<T> mismatch_region;
    for( ssize_t j = 0; j < num_points; ++j ){
        if( gt_classification[j] != re_classification[j] ){
            mismatch_region.add_point(j);
        }
    }

    Region<T> grown_region;
    const Region<T>* selected_region = &mismatch_region;
    if( grow_amount > 0 ){
        grown_region = mismatch_region.grow(grow_amount, size_x, size_y, size_z);
        selected_region = &grown_region;
    }

    for( ssize_t i = 0; i < selected_region->size(); ++i ){
        master_tighten_region.add_point((*selected_region)[i]);
    }
}

template<typename T>
void tighten_region_bounds(
    sf_bounds<T>& bounds,
    const Region<T>& tighten_region,
    int iteration
){
    if( iteration > MAX_ITERATIONS ){
        for( ssize_t i = 0; i < tighten_region.size(); ++i ){
            ssize_t point = tighten_region[i];
            bounds[point].ub = 0;
            bounds[point].lb = 0;
        }
    } else {
        for( ssize_t i = 0; i < tighten_region.size(); ++i ){
            ssize_t point = tighten_region[i];
            bounds[point].ub /= 2;
            bounds[point].lb /= 2;
        }
    }
}

template<typename T>
void tighten_sf(
    const T* sf,
    sf_bounds<T>& bounds,
    const classification_list& gt_classifications,
    const classification_list& re_classifications,
    const compression_strategy_list& compression_strategies,
    ssize_t num_points,
    ssize_t size_x,
    ssize_t size_y,
    ssize_t size_z,
    int iteration
){

    ssize_t grow_amount = iteration / 3;
    Region<T> master_tighten_region;

    for( size_t i = 0; i < gt_classifications.size(); ++i ){
        classification gt_classification = gt_classifications[i];
        classification re_classification = re_classifications[i];

        if( compression_strategies[i].region_based_tightening ){
            tighten_sf_by_partitions<T>(sf, master_tighten_region, gt_classification, re_classification, size_x, size_y, size_z, num_points, grow_amount );
        } else {
            tighten_sf_by_mismatch<T>(master_tighten_region, gt_classification, re_classification, size_x, size_y, size_z, num_points, grow_amount);
        }
    }

    tighten_region_bounds<T>(bounds, master_tighten_region, iteration);
}

template<typename T>
py::bytes compression_output_to_bytes(
    const T* sf,
    const std::vector<int32_t>& quantization_numbers,
    ssize_t num_points
){
    std::vector<std::uint8_t> huffman_bytes = huffman_encode(quantization_numbers);

    std::vector<T> lossless_storage;

    for( ssize_t i = 0; i < num_points; ++i ){
        if( quantization_numbers[i] == LOSSLESS_SYMBOL ){
            lossless_storage.push_back(sf[i]);
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

template<typename T>
py::bytes compress_sf(
    const py::array_t<T, py::array::f_style | py::array::forcecast>& gt_sf_py,
    const py::array_t<T, py::array::f_style | py::array::forcecast>& de_sf_py,
    T xi,
    const std::vector<py::object>& functions,
    const std::vector<py::object>& param_dicts,
    const compression_strategy_list& compression_strategies
){
    if( functions.size() != compression_strategies.size() ){
        throw std::runtime_error("compression strategy size must match number of feature functions");
    }
    if( functions.size() != param_dicts.size() ){
        throw std::runtime_error("feature parameter dict count must match number of feature functions");
    }

    ssize_t ndim = gt_sf_py.ndim();
    if( ndim != 2 && ndim != 3 ){
        throw std::runtime_error("Feature preserving compressor only supports 2D or 3D scalar fields.");
    }

    ssize_t size_x = gt_sf_py.shape(0);
    ssize_t size_y = gt_sf_py.shape(1);
    ssize_t size_z = (ndim == 3) ? gt_sf_py.shape(2) : 1;
    ssize_t num_points = size_x * size_y * size_z;

    std::vector<py::array> gt_f = {gt_sf_py};

    py::buffer_info gt_info = gt_sf_py.request();
    py::buffer_info de_info = de_sf_py.request();

    const T* gt_sf = static_cast<T*>(gt_info.ptr);
    const T* de_sf = static_cast<T*>(de_info.ptr);

    classification_list_py gt_classifications_py = get_classifications_py(gt_f, functions, param_dicts);
    classification_list gt_classifications = classifications_from_py(gt_classifications_py);
    if( gt_classifications.size() != compression_strategies.size() ){
        throw std::runtime_error("classification count must match compression strategy count");
    }

    std::vector<bool> needs_partitions;
    needs_partitions.reserve(compression_strategies.size());
    bool any_region_based_ublb = false;
    bool any_preserve_mins_and_maxes = false;
    bool any_most_accurate_quantization = false;

    for( const auto& strategy : compression_strategies ){
        needs_partitions.push_back(strategy.region_based_ublb || strategy.preserve_mins_and_maxes);
        any_region_based_ublb = any_region_based_ublb || strategy.region_based_ublb;
        any_preserve_mins_and_maxes = any_preserve_mins_and_maxes || strategy.preserve_mins_and_maxes;
        any_most_accurate_quantization = any_most_accurate_quantization || strategy.most_accurate_quantization;
    }

    optional_partition_list<T> gt_partitions;
    if( any_region_based_ublb || any_preserve_mins_and_maxes ){
        gt_partitions = partition_list_from_classification_list<T>(gt_classifications, gt_sf, num_points, needs_partitions);
    } else {
        gt_partitions = optional_partition_list<T>(gt_classifications.size(), std::nullopt);
    }

    sf_bounds<T> bounds = get_initial_sf_bounds<T>(gt_sf, xi, gt_partitions, compression_strategies, num_points, any_region_based_ublb, any_preserve_mins_and_maxes);

    // I can't spin this into a function due to lifetime issues :(
    sf_quantization_result<T> quantization = quantize_sf<T>(gt_sf, de_sf, xi, bounds, num_points, any_most_accurate_quantization);
    py::array re_sf_view;
    if( ndim == 3 ){
        re_sf_view = make_py_array(quantization.sf.data(), size_x, size_y, size_z);
    } else {
        const ssize_t item = static_cast<ssize_t>(sizeof(T));
        std::array<ssize_t, 2> shape = {size_x, size_y};
        std::array<ssize_t, 2> strides = {item, item * size_x};
        re_sf_view = py::array_t<T>(shape, strides, quantization.sf.data(), py::none());
    }
    std::vector<py::array> re_f = { re_sf_view };
    classification_list_py re_classifications_py = get_classifications_py(re_f, functions, param_dicts);
    classification_list re_classifications = classifications_from_py(re_classifications_py);

    int num_mismatch = 0;
    for( ssize_t i = 0; i < num_points; ++i ){
        if(gt_classifications[0][i] != re_classifications[0][i]){
            ++num_mismatch;
        }
    }
    std::cout << num_mismatch << " / " << num_points << std::endl;

    int iteration = 0;
    while( !classifications_match( gt_classifications, re_classifications, num_points ) ){

        std::cout << "tightening iteration " << iteration << std::endl;
        tighten_sf(gt_sf, bounds, gt_classifications, re_classifications, compression_strategies, num_points, size_x, size_y, size_z, iteration);

        quantization = quantize_sf<T>(gt_sf, de_sf, xi, bounds, num_points, any_most_accurate_quantization);
        if( ndim == 3 ){
            re_sf_view = make_py_array(quantization.sf.data(), size_x, size_y, size_z);
        } else {
            const ssize_t item = static_cast<ssize_t>(sizeof(T));
            std::array<ssize_t, 2> shape = {size_x, size_y};
            std::array<ssize_t, 2> strides = {item, item * size_x};
            re_sf_view = py::array_t<T>(shape, strides, quantization.sf.data(), py::none());
        }
        re_f = { re_sf_view };
        re_classifications_py = get_classifications_py(re_f, functions, param_dicts);
        re_classifications = classifications_from_py(re_classifications_py);

        int num_mismatch = 0;
        for( ssize_t i = 0; i < num_points; ++i ){
            if(gt_classifications[0][i] != re_classifications[0][i]){
                ++num_mismatch;
            }
        }
        std::cout << num_mismatch << " / " << num_points << std::endl;

        iteration += 1;
    }

    return compression_output_to_bytes(gt_sf, quantization.quantization_numbers, num_points );

}

template<typename T>
struct compression_output{
    std::vector<int32_t> quantization_numbers;
    std::vector<T> lossless_storage;
};

template<typename T>
compression_output<T> bytes_to_compression_output(
    const py::bytes& compressed_bytes
){
    std::string in = compressed_bytes;

    if( in.size() < static_cast<ssize_t>(sizeof(std::uint64_t)) ){
        throw std::runtime_error("Invalid compression output: missing Huffman size header.");
    }

    std::size_t off = 0;
    std::uint64_t num_huffman_bytes = 0;
    std::memcpy(&num_huffman_bytes, in.data() + off, sizeof(num_huffman_bytes));
    off += sizeof(num_huffman_bytes);

    if( num_huffman_bytes > static_cast<std::uint64_t>(in.size() - off) ){
        throw std::runtime_error("Invalid compression output: Huffman payload exceeds input size.");
    }

    std::vector<std::uint8_t> huffman_bytes(static_cast<std::size_t>(num_huffman_bytes));
    if( num_huffman_bytes > 0 ){
        std::memcpy(huffman_bytes.data(), in.data() + off, static_cast<std::size_t>(num_huffman_bytes));
        off += static_cast<std::size_t>(num_huffman_bytes);
    }

    compression_output<T> output;
    output.quantization_numbers = huffman_decode(huffman_bytes);

    const std::size_t remaining = static_cast<std::size_t>(in.size()) - off;
    if( remaining % sizeof(T) != 0 ){
        throw std::runtime_error("Invalid compression output: lossless payload size is not aligned to element size.");
    }

    const std::size_t num_lossless = remaining / sizeof(T);
    output.lossless_storage.resize(num_lossless);
    if( num_lossless > 0 ){
        std::memcpy(output.lossless_storage.data(), in.data() + off, remaining);
    }

    return output;
}

template<typename T>
py::array_t<T> decompress_sf(
    const py::array_t<T, py::array::f_style | py::array::forcecast>& de_sf_py,
    py::bytes compressed_bytes,
    T xi
){

    compression_output<T> output = bytes_to_compression_output<T>(compressed_bytes);

    ssize_t ndim = de_sf_py.ndim();
    if( ndim != 2 && ndim != 3 ){
        throw std::runtime_error("Feature preserving decompressor only supports 2D or 3D scalar fields.");
    }

    ssize_t size_x = de_sf_py.shape(0);
    ssize_t size_y = de_sf_py.shape(1);
    ssize_t size_z = (ndim == 3) ? de_sf_py.shape(2) : 1;
    ssize_t num_points = size_x * size_y * size_z;

    py::buffer_info de_info = de_sf_py.request();
    const T* de_sf = static_cast<T*>(de_info.ptr);

    py::array_t<T, py::array::f_style> re_sf_py =
        (ndim == 3) ? py::array_t<T, py::array::f_style>({size_x, size_y, size_z})
                    : py::array_t<T, py::array::f_style>({size_x, size_y});
    T* re_sf = re_sf_py.mutable_data();

    ssize_t next_lossless = 0;

    T un_quantize_constant = get_un_quantize_constant(xi);

    for( ssize_t i = 0; i < num_points; ++i ){

        int32_t quantization_number = output.quantization_numbers[i];

        if( quantization_number == LOSSLESS_SYMBOL ){
            re_sf[i] = output.lossless_storage[next_lossless];
            ++next_lossless;
        } else {
            re_sf[i] = un_quantize(de_sf[i], quantization_number, un_quantize_constant );
        }

    }

    return re_sf_py;

}
