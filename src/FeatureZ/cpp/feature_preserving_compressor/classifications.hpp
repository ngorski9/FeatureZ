#pragma once

#include "feature_preserving_compressor/include.hpp"
#include "feature_preserving_compressor/region.hpp"
#include "common/pybind_conversions.hpp"
#include <optional>
#include <stdexcept>

using classification = const int32_t*;
using classification_list = std::vector<classification>;
using classification_list_py = std::vector<py::array>;

template<typename T>
using partition = std::unordered_map<int32_t,Region<T>>;

template<typename T>
using partition_list = std::vector<partition<T>>;

template<typename T>
using optional_partition_list = std::vector<std::optional<partition<T>>>;

classification_list_py get_classifications_py(
    const std::vector<py::array>& f,
    const std::vector<py::object>& functions,
    const std::vector<py::object>& param_dicts
);

// not combined with the previous function due to lifetime issues.
classification_list classifications_from_py(const classification_list_py& classifications_py );

bool classifications_match(
    const classification_list& first,
    const classification_list& second,
    ssize_t num_points
);

template<typename T>
partition<T> partition_from_classification(
    classification classification,
    const T* sf,
    ssize_t num_points
){

    partition<T> p;

    for( ssize_t i = 0; i < num_points; ++i ){

        int32_t point_class = classification[i];

        if( !p.count(point_class) ){
            p[point_class] = Region<T>(sf);
        }

        p[point_class].add_point(i);

    }

    return p;
}

template<typename T>
optional_partition_list<T> partition_list_from_classification_list(
    const classification_list& classifications,
    const T* sf,
    ssize_t num_points,
    const std::vector<bool>& compute_partition
){
    if( classifications.size() != compute_partition.size() ){
        throw std::runtime_error("compute_partition list must match number of classifications");
    }

    optional_partition_list<T> p;
    p.reserve(classifications.size());

    for( size_t i = 0; i < classifications.size(); ++i ){
        if( compute_partition[i] ){
            p.push_back(partition_from_classification<T>(classifications[i], sf, num_points));
        } else {
            p.push_back(std::nullopt);
        }
    }

    return p;

}
