#include "feature_preserving_compressor/dispatch.hpp"

py::bytes feature_preserving_compress_impl(
    const std::string& data_type,
    const py::dict& gt_data,
    const py::dict& de_data,
    const py::list& functions_list,
    const py::list& param_dicts_list,
    const py::list& compression_strategy_input,
    const double xi
) {
    std::vector<py::object> functions = list_to_object_vector(functions_list);
    std::vector<py::object> param_dicts = list_to_object_vector(param_dicts_list);
    std::vector<py::object> strategy_dicts = list_to_object_vector(compression_strategy_input);

    if( strategy_dicts.size() != functions.size() ){
        throw py::value_error("compression_strategy must provide one dictionary per feature function.");
    }

    compression_strategy_list strategies;
    strategies.reserve(strategy_dicts.size());
    for( const auto& strategy_obj : strategy_dicts ){
        py::dict strategy_dict = strategy_obj.cast<py::dict>();
        compression_strategy_per_function strategy;

        if( strategy_dict.contains("region based ublb") ){
            strategy.region_based_ublb = strategy_dict["region based ublb"].cast<bool>();
        }
        if( strategy_dict.contains("preserve mins and maxes") ){
            strategy.preserve_mins_and_maxes = strategy_dict["preserve mins and maxes"].cast<bool>();
        }
        if( strategy_dict.contains("region based tightening") ){
            strategy.region_based_tightening = strategy_dict["region based tightening"].cast<bool>();
        }
        if( strategy_dict.contains("most accurate quantization") ){
            strategy.most_accurate_quantization = strategy_dict["most accurate quantization"].cast<bool>();
        }

        strategies.push_back(strategy);
    }

    if( data_type == "Scalar" ){

        py::array gt_sf_py = gt_data["scalar"].cast<py::array>();
        py::array de_sf_py = de_data["scalar"].cast<py::array>();
        py::dtype dtype = gt_sf_py.dtype();

        if (dtype.is(py::dtype::of<float>())) {
            return compress_sf<float>(gt_sf_py, de_sf_py, xi, functions, param_dicts, strategies);
        } else if( dtype.is(py::dtype::of<double>()) ){
            return compress_sf<double>(gt_sf_py, de_sf_py, xi, functions, param_dicts, strategies);
        } else {
            throw py::type_error("Data must be float or double");
        }

    } else {
        throw py::type_error("Unsupported data type " + data_type);
    }

    py::bytes result;
    return result;
}

py::dict feature_preserving_decompress_impl(
    const std::string& data_type,
    const py::dict& de_data,
    const py::bytes& compressed_bytes,
    const double xi
){

    if( data_type == "Scalar" ){

        py::array de_sf_py = de_data["scalar"].cast<py::array>();
        py::dtype dtype = de_sf_py.dtype();
        py::dict out;

        if (dtype.is(py::dtype::of<float>())){
            out["scalar"] = decompress_sf<float>(de_sf_py, compressed_bytes, xi);
            return out;
        } else if ( dtype.is(py::dtype::of<double>()) ){
            out["scalar"] = decompress_sf<double>(de_sf_py, compressed_bytes, xi);
            return out;
        } else {
            throw py::type_error("Data must be float or double");
        }

    } else {
        throw py::type_error("Unsupported data type " + data_type);
    }

}
