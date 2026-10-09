#include "main.hpp"

template<size_t N>
py::tuple featurez_compress_impl_dispatch(
    const py::array& gt_data,
    const py::array& de_data,
    const std::optional<py::array>& ub,
    const std::optional<py::array>& lb,
    double xi, py::list compression_features_py,
    bool linear_quantization,
    bool ablation_ublb,
    bool ablation_star
){
    py::dtype dtype = gt_data.dtype();

    std::array<int, N> compression_features;

    for( int i = 0; i < N; ++i ){
        compression_features[i] = compression_features_py[i].cast<int>();
    }

    if (dtype.is(py::dtype::of<float>())){
        return featurez_compress_impl_template<float, N>(gt_data, de_data, ub, lb, xi, compression_features, linear_quantization, ablation_ublb, ablation_star);
    } else if (dtype.is(py::dtype::of<double>())){
        return featurez_compress_impl_template<double, N>(gt_data, de_data, ub, lb, xi, compression_features, linear_quantization, ablation_ublb, ablation_star);
    } else {
        throw py::value_error("Only float and double are supported");
    }
}

py::tuple featurez_compress_impl( 
    const py::array& gt_data, 
    const py::array& de_data, 
    const std::optional<py::array>& ub, 
    const std::optional<py::array>& lb, 
    double xi, 
    py::list compression_features_py,
    bool linear_quantization,
    bool ablation_ublb,
    bool ablation_star
){

    if( compression_features_py.size() == 1 ){
        return featurez_compress_impl_dispatch<1>(gt_data, de_data, ub, lb, xi, compression_features_py, linear_quantization, ablation_ublb, ablation_star);
    } else if( compression_features_py.size() == 2 ){
        return featurez_compress_impl_dispatch<2>(gt_data, de_data, ub, lb, xi, compression_features_py, linear_quantization, ablation_ublb, ablation_star);
    } else if( compression_features_py.size() == 3 ){
        return featurez_compress_impl_dispatch<3>(gt_data, de_data, ub, lb, xi, compression_features_py, linear_quantization, ablation_ublb, ablation_star);
    } else {
        throw py::value_error("Can only preserve up to 3 features");
    }

}

py::array featurez_decompress_impl( const py::array& de_data, const py::bytes& adjust_bytes, double xi ){
    
    py::dtype dtype = de_data.dtype();

    if (dtype.is(py::dtype::of<float>())){
        return featurez_decompress_impl_template<float>(de_data, adjust_bytes, xi);
    } else if (dtype.is(py::dtype::of<double>())){
        return featurez_decompress_impl_template<double>(de_data, adjust_bytes, xi);
    } else {
        throw py::value_error("Only float and double are supported");
    }

}

py::tuple contour_tree_ublb_impl( const py::array& jt_points_py, const py::array& jt_segmentation_py, const py::array& st_points_py, const py::array& st_segmentation_py, const py::array& sf_py, double epsilon ){

    py::dtype dtype = sf_py.dtype();

    if (dtype.is(py::dtype::of<float>())){
        return contour_tree_ublb_impl_template<float>(jt_points_py, jt_segmentation_py, st_points_py, st_segmentation_py, sf_py, epsilon);
    } else if (dtype.is(py::dtype::of<double>())){
        return contour_tree_ublb_impl_template<double>(jt_points_py, jt_segmentation_py, st_points_py, st_segmentation_py, sf_py, epsilon);
    } else {
        throw py::value_error("Only float and double are supported");
    }

}

py::tuple contours_ublb_impl( const py::array& sf_py, const py::list& contour_values_py ){

    py::dtype dtype = sf_py.dtype();

    if (dtype.is(py::dtype::of<float>())){
        return contours_ublb_impl_template<float>(sf_py.cast<py::array_t<float, py::array::f_style | py::array::forcecast>>(), contour_values_py);
    } else if (dtype.is(py::dtype::of<double>())){
        return contours_ublb_impl_template<double>(sf_py.cast<py::array_t<double, py::array::f_style | py::array::forcecast>>(), contour_values_py);
    } else {
        throw py::value_error("Only float and double are supported");
    }

}
