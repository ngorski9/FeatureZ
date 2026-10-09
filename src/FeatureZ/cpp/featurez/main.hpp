#pragma once

#include "include.hpp"
#include "compress.hpp"
#include "decompress.hpp"
#include "contour_tree.hpp"
#include "contours.hpp"

py::tuple featurez_compress_impl( const py::array& gt_data, 
                              const py::array& de_data, 
                              const std::optional<py::array>& ub,
                              const std::optional<py::array>& lb,
                              double xi, 
                              py::list compression_features_py,
                              bool linear_quantization,
                              bool ablation_ublb,
                              bool ablation_star);

py::array featurez_decompress_impl( const py::array& de_data, 
                                const py::bytes& adjust_bytes, 
                                double xi );

py::tuple contour_tree_ublb_impl( const py::array& jt_points_py, 
                                  const py::array& jt_segmentation_py, 
                                  const py::array& st_points_py, 
                                  const py::array& st_segmentation_py, 
                                  const py::array& sf_py, 
                                  double epsilon );

py::tuple contours_ublb_impl( const py::array& sf_py,
                              const py::list& contour_values_py );
