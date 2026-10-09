"""Preserve one scalar-field feature on top of a base compressor."""

import io
import json
import os
import tarfile
import tempfile

import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import vtk

from .base_compressors.data import Data
from .base_compressors.base_compressor import BaseCompressor
from .featurez import _contours_ublb, _contour_tree_ublb, _featurez_compress, _featurez_decompress

from . import ttk
from paraview.simple import Calculator, OutputPort, TrivialProducer, servermanager

_BASE_COMPRESSORS = {
    "HPEZ": ("hpez", "HPEZ"),
    "SZ3": ("sz3", "SZ3"),
    "ZFP": ("zfp", "ZFP"),
    "SPERR": ("sperr", "SPERR"),
}


def _base_compressor(name):
    if name not in _BASE_COMPRESSORS:
        raise ValueError(f"Unsupported base compressor in archive: {name}")
    from importlib import import_module

    module, attribute = _BASE_COMPRESSORS[name]
    return getattr(import_module(f".base_compressors.{module}", __package__), attribute)


def _base_name(compressor):
    for name in _BASE_COMPRESSORS:
        if compressor is _base_compressor(name):
            return name
    raise ValueError("The base compressor must be HPEZ, SZ3, ZFP, or SPERR")


def _compute_tree_and_bounds(data: Data, epsilon=0):
    """Build merge trees and compute bounds using the original TTK arc IDs."""

    numpy_array = data.numpy_array
    dims = numpy_array.shape

    image = vtk.vtkImageData()
    image.SetDimensions(*dims)
    values = numpy_to_vtk(numpy_array.ravel(order="F"), deep=True)
    values.SetName("scalar")
    image.GetPointData().SetScalars(values)
    arr = TrivialProducer()
    arr.GetClientSideObject().SetOutput(image)
    
    jt_pv = ttk.TTKMergeTree(Input=arr)
    jt_pv.ScalarField = ['POINTS', 'scalar']
    jt_pv.InputOffsetField = ['POINTS', 'scalar']
    jt_pv.DebugLevel = 0
    jt_pv.Backend = 'ExTreeM (IEEE TVCG 2023)'
    jt_pv.TreeType = 'Join Tree'

    jt_points_pv = jt_pv
    jt_tree_pv = Calculator(Input=OutputPort(jt_pv,1))
    jt_segmentation_pv = Calculator(Input=OutputPort(jt_pv,2))

    st_pv = ttk.TTKMergeTree(Input=arr)
    st_pv.ScalarField = ['POINTS', 'scalar']
    st_pv.InputOffsetField = ['POINTS', 'scalar']
    st_pv.DebugLevel = 0
    st_pv.Backend = 'ExTreeM (IEEE TVCG 2023)'
    st_pv.TreeType = 'Split Tree'

    st_points_pv = st_pv
    st_tree_pv = Calculator(Input=OutputPort(st_pv,1))
    st_segmentation_pv = Calculator(Input=OutputPort(st_pv,2))

    jt_points_vtk = servermanager.Fetch(jt_points_pv)
    jt_tree_vtk = servermanager.Fetch(jt_tree_pv)
    jt_segmentation_vtk = servermanager.Fetch(jt_segmentation_pv)

    st_points_vtk = servermanager.Fetch(st_points_pv)
    st_tree_vtk = servermanager.Fetch(st_tree_pv)
    st_segmentation_vtk = servermanager.Fetch(st_segmentation_pv)

    trees_numpy = []
    for vtk_tree, vtk_points, peak_critical_type in (
        (jt_tree_vtk, jt_points_vtk, 3),
        (st_tree_vtk, st_points_vtk, 0),
    ):
        num_cells = vtk_tree.GetNumberOfCells()

        out = np.zeros((num_cells+1, 2), dtype=np.int32, order="F")

        cell_data = vtk_tree.GetCellData()
        up_nodes = cell_data.GetArray("upNodeId")
        down_nodes = cell_data.GetArray("downNodeId")

        num_duplicates = 0

        next_row = 0
        for i in range(num_cells):
            up_node = up_nodes.GetTuple1(i)
            down_node = down_nodes.GetTuple1(i)

            if up_node == down_node:
                num_duplicates += 1
            else:
                out[next_row,0] = down_node
                out[next_row,1] = up_node
                next_row += 1

        out = out[:num_cells + 1 - num_duplicates, :]

        # find the root
        critical_types = vtk_points.GetPointData().GetArray("CriticalType")
        num_points = vtk_points.GetNumberOfPoints()
        for i in range(num_points):
            if critical_types.GetTuple1(i) == peak_critical_type:
                coords = vtk_points.GetPoint(i)
                index = coords[0] + coords[1]*dims[0]
                if len(coords) == 3:
                    index += coords[2]*dims[0]*dims[1]

                out[num_cells - num_duplicates,0] = index
                out[num_cells - num_duplicates,1] = -1
        trees_numpy.append(out)
    jt_numpy, st_numpy = trees_numpy

    jt_segmentation_array = jt_segmentation_vtk.GetPointData().GetArray("SegmentationId")
    jt_segmentation_numpy = vtk_to_numpy( jt_segmentation_array ).reshape(dims, order="F")

    st_segmentation_array = st_segmentation_vtk.GetPointData().GetArray("SegmentationId")
    st_segmentation_numpy = vtk_to_numpy( st_segmentation_array ).reshape(dims, order="F")

    if epsilon > 0:
        min_ = np.min(numpy_array)
        max_ = np.max(numpy_array)
        epsilon *= (max_ - min_)

    ttk.reset()

    return _contour_tree_ublb(
        jt_numpy,
        jt_segmentation_numpy,
        st_numpy,
        st_segmentation_numpy,
        numpy_array,
        epsilon,
    )


def _feature_bounds(data, feature_name, params, eb):
    scalar = np.asfortranarray(data.numpy_array)
    if feature_name in ("critical points", "local-order", "Morse-Smale-segmentation"):
        return None, None, {"critical points": 0, "local-order": 1,
                            "Morse-Smale-segmentation": 2}[feature_name]
    if feature_name == "contour-tree":
        epsilon = params.get("ε", 0)
        if epsilon < 0:
            raise ValueError("Contour tree epsilon must be nonnegative")
        if epsilon == 0:
            _, upper, lower = _compute_tree_and_bounds(data, 0)
            return upper, lower, 3
        if epsilon < 2 * eb:
            raise ValueError("Contour tree epsilon must be at least twice eb")
        _, upper, lower = _compute_tree_and_bounds(data, epsilon - 2 * eb)
        return upper, lower, -1
    if feature_name == "quartiles":
        contours = np.quantile(scalar, [0.25, 0.5, 0.75]).tolist()
    elif feature_name == "percentiles":
        contours = np.quantile(scalar, np.arange(1, 100) / 100).tolist()
    elif feature_name == "isocontours":
        contours = params["isovalues"]
    else:
        raise ValueError(f"Unsupported feature: {feature_name}")
    upper, lower = _contours_ublb(scalar, list(contours))
    return upper, lower, -1


def _archive_member(tar, name):
    member = tar.getmember(name)
    source = tar.extractfile(member)
    if source is None:
        raise ValueError(f"Missing archive member: {name}")
    return source.read()


def compress(data: Data, compressed_filename, eb: float, feature_name: str,
             feature_parameters: dict, base_compressor: BaseCompressor,
             base_compressor_binary=None) -> None:
    """Write a self-contained compressed archive to compressed_filename."""

    if not isinstance(data, Data):
        raise TypeError("data must be a Data object")
    if not isinstance(feature_parameters, dict):
        raise TypeError("feature_parameters must be a dictionary")
    if eb < 0:
        raise ValueError("eb must be nonnegative")
    name = _base_name(base_compressor)
    binary = base_compressor_binary or base_compressor.default_binary
    feature_names = [feature_name] if isinstance(feature_name, str) else list(feature_name)
    if not 1 <= len(feature_names) <= 3 or len(set(feature_names)) != len(feature_names):
        raise ValueError("Specify one to three distinct features")
    upper = lower = None
    feature_codes = []
    for requested_feature in feature_names:
        next_upper, next_lower, code = _feature_bounds(data, requested_feature, feature_parameters, eb)
        feature_codes.append(code)
        if next_upper is not None:
            upper = next_upper if upper is None else np.minimum(upper, next_upper)
            lower = next_lower if lower is None else np.maximum(lower, next_lower)
    scalar = np.asfortranarray(data.numpy_array)
    scalar_range = data.get_range()
    aeb = float(eb * (scalar_range[1] - scalar_range[0]))

    with tempfile.TemporaryDirectory() as directory:
        base_file = os.path.join(directory, "base.cmp")
        raw_file = os.path.join(directory, "base.raw")
        created_raw = data._raw_filename is None
        if created_raw:
            data.get_raw_filename(os.path.join(directory, "input.raw"))
        base_metadata = base_compressor.compress(binary, data, base_file, eb)
        base_data = base_compressor.decompress(binary, base_file, raw_file, base_metadata)
        changes, modified_sf = _featurez_compress(scalar, np.asfortranarray(base_data.numpy_array),
                                     upper, lower, aeb, feature_codes)
        metadata = {"base compressor": name, "base metadata": base_metadata, "aeb": aeb,
                    "array name": data._vti_array_name}
        encoded_metadata = json.dumps(metadata).encode("utf-8")
        with open(base_file, "rb") as source:
            base_bytes = source.read()
        with tarfile.open(compressed_filename, mode="w:xz") as archive:
            for member_name, contents in (("metadata.json", encoded_metadata),
                                          ("base.cmp", base_bytes),
                                          ("changes.bin", changes)):
                info = tarfile.TarInfo(member_name)
                info.size = len(contents)
                archive.addfile(info, io.BytesIO(contents))
        if created_raw:
            data.clear_temp_raw_files()
    
        return modified_sf


def decompress(compressed_filename, decompressed_filename, base_compressor_binary=None, vtk_array_name=None) -> Data:
    """Restore an archive to a raw file and return its Data object."""

    with tarfile.open(compressed_filename, "r:xz") as archive:
        metadata = json.loads(_archive_member(archive, "metadata.json"))
        base_bytes = _archive_member(archive, "base.cmp")
        changes = _archive_member(archive, "changes.bin")

    compressor = _base_compressor(metadata["base compressor"])
    binary = base_compressor_binary or compressor.default_binary
    with tempfile.TemporaryDirectory() as directory:
        base_file = os.path.join(directory, "base.cmp")
        raw_file = os.path.join(directory, "base.raw")
        with open(base_file, "wb") as output:
            output.write(base_bytes)
        base_data = compressor.decompress(binary, base_file, raw_file,
                                          metadata["base metadata"])
        result = _featurez_decompress(np.asfortranarray(base_data.numpy_array),
                                  changes, metadata["aeb"])
        data = Data(np.asfortranarray(result))

        if str(decompressed_filename).lower().endswith(".vti"):
            data.save_as_vti(decompressed_filename, vtk_array_name or metadata.get("array name") or "sf")
        else:
            data.save_as_raw(decompressed_filename)
        
        return data
