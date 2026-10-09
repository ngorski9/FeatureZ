from enum import Enum, auto
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from pathlib import Path
import numpy as np
import os
import vtk


def _vti_to_numpy_array(vti_filename, array_name):
    if not os.path.isfile(vti_filename):
        raise FileNotFoundError(f"No such vtk file {vti_filename}")
    
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(vti_filename)
    reader.Update()

    image_data = reader.GetOutput()

    if image_data is None:
        raise FileNotFoundError(f"File {vti_filename} is not properly formatted vtk image data")

    if image_data.GetPointData() is None:
        raise Exception(f"File {vti_filename} does not have any point data")

    vtk_array = image_data.GetPointData().GetArray(array_name)

    if vtk_array is None:
        raise Exception(f"File {vti_filename} has no point array {array_name}")

    array_as_numpy = vtk_to_numpy(vtk_array)

    dimensions = [int(dimension) for dimension in image_data.GetDimensions()]

    array_as_numpy = np.asfortranarray(
        array_as_numpy.reshape(dimensions, order="F")
    )

    return array_as_numpy

def _numpy_array_to_vti(numpy_array, vti_filename, array_name):

    if numpy_array.ndim != 3:
        raise ValueError("Input array must be 3-dimensional")

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(*numpy_array.shape)

    vtk_array = numpy_to_vtk(
        numpy_array.ravel(order="F"),
        deep=True
    )
    vtk_array.SetName(array_name)

    image_data.GetPointData().AddArray(vtk_array)
    image_data.GetPointData().SetActiveScalars(array_name)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(vti_filename)
    writer.SetInputData(image_data)

    if writer.Write() != 1:
        raise IOError(f"Failed to write VTK file {vti_filename}")

def _raw_to_numpy_array(raw_filename, dimensions, double_precision):

    if double_precision:
        dtype = np.float64
    else:
        dtype = np.float32

    numpy_array = np.asfortranarray(
        np.fromfile(raw_filename, dtype).reshape(dimensions, order="F")
    )

    return numpy_array

def _numpy_array_to_raw(numpy_array, raw_filename):
    numpy_array.ravel(order="F").tofile(raw_filename)

class Data():

# constructor

    def __init__(self, numpy_array):
        if len(numpy_array.shape) != 3:
            raise ValueError("Data must be based on 3D numpy array")
        
        if numpy_array.dtype not in [np.float32, np.float64]:
            raise ValueError("Only floating point types are allowed")

        self._dimensions = numpy_array.shape
        self._double_precision = numpy_array.dtype == np.float64
        self._numpy_array = numpy_array

        self._vti_filename = None
        self._vti_array_name = None
        self._raw_filename = None
        self._originally_raw = False

# Initializers

    @classmethod
    def from_vti(cls, vti_filename, vti_array_name):
        numpy_array = _vti_to_numpy_array(vti_filename, vti_array_name)
        instance = cls(numpy_array)
        instance._vti_filename = vti_filename
        instance._vti_array_name = vti_array_name

        return instance
    
    @classmethod
    def from_raw(cls, raw_filename, dimensions, double_precision):
        numpy_array = _raw_to_numpy_array(raw_filename, dimensions, double_precision)
        instance = cls(numpy_array)
        instance._raw_filename = raw_filename
        instance._originally_raw = True

        return instance

# properties / getters etc

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def double_precision(self):
        return self._double_precision

    @property
    def numpy_array(self):
        return self._numpy_array

    def get_number_of_points(self):
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

    def get_raw_file_size(self):
        if self.double_precision:
            return self.get_number_of_points() * 8
        else:
            return self.get_number_of_points() * 4

    def get_range(self):
        return (np.min(self.numpy_array), np.max(self.numpy_array))

    def save_as_raw(self, raw_filename):
        _numpy_array_to_raw(self._numpy_array, raw_filename)

    def save_as_vti(self, vti_filename, vti_array_name = None):
        if vti_array_name is None:
            if self._vti_array_name is None:
                vti_array_name = "sf"
            else:
                vti_array_name = self._vti_array_name

        _numpy_array_to_vti(self._numpy_array, vti_filename, vti_array_name)

# temp file management

    def get_raw_filename(self, temp_filename = "./temp.raw"):
        if self._raw_filename is None:
            self._raw_filename = temp_filename
            self.save_as_raw(temp_filename)
        return self._raw_filename
        
    def remove_raw_file(self):
        if os.path.isfile(self._raw_filename):
            os.remove(self._raw_filename)
        self._raw_filename = None

    def clear_temp_raw_files(self):
        if not self._originally_raw and self._raw_filename is not None:

            self.remove_raw_file()
