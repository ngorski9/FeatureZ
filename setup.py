import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


class BuildExtInBuildDirectory(build_ext):
    def finalize_options(self):
        super().finalize_options()
        if self.editable_mode:
            self.inplace = False

if sys.platform == "win32":
    compile_args = ["/O2", "/w"]
else:
    compile_args = ["-O3", "-w"]

ext_modules = [
    Pybind11Extension(
        "FeatureZ.feature_preserving_compressor",
        [
            "src/FeatureZ/cpp/pybind/feature_preserving_compressor_pybind.cpp",
            "src/FeatureZ/cpp/feature_preserving_compressor/classifications.cpp",
            "src/FeatureZ/cpp/feature_preserving_compressor/dispatch.cpp",
            "src/FeatureZ/cpp/common/huffman.cpp",
        ],
        include_dirs=["src/FeatureZ/cpp"],
        cxx_std=17,
        extra_compile_args=compile_args,
    ),
    Pybind11Extension(
        "FeatureZ.featurez",[
            "src/FeatureZ/cpp/pybind/featurez.cpp",
            "src/FeatureZ/cpp/featurez/main.cpp",
            "src/FeatureZ/cpp/common/huffman.cpp"
        ],
        include_dirs=["src/FeatureZ/cpp"],
        cxx_std=17,
        extra_compile_args=compile_args,
    ),    
    Pybind11Extension(
        "FeatureZ.tensor_topology",
        [
            "src/FeatureZ/cpp/pybind/sf_pybind.cpp",
        ],
        include_dirs=["src/FeatureZ/cpp"],
        cxx_std=17,
        extra_compile_args=compile_args,
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtInBuildDirectory},
)
