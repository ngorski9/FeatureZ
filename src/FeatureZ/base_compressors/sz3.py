import os
import subprocess
from dulwich import porcelain

from .base_compressor import BaseCompressor, BaseCompressorAlreadyInstalledException
from ..data import Data
from FeatureZ.installation_utils import cmake_build


def _create_sz3():
    def install_sz3(directory="."):
        sz3_dir = f"{directory}/SZ3"
        if os.path.isdir(sz3_dir):
            raise BaseCompressorAlreadyInstalledException()
        os.mkdir(sz3_dir)
        porcelain.clone("https://github.com/szcompressor/SZ3", f"{sz3_dir}/src")
        os.mkdir(f"{sz3_dir}/build")
        cmake_build("SZ3", sz3_dir, f"-DCMAKE_INSTALL_PREFIX:PATH={sz3_dir}/build", install=True)

    def _flags(dimensions, double_precision):
        dtype = "-d" if double_precision else "-f"
        if dimensions[2] == 1:
            return [dtype, "-2", str(dimensions[0]), str(dimensions[1])]
        return [dtype, "-3", *map(str, dimensions)]

    def _run(cmd, stage):
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"SZ3 {stage} failed with exit code {result.returncode}: "
                               f"{result.stderr.strip() or result.stdout.strip()}")

    def compress_with_sz3(binary: str, data: Data, compressed_filename, error_bound):
        scalar_range = data.get_range()
        aeb = error_bound * (scalar_range[1] - scalar_range[0])
        _run([binary, *_flags(data.dimensions, data.double_precision),
              "-i", data.get_raw_filename(), "-z", compressed_filename,
              "-M", "ABS", str(aeb)], "compression")
        return {"aeb": aeb, "dimensions": data.dimensions,
                "double precision": data.double_precision}

    def decompress_with_sz3(binary: str, compressed_filename: str,
                            decompressed_filename, metadata: dict):
        _run([binary, *_flags(metadata["dimensions"], metadata["double precision"]),
              "-z", compressed_filename, "-o", decompressed_filename,
              "-M", "ABS", str(metadata["aeb"])], "decompression")
        return Data.from_raw(decompressed_filename, metadata["dimensions"],
                             metadata["double precision"])

    return BaseCompressor("./SZ3/build/bin/sz3", install_sz3,
                          compress_with_sz3, decompress_with_sz3)


SZ3 = _create_sz3()
