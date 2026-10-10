import os
import subprocess
from dulwich import porcelain

from .base_compressor import BaseCompressor, BaseCompressorAlreadyInstalledException
from ..data import Data
from FeatureZ.installation_utils import cmake_build


def _create_sperr():
    def install_sperr(directory="."):
        sperr_dir = f"{directory}/SPERR"
        if os.path.isdir(sperr_dir):
            raise BaseCompressorAlreadyInstalledException()
        os.mkdir(sperr_dir)
        porcelain.clone("https://github.com/NCAR/SPERR.git", f"{sperr_dir}/src")
        os.mkdir(f"{sperr_dir}/build")
        cmake_build("SPERR", sperr_dir, "-DBUILD_UNIT_TESTS=OFF",
                    "--target sperr2d sperr3d", install=False)

    def _utility(binary, dimensions):
        name = "sperr2d" if dimensions[2] == 1 else "sperr3d"
        return os.path.join(os.path.dirname(binary), name)

    def _run(cmd, stage):
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"SPERR {stage} failed with exit code {result.returncode}: "
                               f"{result.stderr.strip() or result.stdout.strip()}")

    def compress_with_sperr(binary: str, data: Data, compressed_filename, error_bound):
        scalar_range = data.get_range()
        pwe = error_bound * (scalar_range[1] - scalar_range[0])
        dimensions = data.dimensions[:2] if data.dimensions[2] == 1 else data.dimensions
        _run([_utility(binary, data.dimensions), "-c", "--ftype",
              "64" if data.double_precision else "32", "--dims", *map(str, dimensions),
              "--pwe", str(pwe), "--bitstream", compressed_filename,
              data.get_raw_filename()], "compression")
        return {"pwe": pwe, "dimensions": data.dimensions,
                "double precision": data.double_precision}

    def decompress_with_sperr(binary: str, compressed_filename: str,
                              decompressed_filename, metadata: dict):
        _run([_utility(binary, metadata["dimensions"]), "-d",
              "--decomp_d" if metadata["double precision"] else "--decomp_f",
              decompressed_filename, compressed_filename], "decompression")
        return Data.from_raw(decompressed_filename, metadata["dimensions"],
                             metadata["double precision"])

    return BaseCompressor("./SPERR/build/bin/sperr2d", install_sperr,
                          compress_with_sperr, decompress_with_sperr)


SPERR = _create_sperr()
