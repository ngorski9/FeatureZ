import os
import subprocess
from dulwich import porcelain

from .base_compressor import BaseCompressor, BaseCompressorAlreadyInstalledException
from .data import Data
from FeatureZ.installation_utils import cmake_build


def _create_zfp():
    def install_zfp(directory="."):
        zfp_dir = f"{directory}/ZFP"
        if os.path.isdir(zfp_dir):
            raise BaseCompressorAlreadyInstalledException()
        os.mkdir(zfp_dir)
        porcelain.clone("https://github.com/LLNL/zfp.git", f"{zfp_dir}/src")
        os.mkdir(f"{zfp_dir}/build")
        cmake_build("ZFP", zfp_dir, install=False)

    def _flags(dimensions, double_precision):
        dtype = "-d" if double_precision else "-f"
        if dimensions[2] == 1:
            return [dtype, "-2", str(dimensions[0]), str(dimensions[1])]
        return [dtype, "-3", *map(str, dimensions)]

    def _run(cmd, stage):
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ZFP {stage} failed with exit code {result.returncode}: "
                               f"{result.stderr.strip() or result.stdout.strip()}")

    def compress_with_zfp(binary: str, data: Data, compressed_filename, error_bound):
        scalar_range = data.get_range()
        aeb = error_bound * (scalar_range[1] - scalar_range[0])
        _run([binary, *_flags(data.dimensions, data.double_precision), "-a", str(aeb),
              "-i", data.get_raw_filename(), "-z", compressed_filename], "compression")
        return {"aeb": aeb, "dimensions": data.dimensions,
                "double precision": data.double_precision}

    def decompress_with_zfp(binary: str, compressed_filename: str,
                            decompressed_filename, metadata: dict):
        _run([binary, *_flags(metadata["dimensions"], metadata["double precision"]),
              "-a", str(metadata["aeb"]), "-z", compressed_filename,
              "-o", decompressed_filename], "decompression")
        return Data.from_raw(decompressed_filename, metadata["dimensions"],
                             metadata["double precision"])

    return BaseCompressor("./ZFP/build/bin/zfp", install_zfp,
                          compress_with_zfp, decompress_with_zfp)


ZFP = _create_zfp()
