import os
import shutil
import subprocess
import tarfile
from dulwich import porcelain

from .base_compressor import BaseCompressorAlreadyInstalledException, BaseCompressor
from .data import Data
from FeatureZ.installation_utils import cmake_build


def _create_hpez():

    def _patch_hpez_source(hpez_dir: str):
        config_path = f"{hpez_dir}/src/include/QoZ/utils/Config.hpp"
        with open(config_path, "r", encoding="utf-8") as inf:
            config_text = inf.read()

        original_text = config_text

        if "#include <array>" not in config_text:
            config_text = config_text.replace(
                '#include <iostream>\n',
                '#include <array>\n#include <iostream>\n',
                1,
            )

        if "#include <cstdint>" not in config_text:
            config_text = config_text.replace(
                '#include <iostream>\n',
                '#include <cstdint>\n#include <iostream>\n',
                1,
            )

        if config_text != original_text:
            with open(config_path, "w", encoding="utf-8") as outf:
                outf.write(config_text)

    def install_hpez(directory = "."):
        hpez_dir = f"{directory}/HPEZ"
        if os.path.isdir(hpez_dir):
            raise BaseCompressorAlreadyInstalledException()

        os.mkdir(hpez_dir)

        porcelain.clone("https://github.com/JLiu-1/HPEZ-QoZ2.0", f"{hpez_dir}/src")

        _patch_hpez_source(hpez_dir)

        os.mkdir(f"{hpez_dir}/build")
        
        cmake_build(
            "HPEZ",
            hpez_dir,
            f"-DCMAKE_INSTALL_PREFIX:PATH={hpez_dir}/build",
            install=True,
        )

    def _dtype_flag(double_precision: bool):
        return "-d" if double_precision else "-f"

    def _size_flags(dimensions):
        return ["-3", str(dimensions[0]), str(dimensions[1]), str(dimensions[2])]

    def _run_hpez_command(cmd, stage):
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"HPEZ {stage} failed with exit code {result.returncode}: "
                               f"{result.stderr.strip() or result.stdout.strip()}")

    def compress_with_hpez(binary : str, data: Data, compressed_filename, error_bound):

        scalar_range = data.get_range()
        aeb = error_bound * (scalar_range[1] - scalar_range[0])

        metadata = {
            "arrays": {},
            "aeb": aeb,
            "dim": data.dimensions,
            "double precision": data.double_precision,
        }

        raw_filename = data.get_raw_filename()
        cmd = [
            binary,
            "-q",
            "4",
            _dtype_flag(data.double_precision),
            *_size_flags(data.dimensions),
            "-M",
            "ABS",
            str(aeb),
            "-i",
            raw_filename,
            "-z",
            compressed_filename
        ]

        _run_hpez_command(cmd, "compression")

        metadata["double precision"] = data.double_precision
        metadata["dimensions"] = data.dimensions
        metadata["aeb"] = aeb

        return metadata

    def decompress_with_hpez(binary : str, compressed_filename : str, decompressed_filename, metadata : dict):

        cmd = [
            binary,
            _dtype_flag(metadata["double precision"]),
            *_size_flags(metadata["dimensions"]),
            "-M",
            "ABS",
            str(metadata["aeb"]),
            "-z",
            compressed_filename,
            "-o",
            decompressed_filename
        ]

        _run_hpez_command(cmd, "decompression")

        data = Data.from_raw(
            decompressed_filename,
            metadata["dimensions"],
            metadata["double precision"]
        )

        return data

    return BaseCompressor(
        "./HPEZ/build/bin/hpez",
        install_hpez,
        compress_with_hpez,
        decompress_with_hpez,
    )

HPEZ = _create_hpez()
