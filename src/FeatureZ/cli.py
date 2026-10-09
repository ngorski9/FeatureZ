"""Command line entry points for FeatureZ compression."""

import argparse
import math
import os
import tempfile
import time

import numpy as np

from .FeatureZ_compressor import _base_compressor

FEATURES = (
    "contour-tree", "Morse-Smale-segmentation", "critical points",
    "isocontours", "percentiles", "local-order", "quartiles",
)
BASE_COMPRESSORS = ("SZ3", "ZFP", "SPERR", "HPEZ")


class _HelpOnErrorParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, f"{self.prog}: error: {message}\n")


def _parser(description):
    return _HelpOnErrorParser(description=description)


def _psnr(original, reconstructed):
    difference = original.astype(np.float64) - reconstructed.astype(np.float64)
    mse = float(np.mean(np.square(difference)))
    if mse == 0:
        return math.inf
    value_range = float(np.max(original) - np.min(original))
    if value_range == 0:
        return -math.inf
    return 20 * math.log10(value_range / math.sqrt(mse))


def _compress(args):
    from .FeatureZ_compressor import compress, decompress, _base_compressor
    from .base_compressors.data import Data

    started = time.perf_counter()

    is_vti = args.input.lower().endswith(".vti")
    if is_vti:
        if not args.array:
            raise ValueError("-array is required for .vti input")
        if args.dimensions or args.float_precision or args.double_precision:
            raise ValueError("Do not specify -3, -f, or -d for .vti input")
        data = Data.from_vti(args.input, args.array)
    else:
        if args.array:
            raise ValueError("-array is only valid for .vti input")
        if not args.dimensions or args.float_precision == args.double_precision:
            raise ValueError("Raw input requires -3 X Y Z and exactly one of -f or -d")
        data = Data.from_raw(args.input, tuple(args.dimensions), args.double_precision)

    if any(d <= 1 for d in data.dimensions):
        raise ValueError("FeatureZ requires 3D data with every dimension greater than one")
    if args.eb < 0:
        raise ValueError("-eb must be nonnegative")
    if "contour-tree" in args.features and args.epsilon is None:
        raise ValueError("-eps is required for contour-tree")
    if args.epsilon is not None and "contour-tree" not in args.features:
        raise ValueError("-eps requires contour-tree")
    if "isocontours" in args.features and not args.isovalues:
        raise ValueError("-isovalues is required for isocontours")
    if args.isovalues and "isocontours" not in args.features:
        raise ValueError("-isovalues requires isocontours")
    if not args.compressed and not args.output:
        raise ValueError("Specify -z, -o, or both")

    parameters = {}
    if args.epsilon is not None:
        parameters["ε"] = args.epsilon
    if args.isovalues:
        parameters["isovalues"] = args.isovalues

    with tempfile.TemporaryDirectory() as directory:
        compressed_filename = args.compressed or os.path.join(directory, "featurez.tar.xz")
        modified = compress(data, compressed_filename, args.eb, args.features, parameters,
                            _base_compressor(args.base_compressor), args.base_compressor_binary)
        elapsed = time.perf_counter() - started
        if args.output:
            decompression_started = time.perf_counter()
            decompress(compressed_filename, args.output, args.base_compressor_binary)
            decompression_elapsed = time.perf_counter() - decompression_started
        ratio = data.get_raw_file_size() / os.path.getsize(compressed_filename)
        print(f"Compression ratio: {ratio:.6g}")
        print(f"PSNR: {_psnr(data.numpy_array, modified):.6g} dB")
        print(f"Compression time: {elapsed:.6f} s")
        if args.output:
            print(f"Decompression time: {decompression_elapsed:.6f} s")

def _error(message):
    print(f"featurez-compress: {message}. Try featurez-compress -h.")
    exit(1)

def _validate_compression_args(args):
    if args.input is None:
        _error("You must specify an input file")

    is_vti = args.input[-3:] == "vti"

    if args.compressed is None and args.output is None:
        _error("You must either specify a compressed or decompressed filename")

    if is_vti:
        if args.dimensions is not None:
            _error("You cannot specify dimensions if compressing a vti file")

        if args.float_precision or args.double_precision:
            _error("You may not specify single or double precision if using a .vti file")

        if args.array is None:
            _error("You must specify an array name when using a .vti file")
    else:
        if args.dimensions is not None:
            _error("You must specify dimensions when using a raw binary file")

        if not args.float_precision and not args.double_precision:
            _error("You must specify single or double precision when using a raw binary file")

        if args.array is not None:
            _error("You must not specify an array name when using a vti file")

    if args.eb is None:
        _error("You must specify an error bound")

    if args.eb <= 0 or args.eb > 1:
        _error("Error bound must be between 0 and 1")

    if args.base_compressor is None:
        _error("You must specify a base compressor with -bc")
            
    base_compressor_object = _base_compressor(args.base_compressor)

    if args.base_compressor_binary is not None and not base_compressor_object.is_installed(args.base_compressor_binary):
        _error(f"There is no file located at your base compressor binary path {args.base_compressor_binary}")
    elif args.base_compressor_binary is None and not base_compressor_object.is_installed():
        _error(f"The base compressor {args.base_compressor} is not instaled in its default location, and you did not specify an alternative location.")

    if args.features is None or len(args.features) == 0:
        _error("You did not specify any features to preserve. Please specify 1-3 features")

    if "contour-tree" in args.features:
        
        if args.epsilon is None:
            _error("You indicated that you would like to preserve the contour tree, yet you did not provide an epsilon value")

        if args.epsilon < 0 or args.epsilon > 1:
            _error("Epsilon should be between 0 and 1")
    
    if "isocontours" in args.features and (args.isovalues is None or len(args.isovalues) == 0):
        _error("You must provide isovalues in order to preserve isocontours")


def compress_main(argv=None):
    parser = _parser("Compress a 3D scalar field with FeatureZ")
    parser.add_argument("-i", dest="input", help="File to compress. Either in raw binary or .vti format.")
    parser.add_argument("-z", dest="compressed", help="Compressed output file")
    parser.add_argument("-o", dest="output", help="Name of decompressed file (either raw or .vti format)")
    parser.add_argument("-3", dest="dimensions", nargs=3, type=int, metavar=("X", "Y", "Z"), help="If specifying a raw binary file, give its dimensions (Fortran order)")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("-f", dest="float_precision", action="store_true", help="If specifying a raw binary file, indicate that it is float32")
    precision.add_argument("-d", dest="double_precision", action="store_true", help="If specifying a raw binary file, indicate that it is float64")
    parser.add_argument("-array", help="If specifying a .vti file, what is the array name that you are compressing")
    parser.add_argument("-eb", type=float, help="Error bound as a fraction of the data range")
    parser.add_argument("-bc", dest="base_compressor", choices=BASE_COMPRESSORS, help="Base compressor that is augmented by FeatureZ")
    parser.add_argument("-bc-binary", dest="base_compressor_binary", help="(optional) location of the binary of the base compressor. "\
                                           "If not specified, FeatureZ will assume that it is in the location where it would go if you "\
                                           "installed it using featurez-install-base-compressor (such that the repo was cloned to your cwd).")
    parser.add_argument("-feat", dest="features", nargs="+", choices=FEATURES, help="which feature(s) do you want to preserve. You may list up to 3.")
    parser.add_argument("-eps", dest="epsilon", type=float, help="If preserving the contour tree, this indicates persistence threshold should be preserved. This should be "\
                                                                 "a fraction of the function range between 0 and 1 (e.g., 0.05).")
    parser.add_argument("-isovalues", nargs="+", type=float, help="If preserving isocontours, this indicates which isovalues would you like to use for the contours? "\
                                                                  "you may list any number of isovalues.")
    args = parser.parse_args(argv)

    _validate_compression_args(args)

    try:
        _compress(args)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"featurez-compress: {exc}. Try featurez-compress -h\n")


def decompress_main(argv=None):
    parser = _parser("Decompress a FeatureZ archive")
    parser.add_argument("-z", dest="compressed", required=True, help="Compressed file to decompress")
    parser.add_argument("-o", dest="output", required=True, help="Final decompressed filename")
    parser.add_argument("-array", help="Point array name for .vti output (if saving to a .vti file)")
    parser.add_argument("-bc-binary", dest="base_compressor_binary", help="(Optional) if the base compressor binary not located in the default location used by featurez-install-base-compressor "\
                                                                          "(which is where it would be git cloned into the cwd), specify the location for the binary")
    args = parser.parse_args(argv)
    from .FeatureZ_compressor import decompress
    started = time.perf_counter()
    data = decompress(args.compressed, args.output, args.base_compressor_binary,
                      vtk_array_name=args.array)
    elapsed = time.perf_counter() - started
    ratio = data.get_raw_file_size() / os.path.getsize(args.compressed)
    print(f"Compression ratio: {ratio:.6g}")
    print(f"Decompression time: {elapsed:.6f} s")


def install_base_compressor_main(argv=None):
    parser = _parser("Install a FeatureZ base compressor in the current directory")
    parser.add_argument("compressor", choices=BASE_COMPRESSORS)
    args = parser.parse_args(argv)
    _base_compressor(args.compressor).install(os.getcwd())
