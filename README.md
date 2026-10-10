# FeatureZ

Related Publication:
- Nathaniel Gorski, Xin Liang, Hanqi Guo, Bei Wang. FeatureZ: A General Framework for Feature-Preserving Compression Via Pointwise Bounds and Star Classification. IEEE TVCG. 2027.

[arxiv](https://arxiv.org/abs/2610.12371)

## Dependencies

- ParaView
- The Topology Toolkit (TTK)
- Note: FeatureZ has only been tested on Ubuntu Linux. It has been designed to load the TTK plugin as a .so. We suspect that FeatureZ will also work on MacOS, but not Windows. If a Windows user would like to use FeatureZ, we will happily make the adjustment, or review any related pull request.

## Installation

FeatureZ must be installed in a Python environment that has ParaView support. It will attempt to automatically detect the location of your Topology Toolkit binary. If it cannot, you will be prompted to enter it yourself.

To install, use:

```
pip install -e FeatureZ
```

## Usage

When the FeatureZ Python library is installed, it exposes three commands that can be used in the terminal, which are described below:

### Base Compressor Installation

FeatureZ works by augmenting an existing lossy compressor, called the *base compressor*. Currently, four base compressors are supported: ZFP, SZ3, SPERR, and HPEZ. FeatureZ provides the option to automatically install a base compressor on your machine. It will clone the git repo and build it with cmake. It uses git and cmake versions distributed from PyPI, so neither git nor cmake are dependencies. You may install a base compressor using:

```
featurez-install-base-compressor [compressor name]
```

This will clone the repo for the base compressor into the `cwd` and build it there. If you would like to provide your own installation, you may do so by instead using the `-bc-binary` flag during compression and decompression.

### Compression

FeatureZ supports the compression of 3D single and double precision scalar fields. They may be stored in `.vti` or raw binary format. To compress, use:

```
featurez-compress [options]
```

Use the `-h` flag for more information.

### Decompression

Decompress files compressed with FeatureZ using

```
featurez-decompress [options]
```