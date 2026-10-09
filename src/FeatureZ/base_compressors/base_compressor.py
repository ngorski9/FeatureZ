from dataclasses import dataclass
import types
import os

from FeatureZ.installation_utils import silent_subprocess

class BaseCompressorAlreadyInstalledException(Exception):
    pass

@dataclass
class BaseCompressor():
    default_binary : str
    install : types.FunctionType
    compress : types.FunctionType
    decompress : types.FunctionType

    def is_installed(self, binary = None):
        if binary is None:
            binary = self.default_binary

        return os.path.isfile(binary) and silent_subprocess(binary)
