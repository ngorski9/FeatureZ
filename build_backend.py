"""Keep editable-install artifacts in build/ by default."""

from setuptools.build_meta import *  # noqa: F403
from setuptools.build_meta import build_editable as _build_editable
from setuptools.build_meta import get_requires_for_build_editable as _get_requires
from setuptools.build_meta import prepare_metadata_for_build_editable as _prepare_metadata


def _strict_mode(config_settings):
    settings = dict(config_settings or {})
    settings.setdefault("editable_mode", "strict")
    return settings


def get_requires_for_build_editable(config_settings=None):
    return _get_requires(_strict_mode(config_settings))


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return _prepare_metadata(metadata_directory, _strict_mode(config_settings))


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _build_editable(wheel_directory, _strict_mode(config_settings), metadata_directory)
