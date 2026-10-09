# FeatureZ

For development, run `python -m pip install -e .` from this directory. Editable
installs keep compiled extensions under `build/`, leaving `src/FeatureZ` free of
generated `.so` files. Reinstall after changing C++ sources to rebuild them.

FeatureZ uses the VTK Python modules supplied with ParaView. Install ParaView
and make its Python modules available to the environment used for FeatureZ
(for example, create a virtual environment with `--system-site-packages` when
using a system ParaView installation). Do not install the separate PyPI `vtk`
wheel in that environment: it can shadow ParaView's matching VTK modules and
make `import paraview` fail with `_paraview_modules_static` missing. If that
wheel is already installed, remove it with `python -m pip uninstall vtk`.
