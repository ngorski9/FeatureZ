from paraview.simple import *

def load_ttk():

    import shutil
    import os
    
    paraview_install = shutil.which("paraview")
    paraview_install_folder = os.path.dirname(paraview_install)
    
    import paraview
    paraview_version = paraview.__version__
    cropped_version = paraview_version[:paraview_version.rindex(".")]

    possible_plugin_folders = ["plugins", f"lib/paraview-{cropped_version}/plugins", f"lib64/paraview-{cropped_version}/plugins"]

    found = False
    for folder in possible_plugin_folders:
        candidate_path = f"{paraview_install_folder}/{folder}/TopologyToolKit/TopologyToolKit.so"
        if os.path.isfile(candidate_path):
            LoadPlugin(candidate_path, False, globals())
            found = True
            break
    
    if not found:
        plugin_location = input("Cannot identify the location of TopologyToolKit.so. Please specify the location below (as an absolute path):\n")
        LoadPlugin(plugin_location, False, globals())

def reset():
    Disconnect()
    Connect()

try:
    TTKContourTree # see if TTK comes with ParaView automatically.
except:
    load_ttk()
