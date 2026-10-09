import os
import cmake
import shlex
import subprocess
from dulwich import porcelain
import shutil

def silent_subprocess(cmd_string):

    cmd_list = cmd_string.split(" ")

    proc = subprocess.Popen(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False
    )

    exit_code = proc.wait()
    return exit_code == 0

def _run_cmake_step(cmake_bin, args, id=-1, step=""):
    cmd = [cmake_bin] + args

    if id == -1:
        return subprocess.run(cmd).returncode

    with open(_cmake_step_log_file(id, step), "w", encoding="utf-8") as outf:
        return subprocess.run(cmd, stdout=outf, stderr=subprocess.STDOUT).returncode

def _cmake_step_log_file(id, step):
    return f"{log_dir()}/task{id}_cmake_{step}.log"

def cmake_build(name, dir, cmake_flags="", build_flags="", install_flags="", install=True, id=-1):
    ninja_available = silent_subprocess("ninja --version")

    if id == -1:
        inspect_arg = ""
    else:
        inspect_arg = f"Inspect the log with {id} for more details."

    if ninja_available:
        ninja_flag = "-GNinja"
    else:
        ninja_flag = ""

    cmake_bin = os.path.join(cmake.CMAKE_BIN_DIR, "cmake")
    if os.name == "nt":
        cmake_bin += ".exe"

    configure_args = []
    if ninja_flag:
        configure_args.append(ninja_flag)
    configure_args += ["-S", f"{dir}/src", "-B", f"{dir}/build"]
    if cmake_flags:
        configure_args += shlex.split(cmake_flags)

    if 0 != _run_cmake_step(cmake_bin, configure_args, id=id, step="generate"):
        raise Exception(f"{name} cmake file generation failed. {inspect_arg}")

    build_args = ["--build", f"{dir}/build"]
    if build_flags:
        build_args += shlex.split(build_flags)

    if 0 != _run_cmake_step(cmake_bin, build_args, id=id, step="build"):
        raise Exception(f"{name} cmake --build failed. {inspect_arg}")

    if install:
        install_args = ["--install", f"{dir}/build"]
        if install_flags:
            install_args += shlex.split(install_flags)

        if 0 != _run_cmake_step(cmake_bin, install_args, id=id, step="install"):
            raise Exception(f"{name} cmake --install failed. {inspect_arg}")


def git_clone_and_cmake_build(name, repo, dir, cmake_flags="", build_flags="", install_flags="", install=True, recurse=False, id=-1):

    if os.path.isdir(dir):
        shutil.rmtree(dir)

    os.mkdir(dir)

    porcelain.clone(repo, f"{dir}/src", recurse_submodules=recurse)

    os.mkdir(f"{dir}/build")

    cmake_build(name, dir, cmake_flags, build_flags, install_flags, install, id=id)