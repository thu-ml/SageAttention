"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import subprocess
import sys
import warnings
from typing import Any

from setuptools import setup


def get_extra_args() -> dict[str, Any]:
    # Bypass detection unless we're building the extensions
    build_commands = {"bdist_wheel", "build_ext", "develop", "install"}
    if not build_commands.intersection(sys.argv):
        return {}

    from packaging.version import parse, Version

    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    except ImportError as exc:
        raise ImportError(
            "Failed to import 'torch' which is required to build this package. "
            "Ensure that 'torch' is installed and visible to the Python interpreter, "
            "and that you are using the '--no-build-isolation' flag when installing "
            "this package with 'pip'."
        ) from exc

    HAS_SM80 = False
    HAS_SM86 = False
    HAS_SM89 = False
    HAS_SM90 = False
    HAS_SM120 = False

    # Compiler flags.
    CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
    NVCC_FLAGS = [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--use_fast_math",
        "--threads=8",
        "-Xptxas=-v",
        "-diag-suppress=174", # suppress the specific warning
    ]

    ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
    NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

    if CUDA_HOME is None:
        raise RuntimeError(
            "Cannot find a CUDA installation. "
            "CUDA must be available to build the package."
        )

    # Iterate over all GPUs on the current machine. Also you can modify this part to specify the architecture if you want to build for specific GPU architectures.
    compute_capabilities = set()
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 8:
            warnings.warn(f"skipping GPU {i} with compute capability {major}.{minor}")
            continue
        compute_capabilities.add(f"{major}.{minor}")

    # Get the CUDA version from nvcc.
    # Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    nvcc_output = subprocess.check_output([CUDA_HOME + "/bin/nvcc", "-V"], text=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    if not compute_capabilities:
        raise RuntimeError("No GPUs found. Please specify the target GPU architectures or build on a machine with GPUs.")
    else:
        print(f"Detect GPUs with compute capabilities: {compute_capabilities}")

    # Validate the NVCC CUDA version.
    if nvcc_cuda_version < Version("12.0"):
        raise RuntimeError("CUDA 12.0 or higher is required to build the package.")
    if nvcc_cuda_version < Version("12.4") and any(cc.startswith("8.9") for cc in compute_capabilities):
        raise RuntimeError(
            "CUDA 12.4 or higher is required for compute capability 8.9.")
    if nvcc_cuda_version < Version("12.3") and any(cc.startswith("9.0") for cc in compute_capabilities):
        raise RuntimeError(
            "CUDA 12.3 or higher is required for compute capability 9.0.")
    if nvcc_cuda_version < Version("12.8") and any(cc.startswith("12.0") for cc in compute_capabilities):
        raise RuntimeError(
            "CUDA 12.8 or higher is required for compute capability 12.0.")

    # Add target compute capabilities to NVCC flags.
    for capability in compute_capabilities:
        if capability.startswith("8.0"):
            HAS_SM80 = True
            num = "80"
        elif capability.startswith("8.6"):
            HAS_SM86 = True
            num = "86"
        elif capability.startswith("8.9"):
            HAS_SM89 = True
            num = "89"
        elif capability.startswith("9.0"):
            HAS_SM90 = True
            num = "90a" # need to use sm90a instead of sm90 to use wgmma ptx instruction.
        elif capability.startswith("12.0"):
            HAS_SM120 = True
            num = "120" # need to use sm120a to use mxfp8/mxfp4/nvfp4 instructions.

        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
        if capability.endswith("+PTX"):
            NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]

    ext_modules = []

    if HAS_SM80 or HAS_SM86 or HAS_SM89 or HAS_SM90 or HAS_SM120:
        qattn_extension = CUDAExtension(
            name="sageattention._qattn_sm80",
            sources=[
                "csrc/qattn/pybind_sm80.cpp",
                "csrc/qattn/qk_int_sv_f16_cuda_sm80.cu",
            ],
            extra_compile_args={
                "cxx": CXX_FLAGS,
                "nvcc": NVCC_FLAGS,
            },
        )
        ext_modules.append(qattn_extension)

    if HAS_SM89 or HAS_SM120:
        qattn_extension = CUDAExtension(
            name="sageattention._qattn_sm89",
            sources=[
                "csrc/qattn/pybind_sm89.cpp",
                "csrc/qattn/qk_int_sv_f8_cuda_sm89.cu",
            ],
            extra_compile_args={
                "cxx": CXX_FLAGS,
                "nvcc": NVCC_FLAGS,
            },
        )
        ext_modules.append(qattn_extension)

    if HAS_SM90:
        qattn_extension = CUDAExtension(
            name="sageattention._qattn_sm90",
            sources=[
                "csrc/qattn/pybind_sm90.cpp",
                "csrc/qattn/qk_int_sv_f8_cuda_sm90.cu",
            ],
            extra_compile_args={
                "cxx": CXX_FLAGS,
                "nvcc": NVCC_FLAGS,
            },
            extra_link_args=['-lcuda'],
        )
        ext_modules.append(qattn_extension)

    # Fused kernels.
    fused_extension = CUDAExtension(
        name="sageattention._fused",
        sources=["csrc/fused/pybind.cpp", "csrc/fused/fused.cu"],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
    )
    ext_modules.append(fused_extension)

    return {
        "extensions": ext_modules,
        "cmdclass": {"build_ext": BuildExtension},
    }


setup(**get_extra_args())
