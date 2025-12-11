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

import os
import sys
import subprocess
import threading
import warnings
from packaging.version import parse, Version

from setuptools import setup, find_packages

# Skip CUDA build in CI or when explicitly requested
SKIP_CUDA_BUILD = (
    os.getenv("SAGEATTN_SKIP_CUDA_BUILD", "0").upper() in {"1", "TRUE", "YES"}
    or ("sdist" in sys.argv)
)

ext_modules = []
cmdclass = {}

if not SKIP_CUDA_BUILD:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

    HAS_SM80 = False
    HAS_SM86 = False
    HAS_SM89 = False
    HAS_SM90 = False
    HAS_SM100 = False
    HAS_SM120 = False

    # Supported NVIDIA GPU architectures.
    SUPPORTED_ARCHS = {"8.0", "8.6", "8.9", "9.0", "10.0" "12.0"}

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
        "-diag-suppress=174",
    ]

    # Append flags from env if provided
    cxx_append = os.getenv("CXX_APPEND_FLAGS", "").strip()
    if cxx_append:
        CXX_FLAGS += cxx_append.split()
    nvcc_append = os.getenv("NVCC_APPEND_FLAGS", "").strip()
    if nvcc_append:
        NVCC_FLAGS += nvcc_append.split()

    ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
    NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

    if CUDA_HOME is None:
        raise RuntimeError(
            "Cannot find CUDA_HOME. CUDA must be available to build the package.")

    def get_nvcc_cuda_version(cuda_dir: str) -> Version:
        """Get the CUDA version from nvcc.

        Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
        """
        nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                              universal_newlines=True)
        output = nvcc_output.split()
        release_idx = output.index("release") + 1
        nvcc_cuda_version = parse(output[release_idx].split(",")[0])
        return nvcc_cuda_version

    # Determine target compute capabilities
    compute_capabilities = set()

    # Prefer TORCH_CUDA_ARCH_LIST if explicitly specified (works without GPUs)
    arch_list_env = os.getenv("TORCH_CUDA_ARCH_LIST", "").strip()
    if arch_list_env:
        for item in arch_list_env.replace(",", ";").split(";"):
            it = item.strip()
            if not it:
                continue
            it = it.lower().replace("sm_", "").replace("compute_", "")
            it = it.replace("a", "")
            if it.endswith("+ptx"):
                it = it[:-4]
                compute_capabilities.add(f"{it}+PTX")
            else:
                if len(it) == 2 and it.isdigit():
                    it = f"{it[0]}.{it[1]}"
                compute_capabilities.add(it)

    # If not provided, try to detect from local GPUs
    if not compute_capabilities:
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        for i in range(device_count):
            major, minor = torch.cuda.get_device_capability(i)
            if major < 8:
                warnings.warn(f"skipping GPU {i} with compute capability {major}.{minor}")
                continue
            compute_capabilities.add(f"{major}.{minor}")

    nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)

    if not compute_capabilities:
        raise RuntimeError(
            "No target compute capabilities. Set TORCH_CUDA_ARCH_LIST or build on a machine with GPUs.")
    else:
        print(f"Target compute capabilities: {compute_capabilities}")

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
            num = "90a"
        elif capability.startswith("10.0"):
            HAS_SM100 = True
            num = "100a"
        elif capability.startswith("12.0"):
            HAS_SM120 = True
            num = "120a"
        else:
            continue
        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
        if capability.endswith("+PTX"):
            NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]

    # Fused kernels and QAttn variants
    from torch.utils.cpp_extension import CUDAExtension

    if HAS_SM80 or HAS_SM86 or HAS_SM89 or HAS_SM90 or HAS_SM100 or HAS_SM120:
        ext_modules.append(
            CUDAExtension(
                name="sageattention._qattn_sm80",
                sources=[
                    "csrc/qattn/pybind_sm80.cpp",
                    "csrc/qattn/qk_int_sv_f16_cuda_sm80.cu",
                ],
                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
            )
        )

    if HAS_SM89 or HAS_SM90 or HAS_SM100 or HAS_SM120:
        ext_modules.append(
            CUDAExtension(
                name="sageattention._qattn_sm89",
                sources=[
                    "csrc/qattn/pybind_sm89.cpp",
                    "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn_inst_buf.cu",
                    "csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_attn_inst_buf.cu",
                    "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_attn.cu",
                    "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn.cu",
                    "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn.cu",
                    "csrc/qattn/sm89_qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf.cu",
                    "csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf.cu",
                ],
                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
            )
        )

    if HAS_SM90:
        ext_modules.append(
            CUDAExtension(
                name="sageattention._qattn_sm90",
                sources=[
                    "csrc/qattn/pybind_sm90.cpp",
                    "csrc/qattn/qk_int_sv_f8_cuda_sm90.cu",
                ],
                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
                extra_link_args=['-lcuda'],
            )
        )

    ext_modules.append(
        CUDAExtension(
            name="sageattention._fused",
            sources=["csrc/fused/pybind.cpp", "csrc/fused/fused.cu"],
            extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
        )
    )

    # Resolve parallelism from env
    parallel = None
    if 'EXT_PARALLEL' in os.environ:
        try:
            parallel = int(os.getenv('EXT_PARALLEL'))
        finally:
            pass
    if parallel is None and 'MAX_JOBS' in os.environ:
        try:
            parallel = int(os.getenv('MAX_JOBS'))
        finally:
            pass
    # Defaults if not provided
    if parallel is None:
        parallel = 4
    # Ensure MAX_JOBS for underlying tooling if not explicitly set
    os.environ.setdefault('MAX_JOBS', '32')

    class BuildExtensionSeparateDir(BuildExtension):
        build_extension_patch_lock = threading.Lock()
        thread_ext_name_map = {}

        def finalize_options(self):
            if parallel is not None:
                self.parallel = parallel
            super().finalize_options()

        def build_extension(self, ext):
            with self.build_extension_patch_lock:
                if not getattr(self.compiler, "_compile_separate_output_dir", False):
                    compile_orig = self.compiler.compile

                    def compile_new(*args, **kwargs):
                        return compile_orig(*args, **{
                            **kwargs,
                            "output_dir": os.path.join(
                                kwargs["output_dir"],
                                self.thread_ext_name_map[threading.current_thread().ident]),
                        })
                    self.compiler.compile = compile_new
                    self.compiler._compile_separate_output_dir = True
            self.thread_ext_name_map[threading.current_thread().ident] = ext.name
            objects = super().build_extension(ext)
            return objects

    cmdclass = {"build_ext": BuildExtensionSeparateDir} if ext_modules else {}

setup(
    name='sageattention',
    version='2.2.0',
    author='SageAttention team',
    license='Apache 2.0 License',
    description='Accurate and efficient plug-and-play low-bit attention.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/thu-ml/SageAttention',
    packages=find_packages(),
    python_requires='>=3.9',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
