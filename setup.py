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
import platform
import subprocess
import threading
import warnings
import torch
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
    # 获取当前PyTorch版本
    torch_version = parse(torch.__version__)
    
    HAS_SM80 = False
    HAS_SM86 = False
    HAS_SM89 = False
    HAS_SM90 = False
    HAS_SM100 = False
    HAS_SM120 = False
    
    # Supported NVIDIA GPU architectures.
    SUPPORTED_ARCHS = {"8.0", "8.6", "8.9", "9.0", "10.0", "12.0"}
    
    # 检测是否为Windows系统
    IS_WINDOWS = platform.system() == "Windows"
    
    # 根据平台和PyTorch版本设置不同的编译标志
    if IS_WINDOWS:
        # Windows/MSVC specific flags
        CXX_FLAGS = [
            '/permissive-',  # 放宽标准符合性检查
            '/std:c++17',
            '/Zc:__cplusplus',  # 正确设置__cplusplus宏
            '/wd4819',  # 抑制特定警告
            '/wd4251',
            '/wd4244',
            '/wd4267',
            '/wd4275',
            '/wd4018',
            '/wd4190',
            '/wd4624',
            '/wd4067',
            '/wd4068',
            '/EHsc',
            '/D_ENABLE_EXTENDED_ALIGNED_STORAGE',  # 解决Windows对齐问题
            '/D_CRT_SECURE_NO_WARNINGS'  # 禁用安全警告
        ]
        
        # 为较新版本的PyTorch添加特定标志
        if torch_version >= Version("2.2.0"):
            CXX_FLAGS.extend([
                '/DC10_USE_MSGLOG_CTOR_WITH_BOOL=1',  # 适配新的MessageLogger构造函数
                '/DTORCH_VERSION_MAJOR=' + str(torch_version.major),
                '/DTORCH_VERSION_MINOR=' + str(torch_version.minor),
            ])
        
        NVCC_FLAGS = [
            "-O3",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--use_fast_math",
            "--threads=8",
            "-Xptxas=-v",
            "-diag-suppress=174",
            # 传递MSVC特定选项给nvcc
            "-Xcompiler", "/permissive-",
            "-Xcompiler", "/wd4819",
            "-Xcompiler", "/wd4251",
            "-Xcompiler", "/wd4244",
            "-Xcompiler", "/wd4267",
            "-Xcompiler", "/wd4275",
            "-Xcompiler", "/wd4018",
            "-Xcompiler", "/wd4190",
            "-Xcompiler", "/wd4624",
            "-Xcompiler", "/wd4067",
            "-Xcompiler", "/wd4068",
            "-Xcompiler", "/EHsc",
            "-Xcompiler", "/D_ENABLE_EXTENDED_ALIGNED_STORAGE"
        ]
        
        # 为较新版本的PyTorch添加特定标志
        if torch_version >= Version("2.2.0"):
            NVCC_FLAGS.extend([
                "-Xcompiler", "/DC10_USE_MSGLOG_CTOR_WITH_BOOL=1",
                "-DTORCH_VERSION_MAJOR=" + str(torch_version.major),
                "-DTORCH_VERSION_MINOR=" + str(torch_version.minor),
            ])
    else:
        # Linux/Unix flags
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
    
    # 设置链接参数 - Windows需要特殊处理
    if IS_WINDOWS:
        # 确保正确的库链接顺序
        EXTRA_LINK_ARGS = ['/NODEFAULTLIB:LIBCMT.LIB']
        
        # 添加PyTorch 2.2.0+需要的额外链接库
        if torch_version >= Version("2.2.0"):
            EXTRA_LINK_ARGS.append('/DEFAULTLIB:legacy_stdio_definitions.lib')
    else:
        EXTRA_LINK_ARGS = ['-lc10', '-ltorch', '-ltorch_cpu', '-ltorch_python']
    
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
    
    from torch.utils.cpp_extension import CUDA_HOME
    
    if CUDA_HOME is None:
        raise RuntimeError(
            "Cannot find CUDA_HOME. CUDA must be available to build the package.")
    
    def get_nvcc_cuda_version(cuda_dir: str) -> Version:
        """Get the CUDA version from nvcc.
        Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
        """
        if IS_WINDOWS:
            # Windows路径处理
            nvcc_path = os.path.join(cuda_dir, "bin", "nvcc")
        else:
            nvcc_path = cuda_dir + "/bin/nvcc"
            
        try:
            nvcc_output = subprocess.check_output([nvcc_path, "-V"],
                                                 universal_newlines=True)
        except FileNotFoundError:
            raise RuntimeError(f"Could not find nvcc at {nvcc_path}. Ensure CUDA is properly installed.")
            
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
    else:
        # If not provided, try to detect from local GPUs
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
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA version used for build: {nvcc_cuda_version}")
    
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
    
    # 为Windows平台添加静态链接选项
    if IS_WINDOWS:
        extra_link_args = EXTRA_LINK_ARGS.copy()
        # 添加CUDA静态库
        cuda_lib_path = os.path.join(CUDA_HOME, "lib", "x64")
        extra_link_args.extend([f'/LIBPATH:{cuda_lib_path}'])
    else:
        extra_link_args = EXTRA_LINK_ARGS
    
    if HAS_SM80 or HAS_SM86 or HAS_SM89 or HAS_SM90 or HAS_SM100 or HAS_SM120:
        ext_modules.append(
            CUDAExtension(
                name="sageattention._qattn_sm80",
                sources=[
                    "csrc/qattn/pybind_sm80.cpp",
                    "csrc/qattn/qk_int_sv_f16_cuda_sm80.cu",
                ],
                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
                extra_link_args=extra_link_args,
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
                extra_link_args=extra_link_args,
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
                extra_link_args=extra_link_args if IS_WINDOWS else ['-lcuda'] + extra_link_args,
            )
        )
    
    ext_modules.append(
        CUDAExtension(
            name="sageattention._fused",
            sources=["csrc/fused/pybind.cpp", "csrc/fused/fused.cu"],
            extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
            extra_link_args=extra_link_args,
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
    
    from torch.utils.cpp_extension import BuildExtension
    
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
    long_description=open('README.md', encoding='utf-8').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/thu-ml/SageAttention',
    packages=find_packages(),
    python_requires='>=3.9',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)