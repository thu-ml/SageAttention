import warnings
import os
from pathlib import Path
from packaging.version import parse, Version
from setuptools import setup, find_packages
import subprocess
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "sageattn3"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("FAHOPPER_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("FAHOPPER_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FAHOPPER_FORCE_CXX11_ABI", "FALSE") == "TRUE"

# Supported NVIDIA GPU architectures; keep in sync with workflows.
SUPPORTED_ARCHS = {"10.0", "12.0"}


def normalize_arch(arch: str):
    """Normalize an architecture spec from TORCH_CUDA_ARCH_LIST."""

    token = arch.strip()
    if not token:
        return None
    cleaned = token.upper().replace("SM_", "").replace("COMPUTE_", "")
    ptx = cleaned.endswith("+PTX")
    if ptx:
        cleaned = cleaned[:-4]
    suffix = ""
    if cleaned.endswith("A"):
        suffix = "a"
        cleaned = cleaned[:-1]
    cleaned = cleaned.replace(" ", "")
    if "." in cleaned:
        major, _, minor = cleaned.partition(".")
        if not (major.isdigit() and minor and minor[0].isdigit()):
            warnings.warn(
                f"Invalid CUDA architecture '{arch}'. Supported architectures are {SUPPORTED_ARCHS}."
            )
            return None
        base = f"{int(major)}.{minor[0]}"
    else:
        if len(cleaned) != 2 or not cleaned.isdigit():
            warnings.warn(
                f"Invalid CUDA architecture '{arch}'. Supported architectures are {SUPPORTED_ARCHS}."
            )
            return None
        base = f"{cleaned[0]}.{cleaned[1]}"
    if base not in SUPPORTED_ARCHS:
        warnings.warn(
            f"Unsupported CUDA architecture '{arch}'. Supported architectures are {SUPPORTED_ARCHS}."
        )
        return None
    capability = f"{base}{suffix}"
    if ptx:
        capability += "+PTX"
    return capability


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]


cmdclass = {}
ext_modules = []

if not SKIP_CUDA_BUILD:
    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])

    cc_flag = []
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.8"):
        raise RuntimeError("Sage3 is only supported on CUDA 12.8 and above")
    compute_capabilities = []
    arch_list_env = os.getenv("TORCH_CUDA_ARCH_LIST", "").strip()
    if arch_list_env:
        for item in arch_list_env.replace(",", ";").split(";"):
            capability = normalize_arch(item)
            if capability is not None and capability not in compute_capabilities:
                compute_capabilities.append(capability)

    if not compute_capabilities:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError(
                "Unable to detect a CUDA device. Set TORCH_CUDA_ARCH_LIST to target 10.0 or 12.0."
            )
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            capability = f"{major}.{minor}"
            if capability not in SUPPORTED_ARCHS:
                warnings.warn(
                    f"skipping GPU {i} with compute capability {capability}; supported {SUPPORTED_ARCHS}"
                )
                continue
            if capability not in compute_capabilities:
                compute_capabilities.append(capability)

    if not compute_capabilities:
        raise RuntimeError(
            "No target compute capabilities. Set TORCH_CUDA_ARCH_LIST or build on a supported GPU (sm_100 or sm_120)."
        )

    for capability in compute_capabilities:
        if capability.startswith("10.0"):
            HAS_SM100 = True
            num = "100a"
        elif capability.startswith("12.0"):
            HAS_SM120 = True
            num = "120a"
        cc_flag.append([f"-gencode=arch=compute_{num},code=sm_{num}"])
        if capability.endswith("+PTX"):
            cc_flag[-1].append(f"-gencode=arch=compute_{num},code=compute_{num}")
        else:
            warnings.warn(
                f"Ignoring unsupported compute capability '{capability}'. Supported bases: {SUPPORTED_ARCHS}."
            )
            continue

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
    repo_dir = Path(this_dir)
    cutlass_dir = repo_dir / "csrc" / "cutlass"
    (repo_dir / "csrc").mkdir(parents=True, exist_ok=True)
    if not cutlass_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/NVIDIA/cutlass.git", str(cutlass_dir)],
            check=True
        )
    nvcc_flags = [
        "-O3",
        # "-O0",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        # "--ptxas-options=-v",  # printing out number of registers
        "--ptxas-options=--verbose,--warn-on-local-memory-usage",  # printing out number of registers
        "-lineinfo",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
        "-DNDEBUG",  # Important, otherwise performance is severely impacted
        "-DQBLKSIZE=128",
        "-DKBLKSIZE=128",
        "-DCTA256",
        "-DDQINRMEM",
    ]
    include_dirs = [
        repo_dir / "sageattn3",
        cutlass_dir / "include",
        cutlass_dir / "tools" / "util" / "include",
    ]

    ext_modules.append(
        CUDAExtension(
            name="fp4attn_cuda",
            sources=["sageattn3/blackwell/api.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads(
                    nvcc_flags + ["-DEXECMODE=0"] + cc_flag
                ),
            },
            include_dirs=include_dirs,
            # Without this we get and error about cuTensorMapEncodeTiled not defined
            libraries=["cuda"]
        )
    )
    ext_modules.append(
        CUDAExtension(
            name="fp4quant_cuda",
            sources=["sageattn3/quantization/fp4_quantization_4d.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads(
                    nvcc_flags + ["-DEXECMODE=0"] + cc_flag
                ),
            },
            include_dirs=include_dirs,
            # Without this we get and error about cuTensorMapEncodeTiled not defined
            libraries=["cuda"]
        )
    )



class CachedWheelsCommand(_bdist_wheel):
    def run(self):
        super().run()

setup(
    name=PACKAGE_NAME,
    version="1.0.0",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    description="FP4FlashAttention",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": BuildExtension}
    if ext_modules
    else {
        "bdist_wheel": CachedWheelsCommand,
    },
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
)
