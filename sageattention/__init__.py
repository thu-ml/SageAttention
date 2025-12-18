import torch

def is_hip() -> bool:
    return torch.version.hip is not None


def on_gfx942() -> bool:
    GPU_ARCH = torch.cuda.get_device_properties("cuda").gcnArchName
    return any(arch in GPU_ARCH for arch in ["gfx942"])

from .core import sageattn, sageattn_varlen
from .core import sageattn_qk_int8_pv_fp16_triton
from .core import sageattn_qk_int8_pv_fp16_cuda 
from .core import sageattn_qk_int8_pv_fp8_cuda
from .core import sageattn_qk_int8_pv_fp8_cuda_sm90