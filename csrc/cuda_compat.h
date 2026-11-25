#pragma once

#if defined(_WIN32) && defined(__CUDACC__) && __CUDACC_VER_MAJOR__ >= 13
#include <sal.h>

// Parse CCCL PTX wrappers before PyTorch headers, without SAL's __out macro.
#pragma push_macro("__out")
#undef __out
#include <cuda/ptx>
#pragma pop_macro("__out")
#endif
