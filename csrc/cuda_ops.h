#pragma once

#include <c10/util/TypeCast.h>
#include <torch/library.h>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace sageattention {

// Dispatcher scalar types are wider than the CUDA launcher parameters.
template<typename T>
using SchemaArg = std::conditional_t<std::is_same_v<T, int>, int64_t,
    std::conditional_t<std::is_same_v<T, float>, double, T>>;

template<typename T>
T kernel_arg(SchemaArg<T> arg) {
  if constexpr (std::is_same_v<T, int>) {
    return c10::checked_convert<int>(arg, "int");
  } else {
    return static_cast<T>(std::move(arg));
  }
}

template<auto Function>
struct KernelAdapter;

template<typename Result, typename... Args, Result (*Function)(Args...)>
struct KernelAdapter<Function> {
  static Result call(SchemaArg<Args>... args) {
    return Function(kernel_arg<Args>(std::move(args))...);
  }
};

template<auto Function>
auto cuda_kernel() {
  return torch::dispatch(c10::DispatchKey::CUDA, &KernelAdapter<Function>::call);
}

}  // namespace sageattention

#define SAGEATTN_QK_SCHEMA(name, extra_tensors) \
  #name "(Tensor query, Tensor key, Tensor value, Tensor(a!) output, Tensor query_scale, Tensor key_scale, " \
  extra_tensors "int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> Tensor"
