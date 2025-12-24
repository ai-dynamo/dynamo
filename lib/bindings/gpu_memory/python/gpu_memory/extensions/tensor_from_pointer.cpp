// Minimal "tensor from CUDA pointer" helper.
//
// This is used by the import-only loader path to reconstruct torch Tensors that
// alias GPU memory mapped via CUDA VMM (cuMemMap) in this process.
//
// WARNING:
// - The returned tensor does NOT own the underlying memory. The caller must
//   ensure the mapped allocation remains valid for the lifetime of the tensor.
// - This is intended for read-only inference weights (RO mappings).
//
// Python API:
//   tensor_from_pointer_contiguous(data_ptr, shape, dtype, device_index) -> Tensor
//   tensor_from_pointer(data_ptr, shape, strides, dtype, device_index) -> Tensor

#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

// No-op deleter: the allocator/lock owner is responsible for unmapping.
void
noop_deleter(void*)
{
}

torch::Tensor
make_tensor_from_ptr(
    uint64_t data_ptr, const std::vector<int64_t>& shape, const c10::optional<std::vector<int64_t>>& strides,
    torch::Dtype dtype, int64_t device_index)
{
  void* ptr = reinterpret_cast<void*>(static_cast<uintptr_t>(data_ptr));

  auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA, static_cast<int>(device_index));

  if (strides.has_value()) {
    return torch::from_blob(ptr, shape, strides.value(), &noop_deleter, options);
  }
  return torch::from_blob(ptr, shape, &noop_deleter, options);
}

}  // namespace

torch::Tensor
tensor_from_pointer_contiguous(
    uint64_t data_ptr, const std::vector<int64_t>& shape, torch::Dtype dtype, int64_t device_index)
{
  return make_tensor_from_ptr(data_ptr, shape, c10::nullopt, dtype, device_index);
}

torch::Tensor
tensor_from_pointer(
    uint64_t data_ptr, const std::vector<int64_t>& shape, const std::vector<int64_t>& strides, torch::Dtype dtype,
    int64_t device_index)
{
  return make_tensor_from_ptr(data_ptr, shape, strides, dtype, device_index);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def(
      "tensor_from_pointer_contiguous", &tensor_from_pointer_contiguous,
      R"doc(
tensor_from_pointer_contiguous(data_ptr: int, shape: Sequence[int], dtype: torch.dtype, device_index: int) -> torch.Tensor

Create contiguous tensor from CUDA pointer
)doc");

  m.def(
      "tensor_from_pointer", &tensor_from_pointer,
      R"doc(
tensor_from_pointer(data_ptr: int, shape: Sequence[int], strides: Sequence[int], dtype: torch.dtype, device_index: int) -> torch.Tensor

Create tensor from CUDA pointer with explicit strides
)doc");
}
