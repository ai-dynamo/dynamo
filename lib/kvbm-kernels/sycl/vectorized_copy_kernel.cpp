// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SYCL vectorized copy kernel for XPU.
//
// This is the SYCL equivalent of the CUDA kvbm_kernels_vectorized_copy_kernel.
// It copies data between arbitrary device-visible pointer pairs using
// alignment-based vectorization (16/8/4/1 byte widths).
//
// Compiled at runtime via oneAPI-rs load_program_from_source() (JIT).
// Launched via SyclKernel::launch_1d() with nd_range<1>.

#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

namespace {

// Portable 16-byte aligned type for vectorized loads/stores.
// Avoids dependence on sycl::vec memory layout.
struct alignas(16) uint128_t {
  uint64_t lo;
  uint64_t hi;
};

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

/// Vectorized memory copy kernel for arbitrary device-visible pointer pairs.
///
/// Each work group handles one or more (src, dst) pairs using a group-strided
/// loop. Per-pair alignment detection selects the widest safe vector width:
///   - 16-byte (uint128_t) if both pointers are 16-byte aligned
///   - 8-byte  (uint64_t)  if both pointers are 8-byte aligned
///   - 4-byte  (uint32_t)  if both pointers are 4-byte aligned
///   - 1-byte  fallback for any remainder
///
/// Source and destination pointers may be device USM or host USM memory.
///
/// Parameters (passed via SyclKernelArg):
///   src_ptrs        - device pointer to array of source pointers
///   dst_ptrs        - device pointer to array of destination pointers
///   copy_size_bytes - size of each copy in bytes (same for all pairs)
///   num_pairs       - number of pointer pairs to copy
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void kvbm_vectorized_copy(void** src_ptrs, void** dst_ptrs,
                          size_t copy_size_bytes, int num_pairs) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  size_t group_id = item.get_group(0);
  size_t group_range = item.get_group_range(0);
  size_t local_id = item.get_local_id(0);
  size_t local_range = item.get_local_range(0);

  for (size_t pair_id = group_id; pair_id < static_cast<size_t>(num_pairs);
       pair_id += group_range) {
    char* src = static_cast<char*>(src_ptrs[pair_id]);
    char* dst = static_cast<char*>(dst_ptrs[pair_id]);

    uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
    uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);

    size_t vectorized_bytes = 0;

    if (((src_addr & 0xF) == 0) && ((dst_addr & 0xF) == 0) &&
        (copy_size_bytes >= 16)) {
      // 16-byte vectorized copy
      size_t num_vec = copy_size_bytes >> 4;
      auto* src_vec = reinterpret_cast<const uint128_t*>(src);
      auto* dst_vec = reinterpret_cast<uint128_t*>(dst);
      for (size_t i = local_id; i < num_vec; i += local_range) {
        dst_vec[i] = src_vec[i];
      }
      vectorized_bytes = num_vec << 4;
    } else if (((src_addr & 0x7) == 0) && ((dst_addr & 0x7) == 0) &&
               (copy_size_bytes >= 8)) {
      // 8-byte vectorized copy
      size_t num_u64 = copy_size_bytes >> 3;
      auto* src_u64 = reinterpret_cast<const uint64_t*>(src);
      auto* dst_u64 = reinterpret_cast<uint64_t*>(dst);
      for (size_t i = local_id; i < num_u64; i += local_range) {
        dst_u64[i] = src_u64[i];
      }
      vectorized_bytes = num_u64 << 3;
    } else if (((src_addr & 0x3) == 0) && ((dst_addr & 0x3) == 0) &&
               (copy_size_bytes >= 4)) {
      // 4-byte vectorized copy
      size_t num_u32 = copy_size_bytes >> 2;
      auto* src_u32 = reinterpret_cast<const uint32_t*>(src);
      auto* dst_u32 = reinterpret_cast<uint32_t*>(dst);
      for (size_t i = local_id; i < num_u32; i += local_range) {
        dst_u32[i] = src_u32[i];
      }
      vectorized_bytes = num_u32 << 2;
    }

    // Handle remaining bytes
    size_t remaining = copy_size_bytes - vectorized_bytes;
    for (size_t i = local_id; i < remaining; i += local_range) {
      dst[vectorized_bytes + i] = src[vectorized_bytes + i];
    }
  }
}

#ifdef __cplusplus
}
#endif
