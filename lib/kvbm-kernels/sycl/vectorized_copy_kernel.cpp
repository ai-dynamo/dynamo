// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SYCL vectorized copy kernel for XPU — extern "C" FFI launcher.
//
// Mirrors the CUDA kvbm_kernels_launch_vectorized_copy() but uses a
// sycl::queue* for submission instead of a cudaStream_t.
//
// Build (part of libkvbm_kernels_xpu.so):
//   icpx -fsycl -shared -fPIC -O2 -o libkvbm_kernels_xpu.so \
//        tensor_permute_kernel.cpp vectorized_copy_kernel.cpp
//
// The queue pointer is passed opaquely from Rust as void*.

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace {

/// Portable 16-byte aligned type for vectorized loads/stores.
struct alignas(16) uint128_t {
  unsigned char bytes[16];
};

constexpr int kBlockDim = 128;

} // namespace

extern "C" {

/// Vectorized copy between device-visible pointer pairs (SYCL queue launcher).
///
/// Each work-group handles one or more (src, dst) pairs using a group-strided
/// loop.  Per-pair alignment detection selects the widest safe vector width:
///   16-byte (uint128_t) → 8-byte (uint64_t) → 4-byte (uint32_t) → 1-byte
///
/// Parameters:
///   src_ptrs        - device-accessible array of `num_pairs` source pointers
///   dst_ptrs        - device-accessible array of `num_pairs` destination pointers
///   copy_size_bytes - bytes to copy per pair (same for all pairs)
///   num_pairs       - number of pointer pairs
///   queue_ptr       - opaque sycl::queue* (passed from Rust as *mut c_void)
///
/// Returns 0 on success, non-zero on error.
int kvbm_kernels_xpu_launch_vectorized_copy(
    void** src_ptrs,
    void** dst_ptrs,
    size_t copy_size_bytes,
    int num_pairs,
    void* queue_ptr)
{
  if (num_pairs <= 0) return 0;
  if (!src_ptrs || !dst_ptrs || !queue_ptr) return -1;

  auto& q = *static_cast<sycl::queue*>(queue_ptr);

  int grid_dim = num_pairs;
  if (grid_dim > 65535) grid_dim = 65535;

  size_t global_size = static_cast<size_t>(grid_dim) * kBlockDim;

  try {
    q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kBlockDim)),
      [=](sycl::nd_item<1> item) {
        size_t group_id = item.get_group(0);
        size_t group_range = item.get_group_range(0);
        size_t local_id = item.get_local_id(0);
        size_t local_range = item.get_local_range(0);

        for (size_t pair_id = group_id;
             pair_id < static_cast<size_t>(num_pairs);
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
            for (size_t i = local_id; i < num_vec; i += local_range)
              dst_vec[i] = src_vec[i];
            vectorized_bytes = num_vec << 4;
          } else if (((src_addr & 0x7) == 0) && ((dst_addr & 0x7) == 0) &&
                     (copy_size_bytes >= 8)) {
            // 8-byte vectorized copy
            size_t num_u64 = copy_size_bytes >> 3;
            auto* src_u64 = reinterpret_cast<const uint64_t*>(src);
            auto* dst_u64 = reinterpret_cast<uint64_t*>(dst);
            for (size_t i = local_id; i < num_u64; i += local_range)
              dst_u64[i] = src_u64[i];
            vectorized_bytes = num_u64 << 3;
          } else if (((src_addr & 0x3) == 0) && ((dst_addr & 0x3) == 0) &&
                     (copy_size_bytes >= 4)) {
            // 4-byte vectorized copy
            size_t num_u32 = copy_size_bytes >> 2;
            auto* src_u32 = reinterpret_cast<const uint32_t*>(src);
            auto* dst_u32 = reinterpret_cast<uint32_t*>(dst);
            for (size_t i = local_id; i < num_u32; i += local_range)
              dst_u32[i] = src_u32[i];
            vectorized_bytes = num_u32 << 2;
          }

          // Handle remaining bytes
          size_t remaining = copy_size_bytes - vectorized_bytes;
          for (size_t i = local_id; i < remaining; i += local_range)
            dst[vectorized_bytes + i] = src[vectorized_bytes + i];
        }
      }
    );
  } catch (const sycl::exception& e) {
    return -1;
  }
  return 0;
}

} // extern "C"
