// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SYCL permute kernels for XPU — mirrors cuda/tensor_kernels.cu.
//
// This file provides queue-based SYCL kernels for the block/universal permute
// operations, compiled to a shared library (libkvbm_kernels_sycl.so) and called
// from Rust via extern "C" FFI — identical to the CUDA path.
//
// Three extern "C" launchers:
//   kvbm_kernels_sycl_launch_universal_from_block()
//   kvbm_kernels_sycl_launch_block_from_universal()
//   kvbm_kernels_sycl_launch_nhd_hnd_transpose()
//
// Build:
//   icpx -fsycl -shared -fPIC -O2 -o libkvbm_kernels_sycl.so tensor_permute_kernel.cpp
//
// The queue pointer is passed opaquely from Rust as void*.
// Element-type dispatch uses elem_size (bytes) instead of C++ templates,
// mirroring the OpenCL approach — permute is pure data movement, no arithmetic.

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

enum class BlockLayout : int {
  NHD = 0,
  HND = 1,
};

/// Compute within-chunk flat byte offset for a given (nt_idx, nh_idx, hd_idx).
inline size_t
block_inner_offset(BlockLayout layout,
                   size_t nt_idx, size_t nh_idx, size_t hd_idx,
                   size_t nt,     size_t nh,     size_t hd)
{
  if (layout == BlockLayout::NHD) {
    return ((nt_idx * nh) + nh_idx) * hd + hd_idx;
  } else {
    return ((nh_idx * nt) + nt_idx) * hd + hd_idx;
  }
}

/// Copy `elem_size` bytes from src to dst with width-optimal access.
inline void
copy_element(const uint8_t* src, uint8_t* dst, size_t elem_size)
{
  if (elem_size == 8) {
    *reinterpret_cast<uint64_t*>(dst) = *reinterpret_cast<const uint64_t*>(src);
  } else if (elem_size == 4) {
    *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<const uint32_t*>(src);
  } else if (elem_size == 2) {
    *reinterpret_cast<uint16_t*>(dst) = *reinterpret_cast<const uint16_t*>(src);
  } else {
    for (size_t i = 0; i < elem_size; ++i)
      dst[i] = src[i];
  }
}

constexpr int kBlockDim = 256;

inline size_t
compute_grid_size(size_t total, int block_dim)
{
  if (total == 0) return 0;
  size_t groups = (total + static_cast<size_t>(block_dim) - 1) / static_cast<size_t>(block_dim);
  if (groups > 65535) groups = 65535;
  return groups;
}

} // namespace

extern "C" {

/// Block stacks -> Universal tensors.
///
/// Parameters:
///   universal_ptrs - device-accessible array of num_blocks pointers to universal buffers
///   block_ptrs     - device-accessible array of num_blocks*nl*no pointers to block chunks
///   num_blocks     - number of independent blocks
///   nh,nl,no,nt,hd - tensor dimensions
///   elem_size      - bytes per element (2=f16/bf16, 4=f32, 8=f64)
///   layout_value   - 0=NHD, 1=HND
///   queue_ptr      - opaque sycl::queue* (passed from Rust as *mut c_void)
///
/// Returns 0 on success, non-zero on error.
int kvbm_kernels_sycl_launch_universal_from_block(
    void* const* universal_ptrs,
    const void* const* block_ptrs,
    size_t num_blocks,
    size_t nh, size_t nl, size_t no, size_t nt, size_t hd,
    size_t nl_full, size_t nl_offset,
    size_t elem_size,
    int layout_value,
    void* queue_ptr)
{
  if (num_blocks == 0) return 0;
  if (!universal_ptrs || !block_ptrs || !queue_ptr) return -1;

  auto& q = *static_cast<sycl::queue*>(queue_ptr);
  auto layout = static_cast<BlockLayout>(layout_value);

  size_t block_stride = nl * no;
  size_t total_per_block = nh * nl * no * nt * hd;
  size_t total = total_per_block * num_blocks;

  size_t groups = compute_grid_size(total, kBlockDim);
  if (groups == 0) return 0;

  size_t global_size = groups * kBlockDim;

  try {
    q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kBlockDim)),
      [=](sycl::nd_item<1> item) {
        size_t thread_id = item.get_global_id(0);
        size_t stride = item.get_global_range(0);

        for (; thread_id < total; thread_id += stride) {
          size_t block_idx = thread_id / total_per_block;
          size_t residual  = thread_id % total_per_block;

          size_t tmp = residual;
          size_t hd_idx = tmp % hd;  tmp /= hd;
          size_t nt_idx = tmp % nt;  tmp /= nt;
          size_t no_idx = tmp % no;  tmp /= no;
          size_t nl_idx = tmp % nl;  tmp /= nl;
          size_t nh_idx = tmp;

          // Source: block chunk
          size_t chunk_ptr_idx = block_idx * block_stride + nl_idx * no + no_idx;
          auto* chunk_base = static_cast<const uint8_t*>(block_ptrs[chunk_ptr_idx]);
          size_t chunk_offset = block_inner_offset(layout, nt_idx, nh_idx, hd_idx, nt, nh, hd);

          // Destination: universal buffer — use nl_full stride and nl_offset
          auto* univ_base = static_cast<uint8_t*>(universal_ptrs[block_idx]);
          size_t univ_head_stride = nl_full * no * nt * hd;
          size_t univ_layer_stride = no * nt * hd;
          size_t univ_offset = nh_idx * univ_head_stride
                             + (nl_offset + nl_idx) * univ_layer_stride
                             + no_idx * (nt * hd)
                             + nt_idx * hd + hd_idx;

          copy_element(chunk_base + chunk_offset * elem_size,
                       univ_base  + univ_offset * elem_size,
                       elem_size);
        }
      }
    );
  } catch (const sycl::exception& e) {
    fprintf(stderr, "kvbm_kernels_sycl: universal_from_block failed: %s\n", e.what());
    return -1;
  }
  return 0;
}

/// Universal tensors -> Block stacks (inverse).
int kvbm_kernels_sycl_launch_block_from_universal(
    const void* const* universal_ptrs,
    void* const* block_ptrs,
    size_t num_blocks,
    size_t nh, size_t nl, size_t no, size_t nt, size_t hd,
    size_t nl_full, size_t nl_offset,
    size_t elem_size,
    int layout_value,
    void* queue_ptr)
{
  if (num_blocks == 0) return 0;
  if (!universal_ptrs || !block_ptrs || !queue_ptr) return -1;

  auto& q = *static_cast<sycl::queue*>(queue_ptr);
  auto layout = static_cast<BlockLayout>(layout_value);

  size_t block_stride = nl * no;
  size_t total_per_block = nh * nl * no * nt * hd;
  size_t total = total_per_block * num_blocks;

  size_t groups = compute_grid_size(total, kBlockDim);
  if (groups == 0) return 0;

  size_t global_size = groups * kBlockDim;

  try {
    q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kBlockDim)),
      [=](sycl::nd_item<1> item) {
        size_t thread_id = item.get_global_id(0);
        size_t stride = item.get_global_range(0);

        for (; thread_id < total; thread_id += stride) {
          size_t block_idx = thread_id / total_per_block;
          size_t residual  = thread_id % total_per_block;

          size_t tmp = residual;
          size_t hd_idx = tmp % hd;  tmp /= hd;
          size_t nt_idx = tmp % nt;  tmp /= nt;
          size_t no_idx = tmp % no;  tmp /= no;
          size_t nl_idx = tmp % nl;  tmp /= nl;
          size_t nh_idx = tmp;

          // Source: universal buffer — use nl_full stride and nl_offset
          auto* univ_base = static_cast<const uint8_t*>(universal_ptrs[block_idx]);
          size_t univ_head_stride = nl_full * no * nt * hd;
          size_t univ_layer_stride = no * nt * hd;
          size_t univ_offset = nh_idx * univ_head_stride
                             + (nl_offset + nl_idx) * univ_layer_stride
                             + no_idx * (nt * hd)
                             + nt_idx * hd + hd_idx;

          // Destination: block chunk
          size_t chunk_ptr_idx = block_idx * block_stride + nl_idx * no + no_idx;
          auto* chunk_base = static_cast<uint8_t*>(block_ptrs[chunk_ptr_idx]);
          size_t chunk_offset = block_inner_offset(layout, nt_idx, nh_idx, hd_idx, nt, nh, hd);

          copy_element(univ_base  + univ_offset * elem_size,
                       chunk_base + chunk_offset * elem_size,
                       elem_size);
        }
      }
    );
  } catch (const sycl::exception& e) {
    fprintf(stderr, "kvbm_kernels_sycl: block_from_universal failed: %s\n", e.what());
    return -1;
  }
  return 0;
}

/// NHD - HND transpose. Both sides are operational block stacks shaped as
/// `[nl, no][nt, nh, hd]` (NHD) or `[nl, no][nh, nt, hd]` (HND); the transform
/// swaps the inner (nt, nh) order while keeping `hd` contiguous.
///
/// Mirrors `kvbm_kernels_launch_nhd_hnd_transpose` in cuda/tensor_kernels.cu.
/// Each `(block, nl, no, nt, nh, hd)` element is one work-item; `hd` is the
/// stride-1 axis on both src and dst so adjacent items access adjacent bytes.
///
/// Parameters:
///   src_ptrs       - device-accessible array of num_blocks*nl*no pointers to source chunks
///   dst_ptrs       - device-accessible array of num_blocks*nl*no pointers to dest chunks
///   num_blocks     - number of independent blocks
///   nl             - number of KV-cache layers
///   no             - number of outer dimensions (typically 2 for K and V)
///   nt             - page size (tokens per block)
///   nh             - number of attention heads
///   hd             - head dimension (elements per head)
///   elem_size      - bytes per element (2=f16/bf16, 4=f32, 8=f64)
///   src_layout_value - 0=NHD (src is NHD, dst is HND), 1=HND (src is HND, dst is NHD)
///   queue_ptr      - opaque sycl::queue* (passed from Rust as *mut c_void)
///
/// Returns 0 on success, non-zero on error.
int kvbm_kernels_sycl_launch_nhd_hnd_transpose(
    const void* const* src_ptrs,
    void* const* dst_ptrs,
    size_t num_blocks,
    size_t nl, size_t no, size_t nt, size_t nh, size_t hd,
    size_t elem_size,
    int src_layout_value,
    void* queue_ptr)
{
  if (num_blocks == 0) return 0;
  if (!src_ptrs || !dst_ptrs || !queue_ptr) return -1;

  auto& q = *static_cast<sycl::queue*>(queue_ptr);
  auto src_layout = static_cast<BlockLayout>(src_layout_value);
  auto dst_layout = (src_layout == BlockLayout::NHD) ? BlockLayout::HND : BlockLayout::NHD;

  size_t block_stride = nl * no;
  size_t total_per_block = nl * no * nt * nh * hd;
  size_t total = total_per_block * num_blocks;

  size_t groups = compute_grid_size(total, kBlockDim);
  if (groups == 0) return 0;

  size_t global_size = groups * kBlockDim;

  try {
    q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kBlockDim)),
      [=](sycl::nd_item<1> item) {
        size_t thread_id = item.get_global_id(0);
        size_t stride = item.get_global_range(0);

        for (; thread_id < total; thread_id += stride) {
          size_t block_idx = thread_id / total_per_block;
          size_t residual  = thread_id % total_per_block;

          size_t tmp = residual;
          size_t hd_idx = tmp % hd;  tmp /= hd;
          size_t nt_idx = tmp % nt;  tmp /= nt;
          size_t nh_idx = tmp % nh;  tmp /= nh;
          size_t no_idx = tmp % no;  tmp /= no;
          size_t nl_idx = tmp;

          size_t chunk_ptr_idx = block_idx * block_stride + nl_idx * no + no_idx;
          auto* src_chunk = static_cast<const uint8_t*>(src_ptrs[chunk_ptr_idx]);
          auto* dst_chunk = static_cast<uint8_t*>(dst_ptrs[chunk_ptr_idx]);

          size_t src_off = block_inner_offset(src_layout, nt_idx, nh_idx, hd_idx, nt, nh, hd);
          size_t dst_off = block_inner_offset(dst_layout, nt_idx, nh_idx, hd_idx, nt, nh, hd);

          copy_element(src_chunk + src_off * elem_size,
                       dst_chunk + dst_off * elem_size,
                       elem_size);
        }
      }
    );
  } catch (const sycl::exception& e) {
    fprintf(stderr, "kvbm_kernels_sycl: nhd_hnd_transpose failed: %s\n", e.what());
    return -1;
  }
  return 0;
}

} // extern "C"
