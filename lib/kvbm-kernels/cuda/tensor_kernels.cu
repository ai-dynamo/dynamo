// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>

// Compile-time CUDA version detection and diagnostics
#if defined(CUDART_VERSION)
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#if CUDART_VERSION >= 13000
#elif CUDART_VERSION >= 12090
#else
#pragma message("Building with CUDA " TOSTRING(CUDART_VERSION) " - cudaMemcpyBatchAsync NOT available (requires 12.9+)")
#endif
#else
#pragma message("Warning: CUDART_VERSION not defined - cannot detect CUDA version")
#endif

#ifndef CUDA_CALLABLE_MEMBER
#define CUDA_CALLABLE_MEMBER __host__ __device__
#endif

namespace {

/**
 * There are three logical tensor views involved in these kernels:
 *
 * 1. Universal blocks: contiguous buffers whose logical shape is
 *    [nh, nl, no, nt, hd]. Every “block” is a separate pointer.
 * 2. NHD/HND block stacks: `nl * no` pointers per block, each pointing
 *    to a chunk shaped either [nt, nh, hd] (NHD) or [nh, nt, hd] (HND).
 *    Stacks are arranged as `[layer][outer]`.
 * 3. Operational blocks: contiguous buffers whose logical shape is
 *    [nl, no, inner], where inner = nt * nh * hd. These are used when
 *    the consumer does not care about the split between nh/nt/hd.
 *
 * Each kernel batch-processes `num_blocks` block pairs. All pointer
 * tables are flattened on the host:
 *   • universal_ptrs_device  : [num_blocks]
 *   • block_ptrs_device      : [num_blocks * nl * no]
 *   • operational_ptrs_device: [num_blocks]
 *
 * This lets us launch a single grid per direction, keeps the per-block
 * math regular, and avoids any per-kernel pointer chasing on the CPU.
 */

enum class TensorDataType : int {
  F16 = 0,
  BF16 = 1,
  F32 = 2,
  F64 = 3,
};

enum class BlockLayout : int {
  NHD = 0,
  HND = 1,
};

enum class OperationalCopyDirection : int {
  BlockToOperational = 0,
  OperationalToBlock = 1,
};

template <TensorDataType>
struct DTypeTraits;

template <>
struct DTypeTraits<TensorDataType::F16> {
  using type = __half;
};

template <>
struct DTypeTraits<TensorDataType::BF16> {
  using type = __nv_bfloat16;
};

template <>
struct DTypeTraits<TensorDataType::F32> {
  using type = float;
};

template <>
struct DTypeTraits<TensorDataType::F64> {
  using type = double;
};

template <typename T>
CUDA_CALLABLE_MEMBER inline T*
ptr_offset(T* base, size_t index)
{
  return base + index;
}

template <typename T>
CUDA_CALLABLE_MEMBER inline const T*
ptr_offset(const T* base, size_t index)
{
  return base + index;
}

template <BlockLayout Layout>
CUDA_CALLABLE_MEMBER inline size_t
block_inner_offset(size_t nt_idx, size_t nh_idx, size_t hd_idx, size_t nt, size_t nh, size_t hd)
{
  if constexpr (Layout == BlockLayout::NHD) {
    return ((nt_idx * nh) + nh_idx) * hd + hd_idx;
  } else {
    return ((nh_idx * nt) + nt_idx) * hd + hd_idx;
  }
}

// Choose a conservative grid size so every thread handles a roughly equal
// share of the work even when the total element count spans many blocks.
inline int
kvbm_kernels_compute_grid_dim(size_t total_elements, int block_dim)
{
  if (total_elements == 0) {
    return 0;
  }
  size_t blocks = (total_elements + static_cast<size_t>(block_dim) - 1) / static_cast<size_t>(block_dim);
  if (blocks == 0) {
    blocks = 1;
  }
  blocks = std::min<size_t>(blocks, 65535);
  return static_cast<int>(blocks);
}

// Returns the log2 shift amount if x is a non-zero power of 2, otherwise -1.
// Used on the host side to pre-compute whether kernel divisors can use
// cheap bit-shift/mask operations instead of expensive integer division.
// Example: po2_shift(64) returns 6, po2_shift(48) returns -1.
inline int
kvbm_kernels_po2_shift(size_t x)
{
  if (x == 0 || (x & (x - 1)) != 0)
    return -1;
  return __builtin_ctzll(static_cast<unsigned long long>(x));
}

// Flatten the [nh, nl, no, nt, hd] coordinates into a linear index so a single
// launch can cover many independent blocks in one pass.
template <typename T, BlockLayout Layout>
__global__ void
kvbm_kernels_block_to_universal_kernel(
    const T* const* block_chunks, T* const* universal_blocks, size_t block_stride, size_t total_per_block,
    size_t num_blocks, size_t nh, size_t nl, size_t no, size_t nt, size_t hd)
{
  size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  size_t total = total_per_block * num_blocks;

  while (thread_id < total) {
    size_t block_idx = thread_id / total_per_block;
    size_t residual = thread_id % total_per_block;

    size_t tmp = residual;
    size_t hd_idx = tmp % hd;
    tmp /= hd;

    size_t nt_idx = tmp % nt;
    tmp /= nt;

    size_t no_idx = tmp % no;
    tmp /= no;

    size_t nl_idx = tmp % nl;
    tmp /= nl;

    size_t nh_idx = tmp;

    const T* const* block_base = block_chunks + block_idx * block_stride;
    const T* chunk_base = block_base[nl_idx * no + no_idx];
    size_t chunk_offset = block_inner_offset<Layout>(nt_idx, nh_idx, hd_idx, nt, nh, hd);

    T* universal_base = universal_blocks[block_idx];
    universal_base[residual] = chunk_base[chunk_offset];
    thread_id += stride;
  }
}

// The inverse of kvbm_kernels_block_to_universal_kernel: peel apart the same linear index
// and scatter back into the layer/outer stacks.
template <typename T, BlockLayout Layout>
__global__ void
kvbm_kernels_universal_to_block_kernel(
    const T* const* universal_blocks, T* const* block_chunks, size_t block_stride, size_t total_per_block,
    size_t num_blocks, size_t nh, size_t nl, size_t no, size_t nt, size_t hd)
{
  size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  size_t total = total_per_block * num_blocks;

  while (thread_id < total) {
    size_t block_idx = thread_id / total_per_block;
    size_t residual = thread_id % total_per_block;

    size_t tmp = residual;
    size_t hd_idx = tmp % hd;
    tmp /= hd;

    size_t nt_idx = tmp % nt;
    tmp /= nt;

    size_t no_idx = tmp % no;
    tmp /= no;

    size_t nl_idx = tmp % nl;
    tmp /= nl;

    size_t nh_idx = tmp;

    T* const* block_base = const_cast<T* const*>(block_chunks + block_idx * block_stride);
    T* chunk_base = block_base[nl_idx * no + no_idx];
    size_t chunk_offset = block_inner_offset<Layout>(nt_idx, nh_idx, hd_idx, nt, nh, hd);

    const T* universal_base = universal_blocks[block_idx];
    chunk_base[chunk_offset] = universal_base[residual];
    thread_id += stride;
  }
}

// Pack or unpack the operational layout by striding across the flattened
// (nl * no) chunk table. chunk_elements == inner.
template <typename T>
__global__ void
kvbm_kernels_operational_pack_kernel(
    const T* const* block_chunks, T* const* operational_blocks, size_t block_stride, size_t chunk_elements,
    size_t total_per_block, size_t num_blocks)
{
  size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  size_t total = total_per_block * num_blocks;

  while (thread_id < total) {
    size_t block_idx = thread_id / total_per_block;
    size_t residual = thread_id % total_per_block;

    size_t chunk_idx = residual / chunk_elements;
    size_t inner_idx = residual % chunk_elements;

    const T* const* block_base = block_chunks + block_idx * block_stride;
    const T* chunk_ptr = block_base[chunk_idx];
    T* operational_base = operational_blocks[block_idx];

    operational_base[residual] = chunk_ptr[inner_idx];

    thread_id += stride;
  }
}

template <typename T>
__global__ void
kvbm_kernels_operational_unpack_kernel(
    const T* const* operational_blocks, T* const* block_chunks, size_t block_stride, size_t chunk_elements,
    size_t total_per_block, size_t num_blocks)
{
  size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  size_t total = total_per_block * num_blocks;

  while (thread_id < total) {
    size_t block_idx = thread_id / total_per_block;
    size_t residual = thread_id % total_per_block;

    size_t chunk_idx = residual / chunk_elements;
    size_t inner_idx = residual % chunk_elements;

    T* const* block_base = block_chunks + block_idx * block_stride;
    T* chunk_ptr = block_base[chunk_idx];
    const T* operational_base = operational_blocks[block_idx];

    chunk_ptr[inner_idx] = operational_base[residual];

    thread_id += stride;
  }
}

// Vectorized operational copy kernel using int64_t (8-byte) loads for maximum bandwidth.
// This kernel handles both pack (block->operational) and unpack (operational->block) directions.
// Inspired by LMCache's approach of using 64-bit vectorized memory access.
__global__ void
kvbm_kernels_operational_copy_vectorized_kernel(
    const int64_t* const* src_chunks, int64_t* const* dst_chunks,
    size_t chunk_stride,           // nl * no for block side
    size_t chunk_elements_64bit,   // inner * elem_size / 8
    size_t total_per_block_64bit,  // chunk_elements_64bit * chunk_count
    size_t num_blocks,
    bool pack_direction,  // true = block->operational (pack)
    int total_shift,      // log2(total_per_block_64bit) if power-of-2, else -1
    int chunk_shift)      // log2(chunk_elements_64bit) if power-of-2, else -1
{
  // Use 128 threads per block for better occupancy with 64-bit loads
  size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  size_t total = total_per_block_64bit * num_blocks;

  while (thread_id < total) {
    // Decompose flat thread_id into (block_idx, chunk_idx, inner_idx).
    // When the divisor is a power of 2 (shift >= 0), use bit shift/mask
    // instead of integer division — saves ~20-30 cycles per division on GPU.
    size_t block_idx, residual, chunk_idx, inner_idx;

    // Decompose: thread_id = block_idx * total_per_block_64bit + residual
    if (total_shift >= 0) {
      block_idx = thread_id >> total_shift;
      residual = thread_id & (total_per_block_64bit - 1);
    } else {
      block_idx = thread_id / total_per_block_64bit;
      residual = thread_id % total_per_block_64bit;
    }

    // Decompose: residual = chunk_idx * chunk_elements_64bit + inner_idx
    if (chunk_shift >= 0) {
      chunk_idx = residual >> chunk_shift;
      inner_idx = residual & (chunk_elements_64bit - 1);
    } else {
      chunk_idx = residual / chunk_elements_64bit;
      inner_idx = residual % chunk_elements_64bit;
    }

    if (pack_direction) {
      // Block -> Operational (pack)
      const int64_t* const* block_base = src_chunks + block_idx * chunk_stride;
      const int64_t* block_chunk = block_base[chunk_idx];
      int64_t* operational_base = dst_chunks[block_idx];
      operational_base[residual] = block_chunk[inner_idx];
    } else {
      // Operational -> Block (unpack)
      const int64_t* operational_base = src_chunks[block_idx];
      int64_t* const* block_base = dst_chunks + block_idx * chunk_stride;
      int64_t* block_chunk = block_base[chunk_idx];
      block_chunk[inner_idx] = operational_base[residual];
    }

    thread_id += stride;
  }
}

// Launch the vectorized operational copy kernel
cudaError_t
kvbm_kernels_launch_operational_copy_vectorized(
    void* const* operational_ptrs_device, void* const* block_ptrs_device, size_t num_blocks, size_t nl, size_t no,
    size_t inner_bytes, OperationalCopyDirection direction, cudaStream_t stream)
{
  size_t chunk_count = nl * no;
  size_t chunk_elements_64bit = inner_bytes / 8;
  size_t total_per_block_64bit = chunk_elements_64bit * chunk_count;
  size_t total = total_per_block_64bit * num_blocks;

  if (total == 0) {
    return cudaSuccess;
  }

  if (!operational_ptrs_device || !block_ptrs_device) {
    return cudaErrorInvalidValue;
  }

  // Use 128 threads for better occupancy with 64-bit loads (following LMCache pattern)
  constexpr int kBlockDim = 128;
  int grid_dim = kvbm_kernels_compute_grid_dim(total, kBlockDim);
  if (grid_dim == 0) {
    return cudaSuccess;
  }

  // Pre-compute shift amounts for power-of-2 fast path.
  // Common LLM dims (nh=8, nt=16/32/64, hd=64/128, elem_size=2/4) yield
  // power-of-2 chunk sizes, enabling bit shifts instead of integer division.
  int total_shift = kvbm_kernels_po2_shift(total_per_block_64bit);
  int chunk_shift = kvbm_kernels_po2_shift(chunk_elements_64bit);

  bool pack_direction = (direction == OperationalCopyDirection::BlockToOperational);

  if (pack_direction) {
    // Block -> Operational
    const int64_t* const* src = reinterpret_cast<const int64_t* const*>(block_ptrs_device);
    int64_t* const* dst = reinterpret_cast<int64_t* const*>(operational_ptrs_device);
    kvbm_kernels_operational_copy_vectorized_kernel<<<grid_dim, kBlockDim, 0, stream>>>(
        src, dst, chunk_count, chunk_elements_64bit, total_per_block_64bit, num_blocks, true, total_shift, chunk_shift);
  } else {
    // Operational -> Block
    const int64_t* const* src = reinterpret_cast<const int64_t* const*>(operational_ptrs_device);
    int64_t* const* dst = reinterpret_cast<int64_t* const*>(block_ptrs_device);
    kvbm_kernels_operational_copy_vectorized_kernel<<<grid_dim, kBlockDim, 0, stream>>>(
        src, dst, chunk_count, chunk_elements_64bit, total_per_block_64bit, num_blocks, false, total_shift,
        chunk_shift);
  }

  return cudaGetLastError();
}

template <typename T>
cudaError_t
kvbm_kernels_launch_block_to_universal_impl(
    void* const* universal_ptrs_device, const void* const* block_ptrs_device, size_t num_blocks, size_t nh, size_t nl,
    size_t no, size_t nt, size_t hd, BlockLayout layout, cudaStream_t stream)
{
  size_t block_stride = nl * no;
  size_t total_per_block = nh * nl * no * nt * hd;
  size_t total = total_per_block * num_blocks;
  if (total == 0) {
    return cudaSuccess;
  }

  if (!block_ptrs_device || !universal_ptrs_device) {
    return cudaErrorInvalidValue;
  }

  constexpr int kBlockDim = 256;
  int grid_dim = kvbm_kernels_compute_grid_dim(total, kBlockDim);
  if (grid_dim == 0) {
    return cudaSuccess;
  }

  const T* const* chunks = reinterpret_cast<const T* const*>(block_ptrs_device);
  T* const* universal_blocks = reinterpret_cast<T* const*>(const_cast<void* const*>(universal_ptrs_device));

  if (layout == BlockLayout::NHD) {
    kvbm_kernels_block_to_universal_kernel<T, BlockLayout::NHD><<<grid_dim, kBlockDim, 0, stream>>>(
        chunks, universal_blocks, block_stride, total_per_block, num_blocks, nh, nl, no, nt, hd);
  } else {
    kvbm_kernels_block_to_universal_kernel<T, BlockLayout::HND><<<grid_dim, kBlockDim, 0, stream>>>(
        chunks, universal_blocks, block_stride, total_per_block, num_blocks, nh, nl, no, nt, hd);
  }

  return cudaGetLastError();
}

template <typename T>
cudaError_t
kvbm_kernels_launch_block_from_universal_impl(
    const void* const* universal_ptrs_device, void* const* block_ptrs_device, size_t num_blocks, size_t nh, size_t nl,
    size_t no, size_t nt, size_t hd, BlockLayout layout, cudaStream_t stream)
{
  size_t block_stride = nl * no;
  size_t total_per_block = nh * nl * no * nt * hd;
  size_t total = total_per_block * num_blocks;
  if (total == 0) {
    return cudaSuccess;
  }

  if (!block_ptrs_device || !universal_ptrs_device) {
    return cudaErrorInvalidValue;
  }

  constexpr int kBlockDim = 256;
  int grid_dim = kvbm_kernels_compute_grid_dim(total, kBlockDim);
  if (grid_dim == 0) {
    return cudaSuccess;
  }

  const T* const* universal_blocks = reinterpret_cast<const T* const*>(universal_ptrs_device);
  T* const* chunks = reinterpret_cast<T* const*>(const_cast<void* const*>(block_ptrs_device));

  if (layout == BlockLayout::NHD) {
    kvbm_kernels_universal_to_block_kernel<T, BlockLayout::NHD><<<grid_dim, kBlockDim, 0, stream>>>(
        universal_blocks, chunks, block_stride, total_per_block, num_blocks, nh, nl, no, nt, hd);
  } else {
    kvbm_kernels_universal_to_block_kernel<T, BlockLayout::HND><<<grid_dim, kBlockDim, 0, stream>>>(
        universal_blocks, chunks, block_stride, total_per_block, num_blocks, nh, nl, no, nt, hd);
  }

  return cudaGetLastError();
}

template <typename T>
cudaError_t
kvbm_kernels_launch_operational_copy_impl(
    void* const* operational_ptrs_device, void* const* block_ptrs_device, size_t num_blocks, size_t nl, size_t no,
    size_t inner, OperationalCopyDirection direction, cudaStream_t stream)
{
  size_t chunk_count = nl * no;
  if (chunk_count == 0 || inner == 0 || num_blocks == 0) {
    return cudaSuccess;
  }

  if (!operational_ptrs_device || !block_ptrs_device) {
    return cudaErrorInvalidValue;
  }

  constexpr int kBlockDim = 256;
  size_t chunk_elements = inner;
  size_t total_per_block = chunk_elements * chunk_count;
  size_t total = total_per_block * num_blocks;
  int grid_dim = kvbm_kernels_compute_grid_dim(total, kBlockDim);
  if (grid_dim == 0) {
    return cudaSuccess;
  }

  T* const* operational_blocks = reinterpret_cast<T* const*>(const_cast<void* const*>(operational_ptrs_device));

  if (direction == OperationalCopyDirection::BlockToOperational) {
    const T* const* block_chunks = reinterpret_cast<const T* const*>(block_ptrs_device);
    kvbm_kernels_operational_pack_kernel<T><<<grid_dim, kBlockDim, 0, stream>>>(
        block_chunks, operational_blocks, chunk_count, chunk_elements, total_per_block, num_blocks);
  } else {
    T* const* block_chunks = reinterpret_cast<T* const*>(block_ptrs_device);
    kvbm_kernels_operational_unpack_kernel<T><<<grid_dim, kBlockDim, 0, stream>>>(
        reinterpret_cast<const T* const*>(operational_ptrs_device), block_chunks, chunk_count, chunk_elements,
        total_per_block, num_blocks);
  }

  return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t
kvbm_kernels_launch_universal_from_block(
    void* const* universal_ptrs_device, const void* const* block_ptrs_device, size_t num_blocks, size_t nh, size_t nl,
    size_t no, size_t nt, size_t hd, int dtype_value, int layout_value, cudaStream_t stream)
{
  auto dtype = static_cast<TensorDataType>(dtype_value);
  auto layout = static_cast<BlockLayout>(layout_value);

  switch (dtype) {
    case TensorDataType::F16:
      return kvbm_kernels_launch_block_to_universal_impl<typename DTypeTraits<TensorDataType::F16>::type>(
          universal_ptrs_device, block_ptrs_device, num_blocks, nh, nl, no, nt, hd, layout, stream);
    case TensorDataType::BF16:
      return kvbm_kernels_launch_block_to_universal_impl<typename DTypeTraits<TensorDataType::BF16>::type>(
          universal_ptrs_device, block_ptrs_device, num_blocks, nh, nl, no, nt, hd, layout, stream);
    case TensorDataType::F32:
      return kvbm_kernels_launch_block_to_universal_impl<typename DTypeTraits<TensorDataType::F32>::type>(
          universal_ptrs_device, block_ptrs_device, num_blocks, nh, nl, no, nt, hd, layout, stream);
    case TensorDataType::F64:
      return kvbm_kernels_launch_block_to_universal_impl<typename DTypeTraits<TensorDataType::F64>::type>(
          universal_ptrs_device, block_ptrs_device, num_blocks, nh, nl, no, nt, hd, layout, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

extern "C" cudaError_t
kvbm_kernels_launch_block_from_universal(
    const void* const* universal_ptrs_device, void* const* block_ptrs_device, size_t num_blocks, size_t nh, size_t nl,
    size_t no, size_t nt, size_t hd, int dtype_value, int layout_value, cudaStream_t stream)
{
  auto dtype = static_cast<TensorDataType>(dtype_value);
  auto layout = static_cast<BlockLayout>(layout_value);

  switch (dtype) {
    case TensorDataType::F16:
      return kvbm_kernels_launch_block_from_universal_impl<typename DTypeTraits<TensorDataType::F16>::type>(
          universal_ptrs_device, block_ptrs_device, num_blocks, nh, nl, no, nt, hd, layout, stream);
    case TensorDataType::BF16:
      return kvbm_kernels_launch_block_from_universal_impl<typename DTypeTraits<TensorDataType::BF16>::type>(
          universal_ptrs_device, block_ptrs_device, num_blocks, nh, nl, no, nt, hd, layout, stream);
    case TensorDataType::F32:
      return kvbm_kernels_launch_block_from_universal_impl<typename DTypeTraits<TensorDataType::F32>::type>(
          universal_ptrs_device, block_ptrs_device, num_blocks, nh, nl, no, nt, hd, layout, stream);
    case TensorDataType::F64:
      return kvbm_kernels_launch_block_from_universal_impl<typename DTypeTraits<TensorDataType::F64>::type>(
          universal_ptrs_device, block_ptrs_device, num_blocks, nh, nl, no, nt, hd, layout, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

enum class OperationalCopyBackend : int {
  Auto = 0,
  VectorizedKernel = 1,  // Force vectorized 64-bit kernel
  KernelOnly = 2,        // Force dtype-specific kernel
  MemcpyAsync = 3,
  MemcpyBatch = 4,
};

extern "C" cudaError_t
kvbm_kernels_launch_operational_copy(
    const void* const* block_ptrs_host, const void* const* block_ptrs_device, void* const* operational_ptrs_host,
    void* const* operational_ptrs_device, size_t num_blocks, size_t nl, size_t no, size_t inner, size_t elem_size,
    int dtype_value, int direction_value, int backend_value, cudaStream_t stream)
{
  auto direction = static_cast<OperationalCopyDirection>(direction_value);
  auto dtype = static_cast<TensorDataType>(dtype_value);
  auto backend = static_cast<OperationalCopyBackend>(backend_value);

  size_t chunk_count = nl * no;
  size_t chunk_bytes = inner * elem_size;
  size_t total_chunks = num_blocks * chunk_count;

  if (chunk_count == 0 || chunk_bytes == 0 || num_blocks == 0) {
    return cudaSuccess;
  }

  // Host pointers are always required
  if (!block_ptrs_host || !operational_ptrs_host) {
    return cudaErrorInvalidValue;
  }

  // Device pointers only required for kernel backends (VectorizedKernel, KernelOnly, Auto)
  // MemcpyAsync and MemcpyBatch use host pointers only
  bool needs_device_ptrs =
      (backend == OperationalCopyBackend::VectorizedKernel || backend == OperationalCopyBackend::KernelOnly ||
       backend == OperationalCopyBackend::Auto);
  if (needs_device_ptrs && (!block_ptrs_device || !operational_ptrs_device)) {
    return cudaErrorInvalidValue;
  }

  std::vector<void*> dst_ptrs(total_chunks);
  std::vector<const void*> src_ptrs(total_chunks);
  std::vector<size_t> sizes(total_chunks, chunk_bytes);

  for (size_t block = 0; block < num_blocks; ++block) {
    auto operational_base = static_cast<std::uint8_t*>(const_cast<void*>(operational_ptrs_host[block]));
    for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
      size_t idx = block * chunk_count + chunk;
      auto operational_ptr = operational_base + chunk * chunk_bytes;
      if (direction == OperationalCopyDirection::BlockToOperational) {
        dst_ptrs[idx] = operational_ptr;
        src_ptrs[idx] = block_ptrs_host[idx];
      } else {
        dst_ptrs[idx] = const_cast<void*>(block_ptrs_host[idx]);
        src_ptrs[idx] = operational_ptr;
      }
    }
  }

  auto launch_kernel = [&]() -> cudaError_t {
    if (!block_ptrs_device) {
      return cudaSuccess;
    }
    switch (dtype) {
      case TensorDataType::F16:
        return kvbm_kernels_launch_operational_copy_impl<typename DTypeTraits<TensorDataType::F16>::type>(
            operational_ptrs_device, const_cast<void* const*>(block_ptrs_device), num_blocks, nl, no, inner, direction,
            stream);
      case TensorDataType::BF16:
        return kvbm_kernels_launch_operational_copy_impl<typename DTypeTraits<TensorDataType::BF16>::type>(
            operational_ptrs_device, const_cast<void* const*>(block_ptrs_device), num_blocks, nl, no, inner, direction,
            stream);
      case TensorDataType::F32:
        return kvbm_kernels_launch_operational_copy_impl<typename DTypeTraits<TensorDataType::F32>::type>(
            operational_ptrs_device, const_cast<void* const*>(block_ptrs_device), num_blocks, nl, no, inner, direction,
            stream);
      case TensorDataType::F64:
        return kvbm_kernels_launch_operational_copy_impl<typename DTypeTraits<TensorDataType::F64>::type>(
            operational_ptrs_device, const_cast<void* const*>(block_ptrs_device), num_blocks, nl, no, inner, direction,
            stream);
      default:
        return cudaErrorInvalidValue;
    }
  };

  auto launch_memcpy_async = [&]() -> cudaError_t {
    for (size_t idx = 0; idx < total_chunks; ++idx) {
      // Use cudaMemcpyDefault to auto-detect direction from pointer types.
      // This enables H2D, D2H, and D2D transfers without explicit direction.
      cudaError_t err = cudaMemcpyAsync(dst_ptrs[idx], src_ptrs[idx], sizes[idx], cudaMemcpyDefault, stream);
      if (err != cudaSuccess) {
        return err;
      }
    }
    return cudaSuccess;
  };

  auto launch_memcpy_batch = [&]() -> cudaError_t {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12090
    std::vector<void*> src_ptrs_mut(total_chunks);
    for (size_t idx = 0; idx < total_chunks; ++idx) {
      src_ptrs_mut[idx] = const_cast<void*>(src_ptrs[idx]);
    }

    // attrIdxList must have one entry per copy, mapping each to an attribute.
    std::vector<size_t> attr_indices(total_chunks, 0);
    cudaMemcpyAttributes attr = {};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;

#if CUDART_VERSION >= 13000
    // CUDA 13.0+: 8-parameter API (no failIdx)
    return cudaMemcpyBatchAsync(
        dst_ptrs.data(), src_ptrs_mut.data(), sizes.data(), total_chunks, &attr, attr_indices.data(), 1, stream);
#else
    // CUDA 12.9: 9-parameter API (with failIdx)
    size_t fail_idx = 0;
    return cudaMemcpyBatchAsync(
        dst_ptrs.data(), src_ptrs_mut.data(), sizes.data(), total_chunks, &attr, attr_indices.data(), 1, &fail_idx,
        stream);
#endif
#else
    return cudaErrorNotSupported;
#endif
  };

  // Check if data is 8-byte aligned for vectorized kernel
  size_t total_bytes = inner * elem_size;
  bool is_8byte_aligned = (total_bytes % 8 == 0) && block_ptrs_device;

  auto launch_vectorized = [&]() -> cudaError_t {
    if (!is_8byte_aligned || !block_ptrs_device) {
      return cudaErrorNotSupported;
    }
    return kvbm_kernels_launch_operational_copy_vectorized(
        operational_ptrs_device, const_cast<void* const*>(block_ptrs_device), num_blocks, nl, no, total_bytes,
        direction, stream);
  };

  cudaError_t status = cudaErrorInvalidValue;
  switch (backend) {
    case OperationalCopyBackend::VectorizedKernel:
      status = launch_vectorized();
      break;
    case OperationalCopyBackend::KernelOnly:
      status = launch_kernel();
      break;
    case OperationalCopyBackend::MemcpyAsync:
      status = launch_memcpy_async();
      break;
    case OperationalCopyBackend::MemcpyBatch:
      status = launch_memcpy_batch();
      if (status == cudaErrorNotSupported || status == cudaErrorInvalidValue) {
        status = launch_memcpy_async();
      }
      break;
    case OperationalCopyBackend::Auto:
    default:
      // Priority: vectorized kernel (if 8-byte aligned) -> batch copy (CUDA 12.9+) -> memcpy async
      if (is_8byte_aligned) {
        status = launch_vectorized();
        if (status == cudaSuccess) {
          break;
        }
      }
      status = launch_memcpy_batch();
      if (status == cudaErrorNotSupported || status == cudaErrorInvalidValue) {
        status = launch_memcpy_async();
      }
      break;
  }

  return status;
}

/// Check if cudaMemcpyBatchAsync is available at compile time.
/// Returns true if CUDA 12.9+ was used to compile this library.
extern "C" bool
kvbm_kernels_has_memcpy_batch_async()
{
#if CUDART_VERSION >= 12090
  return true;
#else
  return false;
#endif
}

/// Controls how kvbm_kernels_memcpy_batch dispatches copies.
enum class MemcpyBatchMode : int {
  BatchedWithFallback = 0,   // Try cudaMemcpyBatchAsync, fall back to individual cudaMemcpyAsync on failure
  FallbackOnly = 1,          // Only use individual cudaMemcpyAsync loop (never attempt batch API)
  BatchWithoutFallback = 2,  // Try cudaMemcpyBatchAsync, return error on failure (no fallback)
};

/// Batched memcpy using cudaMemcpyBatchAsync (CUDA 12.9+) and/or individual cudaMemcpyAsync.
///
/// Takes HOST arrays of src/dst pointers - no device allocation needed.
/// Direction is auto-determined by CUDA from pointer types using cudaMemcpyDefault.
///
/// @param src_ptrs_host Host array of source pointers
/// @param dst_ptrs_host Host array of destination pointers
/// @param size_per_copy Size in bytes for each copy
/// @param num_copies Number of copies to perform
/// @param mode_value Controls dispatch: 0 = BatchedWithFallback, 1 = FallbackOnly, 2 = BatchWithoutFallback
/// @param stream CUDA stream for async execution
/// @return cudaSuccess on success, cudaErrorNotSupported if batch API unavailable and mode disallows fallback
extern "C" cudaError_t
kvbm_kernels_memcpy_batch(
    const void* const* src_ptrs_host, void* const* dst_ptrs_host, size_t size_per_copy, size_t num_copies,
    int mode_value, cudaStream_t stream)
{
  auto mode = static_cast<MemcpyBatchMode>(mode_value);

  if (num_copies == 0 || size_per_copy == 0) {
    return cudaSuccess;
  }

  if (!src_ptrs_host || !dst_ptrs_host) {
    return cudaErrorInvalidValue;
  }

  auto launch_memcpy_async_fallback = [&]() -> cudaError_t {
    for (size_t i = 0; i < num_copies; ++i) {
      cudaError_t copy_err =
          cudaMemcpyAsync(dst_ptrs_host[i], src_ptrs_host[i], size_per_copy, cudaMemcpyDefault, stream);
      if (copy_err != cudaSuccess) {
        return copy_err;
      }
    }
    return cudaSuccess;
  };

  // FallbackOnly: skip batch entirely, always use individual cudaMemcpyAsync
  if (mode == MemcpyBatchMode::FallbackOnly) {
    return launch_memcpy_async_fallback();
  }

#if defined(CUDART_VERSION)
#if CUDART_VERSION >= 12090
  std::vector<size_t> sizes(num_copies, size_per_copy);
  std::vector<void*> src_ptrs_mut(num_copies);
  for (size_t i = 0; i < num_copies; ++i) {
    src_ptrs_mut[i] = const_cast<void*>(src_ptrs_host[i]);
  }
  // attrIdxList must have one entry per copy, mapping each to an attribute.
  // We use a single attribute (index 0) for all copies.
  std::vector<size_t> attr_indices(num_copies, 0);
  cudaMemcpyAttributes attr = {};
  attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;

#if CUDART_VERSION >= 13000
  // CUDA 13.0+: 8-parameter API (no failIdx)
  cudaError_t err = cudaMemcpyBatchAsync(
      const_cast<void**>(dst_ptrs_host), src_ptrs_mut.data(), sizes.data(), num_copies, &attr, attr_indices.data(), 1,
      stream);
#else
  // CUDA 12.9: 9-parameter API (with failIdx)
  size_t fail_idx = 0;
  cudaError_t err = cudaMemcpyBatchAsync(
      const_cast<void**>(dst_ptrs_host), src_ptrs_mut.data(), sizes.data(), num_copies, &attr, attr_indices.data(), 1,
      &fail_idx, stream);
#endif

  if (err == cudaErrorNotSupported || err == cudaErrorInvalidValue) {
    if (mode == MemcpyBatchMode::BatchWithoutFallback) {
      return err;
    }
#ifdef KVBM_TENSOR_KERNELS_DEBUG
    fprintf(
        stderr, "cudaMemcpyBatchAsync failed with error %d (%s), falling back to individual cudaMemcpyAsync\n",
        (int)err, cudaGetErrorString(err));
#endif
    return launch_memcpy_async_fallback();
  }
  return err;

#else
  // CUDA < 12.9: batch API not available at compile time
  if (mode == MemcpyBatchMode::BatchWithoutFallback) {
    return cudaErrorNotSupported;
  }
#pragma message("CUDA < 12.9: Fallback to individual cudaMemcpyAsync with cudaMemcpyDefault")
  return launch_memcpy_async_fallback();
#endif
#else
  // CUDART_VERSION not defined
  if (mode == MemcpyBatchMode::BatchWithoutFallback) {
    return cudaErrorNotSupported;
  }
  return launch_memcpy_async_fallback();
#endif
}

/// Returns false - this is the real CUDA implementation, not stubs.
/// Downstream crates can use this to skip CUDA tests at runtime when stubs are linked.
extern "C" bool
kvbm_kernels_is_stub_build()
{
  return false;
}

/// Vectorized memory copy kernel for arbitrary device-visible pointer pairs.
///
/// Each block handles one or more (src, dst) pairs using a grid-strided loop.
/// Per-pair alignment detection selects the widest safe vector width:
///   - 16-byte (int4) if both pointers are 16-byte aligned
///   - 8-byte  (int2) if both pointers are 8-byte aligned
///   - 4-byte  (int)  if both pointers are 4-byte aligned
///   - 1-byte fallback for any remainder
///
/// Source and destination pointers may be device memory or pinned host memory —
/// any memory reachable via CUDA unified addressing is valid.
__global__ void
kvbm_kernels_vectorized_copy_kernel(void** src_ptrs, void** dst_ptrs, size_t copy_size_in_bytes, int num_pairs)
{
  int pair_id = blockIdx.x;
  int block_stride = gridDim.x;
  int tid = threadIdx.x;
  int block_size = blockDim.x;

  for (; pair_id < num_pairs; pair_id += block_stride) {
    char* src = static_cast<char*>(src_ptrs[pair_id]);
    char* dst = static_cast<char*>(dst_ptrs[pair_id]);

    // Check alignment for THIS specific pair (all threads in block see same values)
    uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
    uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);

    size_t vectorized_bytes = 0;

    if (((src_addr & 0xF) == 0) && ((dst_addr & 0xF) == 0) && (copy_size_in_bytes >= 16)) {
      // Best case: 16-byte vectorized copy using int4
      size_t num_int4 = copy_size_in_bytes >> 4;
      for (size_t i = tid; i < num_int4; i += block_size) {
        reinterpret_cast<int4*>(dst)[i] = reinterpret_cast<const int4*>(src)[i];
      }
      vectorized_bytes = num_int4 << 4;
    } else if (((src_addr & 0x7) == 0) && ((dst_addr & 0x7) == 0) && (copy_size_in_bytes >= 8)) {
      // 8-byte vectorized copy using int2 (matches LMCache int64_t approach)
      size_t num_int2 = copy_size_in_bytes >> 3;
      for (size_t i = tid; i < num_int2; i += block_size) {
        reinterpret_cast<int2*>(dst)[i] = reinterpret_cast<const int2*>(src)[i];
      }
      vectorized_bytes = num_int2 << 3;
    } else if (((src_addr & 0x3) == 0) && ((dst_addr & 0x3) == 0) && (copy_size_in_bytes >= 4)) {
      // 4-byte vectorized copy
      size_t num_int = copy_size_in_bytes >> 2;
      for (size_t i = tid; i < num_int; i += block_size) {
        reinterpret_cast<int*>(dst)[i] = reinterpret_cast<const int*>(src)[i];
      }
      vectorized_bytes = num_int << 2;
    }

    // Handle remaining bytes (from vectorized remainder or full scalar fallback)
    size_t remaining = copy_size_in_bytes - vectorized_bytes;
    for (size_t i = tid; i < remaining; i += block_size) {
      dst[vectorized_bytes + i] = src[vectorized_bytes + i];
    }
  }
}

/// Launch the vectorized copy kernel for copying between arbitrary pointer pairs.
/// This kernel automatically selects optimal vectorization (4/8/16 bytes) based on alignment.
///
/// @param src_ptrs Device pointer to array of source pointers
/// @param dst_ptrs Device pointer to array of destination pointers
/// @param copy_size_bytes Size of each copy in bytes
/// @param num_pairs Number of pointer pairs to copy
/// @param stream CUDA stream for async execution
extern "C" cudaError_t
kvbm_kernels_launch_vectorized_copy(
    void** src_ptrs_device, void** dst_ptrs_device, size_t copy_size_bytes, int num_pairs, cudaStream_t stream)
{
  if (num_pairs == 0 || copy_size_bytes == 0) {
    return cudaSuccess;
  }

  if (!src_ptrs_device || !dst_ptrs_device) {
    return cudaErrorInvalidValue;
  }

  // Use 128 threads per block, one block per pair (up to 65535 blocks)
  constexpr int kBlockDim = 128;
  int grid_dim = std::min(num_pairs, 65535);

  kvbm_kernels_vectorized_copy_kernel<<<grid_dim, kBlockDim, 0, stream>>>(
      src_ptrs_device, dst_ptrs_device, copy_size_bytes, num_pairs);

  return cudaGetLastError();
}
