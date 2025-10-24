// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

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
compute_grid_dim(size_t total_elements, int block_dim)
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

// Flatten the [nh, nl, no, nt, hd] coordinates into a linear index so a single
// launch can cover many independent blocks in one pass.
template <typename T, BlockLayout Layout>
__global__ void
block_to_universal_kernel(
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

// The inverse of block_to_universal_kernel: peel apart the same linear index
// and scatter back into the layer/outer stacks.
template <typename T, BlockLayout Layout>
__global__ void
universal_to_block_kernel(
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
operational_pack_kernel(
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
operational_unpack_kernel(
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

template <typename T>
cudaError_t
launch_block_to_universal_impl(
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
  int grid_dim = compute_grid_dim(total, kBlockDim);
  if (grid_dim == 0) {
    return cudaSuccess;
  }

  const T* const* chunks = reinterpret_cast<const T* const*>(block_ptrs_device);
  T* const* universal_blocks = reinterpret_cast<T* const*>(const_cast<void* const*>(universal_ptrs_device));

  if (layout == BlockLayout::NHD) {
    block_to_universal_kernel<T, BlockLayout::NHD><<<grid_dim, kBlockDim, 0, stream>>>(
        chunks, universal_blocks, block_stride, total_per_block, num_blocks, nh, nl, no, nt, hd);
  } else {
    block_to_universal_kernel<T, BlockLayout::HND><<<grid_dim, kBlockDim, 0, stream>>>(
        chunks, universal_blocks, block_stride, total_per_block, num_blocks, nh, nl, no, nt, hd);
  }

  return cudaGetLastError();
}

template <typename T>
cudaError_t
launch_block_from_universal_impl(
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
  int grid_dim = compute_grid_dim(total, kBlockDim);
  if (grid_dim == 0) {
    return cudaSuccess;
  }

  const T* const* universal_blocks = reinterpret_cast<const T* const*>(universal_ptrs_device);
  T* const* chunks = reinterpret_cast<T* const*>(const_cast<void* const*>(block_ptrs_device));

  if (layout == BlockLayout::NHD) {
    universal_to_block_kernel<T, BlockLayout::NHD><<<grid_dim, kBlockDim, 0, stream>>>(
        universal_blocks, chunks, block_stride, total_per_block, num_blocks, nh, nl, no, nt, hd);
  } else {
    universal_to_block_kernel<T, BlockLayout::HND><<<grid_dim, kBlockDim, 0, stream>>>(
        universal_blocks, chunks, block_stride, total_per_block, num_blocks, nh, nl, no, nt, hd);
  }

  return cudaGetLastError();
}

template <typename T>
cudaError_t
launch_operational_copy_impl(
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
  int grid_dim = compute_grid_dim(total, kBlockDim);
  if (grid_dim == 0) {
    return cudaSuccess;
  }

  T* const* operational_blocks = reinterpret_cast<T* const*>(const_cast<void* const*>(operational_ptrs_device));

  if (direction == OperationalCopyDirection::BlockToOperational) {
    const T* const* block_chunks = reinterpret_cast<const T* const*>(block_ptrs_device);
    operational_pack_kernel<T><<<grid_dim, kBlockDim, 0, stream>>>(
        block_chunks, operational_blocks, chunk_count, chunk_elements, total_per_block, num_blocks);
  } else {
    T* const* block_chunks = reinterpret_cast<T* const*>(block_ptrs_device);
    operational_unpack_kernel<T><<<grid_dim, kBlockDim, 0, stream>>>(
        reinterpret_cast<const T* const*>(operational_ptrs_device), block_chunks, chunk_count, chunk_elements,
        total_per_block, num_blocks);
  }

  return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t
launch_universal_from_block(
    void* const* universal_ptrs_device, const void* const* block_ptrs_device, size_t num_blocks, size_t nh, size_t nl,
    size_t no, size_t nt, size_t hd, int dtype_value, int layout_value, cudaStream_t stream)
{
  auto dtype = static_cast<TensorDataType>(dtype_value);
  auto layout = static_cast<BlockLayout>(layout_value);

  switch (dtype) {
    case TensorDataType::F16:
      return launch_block_to_universal_impl<typename DTypeTraits<TensorDataType::F16>::type>(
          universal_ptrs_device, block_ptrs_device, num_blocks, nh, nl, no, nt, hd, layout, stream);
    case TensorDataType::BF16:
      return launch_block_to_universal_impl<typename DTypeTraits<TensorDataType::BF16>::type>(
          universal_ptrs_device, block_ptrs_device, num_blocks, nh, nl, no, nt, hd, layout, stream);
    case TensorDataType::F32:
      return launch_block_to_universal_impl<typename DTypeTraits<TensorDataType::F32>::type>(
          universal_ptrs_device, block_ptrs_device, num_blocks, nh, nl, no, nt, hd, layout, stream);
    case TensorDataType::F64:
      return launch_block_to_universal_impl<typename DTypeTraits<TensorDataType::F64>::type>(
          universal_ptrs_device, block_ptrs_device, num_blocks, nh, nl, no, nt, hd, layout, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

extern "C" cudaError_t
launch_block_from_universal(
    const void* const* universal_ptrs_device, void* const* block_ptrs_device, size_t num_blocks, size_t nh, size_t nl,
    size_t no, size_t nt, size_t hd, int dtype_value, int layout_value, cudaStream_t stream)
{
  auto dtype = static_cast<TensorDataType>(dtype_value);
  auto layout = static_cast<BlockLayout>(layout_value);

  switch (dtype) {
    case TensorDataType::F16:
      return launch_block_from_universal_impl<typename DTypeTraits<TensorDataType::F16>::type>(
          universal_ptrs_device, block_ptrs_device, num_blocks, nh, nl, no, nt, hd, layout, stream);
    case TensorDataType::BF16:
      return launch_block_from_universal_impl<typename DTypeTraits<TensorDataType::BF16>::type>(
          universal_ptrs_device, block_ptrs_device, num_blocks, nh, nl, no, nt, hd, layout, stream);
    case TensorDataType::F32:
      return launch_block_from_universal_impl<typename DTypeTraits<TensorDataType::F32>::type>(
          universal_ptrs_device, block_ptrs_device, num_blocks, nh, nl, no, nt, hd, layout, stream);
    case TensorDataType::F64:
      return launch_block_from_universal_impl<typename DTypeTraits<TensorDataType::F64>::type>(
          universal_ptrs_device, block_ptrs_device, num_blocks, nh, nl, no, nt, hd, layout, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

enum class OperationalCopyBackend : int {
  Auto = 0,
  KernelOnly = 1,
  MemcpyAsync = 2,
  MemcpyBatch = 3,
};

extern "C" cudaError_t
launch_operational_copy(
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

  if (!block_ptrs_host || !operational_ptrs_host || !operational_ptrs_device) {
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
        return launch_operational_copy_impl<typename DTypeTraits<TensorDataType::F16>::type>(
            operational_ptrs_device, const_cast<void* const*>(block_ptrs_device), num_blocks, nl, no, inner, direction,
            stream);
      case TensorDataType::BF16:
        return launch_operational_copy_impl<typename DTypeTraits<TensorDataType::BF16>::type>(
            operational_ptrs_device, const_cast<void* const*>(block_ptrs_device), num_blocks, nl, no, inner, direction,
            stream);
      case TensorDataType::F32:
        return launch_operational_copy_impl<typename DTypeTraits<TensorDataType::F32>::type>(
            operational_ptrs_device, const_cast<void* const*>(block_ptrs_device), num_blocks, nl, no, inner, direction,
            stream);
      case TensorDataType::F64:
        return launch_operational_copy_impl<typename DTypeTraits<TensorDataType::F64>::type>(
            operational_ptrs_device, const_cast<void* const*>(block_ptrs_device), num_blocks, nl, no, inner, direction,
            stream);
      default:
        return cudaErrorInvalidValue;
    }
  };

  auto launch_memcpy_async = [&]() -> cudaError_t {
    for (size_t idx = 0; idx < total_chunks; ++idx) {
      cudaError_t err = cudaMemcpyAsync(dst_ptrs[idx], src_ptrs[idx], sizes[idx], cudaMemcpyDeviceToDevice, stream);
      if (err != cudaSuccess) {
        return err;
      }
    }
    return cudaSuccess;
  };

  auto launch_memcpy_batch = [&]() -> cudaError_t {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12090
    std::vector<void*> src_mut(total_chunks);
    for (size_t i = 0; i < total_chunks; ++i) {
      src_mut[i] = const_cast<void*>(src_ptrs[i]);
    }
    size_t fail_idx = 0;
    return cudaMemcpyBatchAsync(
        const_cast<void**>(dst_ptrs.data()), src_mut.data(), const_cast<size_t*>(sizes.data()), total_chunks, nullptr,
        nullptr, 0, &fail_idx, stream);
#else
    return cudaErrorNotSupported;
#endif
  };

  cudaError_t status = cudaErrorInvalidValue;
  switch (backend) {
    case OperationalCopyBackend::KernelOnly:
      status = launch_kernel();
      break;
    case OperationalCopyBackend::MemcpyAsync:
      status = launch_memcpy_async();
      break;
    case OperationalCopyBackend::MemcpyBatch:
      status = launch_memcpy_batch();
      break;
    case OperationalCopyBackend::Auto:
    default:
      status = launch_kernel();
      if (status == cudaErrorNotSupported || status == cudaErrorInvalidValue) {
        status = launch_memcpy_batch();
      }
      if (status == cudaErrorNotSupported || status == cudaErrorInvalidValue) {
        status = launch_memcpy_async();
      }
      break;
  }

  return status;
}
