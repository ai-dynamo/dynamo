// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// Error checking macro
#define CUDA_CHECK(call)                                                                            \
  do {                                                                                              \
    cudaError_t error = call;                                                                       \
    if (error != cudaSuccess) {                                                                     \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
      return error;                                                                                 \
    }                                                                                               \
  } while (0)

// Number of elements to process per thread
#define ELEMENTS_PER_THREAD 4

// Use cache-line sized chunks when possible
#define CACHE_LINE_SIZE 128  // 128 bytes for most GPUs

// Optimized kernel that processes elements in a dimension-aware manner
__global__ void
copy_blocks_kernel(
    const void* src_data, void* dst_data, const int* src_block_ids, const int* dst_block_ids, int num_block_pairs,
    int prefix_dim, int suffix_dim, int elem_size, size_t src_prefix_stride, size_t src_block_stride,
    size_t src_suffix_stride, size_t dst_prefix_stride, size_t dst_block_stride, size_t dst_suffix_stride)
{
  // Calculate the total number of elements to process
  const size_t total_elements = (size_t)prefix_dim * num_block_pairs * suffix_dim;

  // Calculate the total number of bytes in the suffix part
  const size_t bytes_per_suffix = (size_t)suffix_dim * elem_size;

  // Calculate how many cache-line sized chunks per suffix part
  const size_t chunks_per_suffix = (bytes_per_suffix + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE;
  const size_t elements_per_chunk = CACHE_LINE_SIZE / elem_size;
  const bool is_perfect_chunk = (bytes_per_suffix % CACHE_LINE_SIZE) == 0;

  // Get global thread index
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread processes ELEMENTS_PER_THREAD chunk indices
  const size_t start_chunk = thread_idx * ELEMENTS_PER_THREAD;
  const size_t total_chunks = prefix_dim * num_block_pairs * chunks_per_suffix;

  // Early exit if completely out of range
  if (start_chunk >= total_chunks) {
    return;
  }

  // Process multiple chunks per thread
  for (int chunk_offset = 0; chunk_offset < ELEMENTS_PER_THREAD; chunk_offset++) {
    // Current chunk index
    size_t chunk_idx = start_chunk + chunk_offset;

    // Check if this chunk is within bounds
    if (chunk_idx >= total_chunks) {
      return;  // No more chunks to process
    }

    // Decompose chunk index into prefix, block, and suffix chunks
    size_t blocks_chunks = num_block_pairs * chunks_per_suffix;
    size_t prefix_idx = chunk_idx / blocks_chunks;
    size_t remainder = chunk_idx % blocks_chunks;
    size_t block_pair_idx = remainder / chunks_per_suffix;
    size_t chunk_in_suffix = remainder % chunks_per_suffix;

    // Bounds check
    if (prefix_idx >= prefix_dim || block_pair_idx >= num_block_pairs) {
      continue;  // Skip this chunk
    }

    // Get the actual source and destination block IDs
    int src_block_id = src_block_ids[block_pair_idx];
    int dst_block_id = dst_block_ids[block_pair_idx];

    // Calculate element offset within the suffix dimension
    size_t suffix_elem_offset = chunk_in_suffix * CACHE_LINE_SIZE / elem_size;

    // Calculate the byte offset using explicit strides for each dimension
    size_t src_byte_offset =
        prefix_idx * src_prefix_stride + src_block_id * src_block_stride + suffix_elem_offset * src_suffix_stride;

    size_t dst_byte_offset =
        prefix_idx * dst_prefix_stride + dst_block_id * dst_block_stride + suffix_elem_offset * dst_suffix_stride;

    // Calculate elements to copy in this chunk
    size_t elements_to_copy = elements_per_chunk;
    if (!is_perfect_chunk && chunk_in_suffix == chunks_per_suffix - 1) {
      // Last chunk might be smaller
      elements_to_copy = suffix_dim - suffix_elem_offset;
    }

    // Copy data based on element size for better performance
    if (elem_size == 2 && (elements_to_copy % 2 == 0)) {
      // Use 32-bit loads/stores for 16-bit data when possible (half precision)
      const uint32_t* src_ptr = (const uint32_t*)((const char*)src_data + src_byte_offset);
      uint32_t* dst_ptr = (uint32_t*)((char*)dst_data + dst_byte_offset);

      for (size_t i = 0; i < elements_to_copy / 2; i++) {
        dst_ptr[i] = src_ptr[i];
      }
    } else if (elem_size == 2) {
      // Handle 16-bit elements one by one if necessary
      const uint16_t* src_ptr = (const uint16_t*)((const char*)src_data + src_byte_offset);
      uint16_t* dst_ptr = (uint16_t*)((char*)dst_data + dst_byte_offset);

      for (size_t i = 0; i < elements_to_copy; i++) {
        dst_ptr[i] = src_ptr[i];
      }
    } else if (elem_size == 4) {
      // Copy 32-bit elements (float, int32)
      const uint32_t* src_ptr = (const uint32_t*)((const char*)src_data + src_byte_offset);
      uint32_t* dst_ptr = (uint32_t*)((char*)dst_data + dst_byte_offset);

      for (size_t i = 0; i < elements_to_copy; i++) {
        dst_ptr[i] = src_ptr[i];
      }
    } else if (elem_size == 8) {
      // Copy 64-bit elements (double, int64)
      const uint64_t* src_ptr = (const uint64_t*)((const char*)src_data + src_byte_offset);
      uint64_t* dst_ptr = (uint64_t*)((char*)dst_data + dst_byte_offset);

      for (size_t i = 0; i < elements_to_copy; i++) {
        dst_ptr[i] = src_ptr[i];
      }
    } else {
      // For other element sizes, copy byte by byte
      const char* src_ptr = (const char*)src_data + src_byte_offset;
      char* dst_ptr = (char*)dst_data + dst_byte_offset;

      for (size_t i = 0; i < elements_to_copy * elem_size; i++) {
        dst_ptr[i] = src_ptr[i];
      }
    }
  }
}

// Simplified launcher that uses the 3D tensor view
extern "C" cudaError_t
copy_blocks_launcher_3d(
    const void* src_data, void* dst_data, const int* h_src_block_ids, const int* h_dst_block_ids, int num_block_pairs,
    int prefix_dim, int suffix_dim, int elem_size, size_t src_prefix_stride, size_t src_block_stride,
    size_t src_suffix_stride, size_t dst_prefix_stride, size_t dst_block_stride, size_t dst_suffix_stride,
    int* d_src_block_ids, int* d_dst_block_ids, cudaEvent_t event, cudaStream_t stream)
{
  // Validate inputs
  if (src_data == NULL || dst_data == NULL) {
    fprintf(stderr, "NULL data pointers\n");
    return cudaErrorInvalidValue;
  }

  if (d_src_block_ids == NULL || d_dst_block_ids == NULL) {
    fprintf(stderr, "NULL device block ID pointers\n");
    return cudaErrorInvalidValue;
  }

  if (num_block_pairs <= 0) {
    fprintf(stderr, "Invalid number of block pairs: %d\n", num_block_pairs);
    return cudaErrorInvalidValue;
  }

  if (prefix_dim <= 0 || suffix_dim <= 0 || elem_size <= 0) {
    fprintf(stderr, "Invalid dimensions: prefix=%d, suffix=%d, elem=%d\n", prefix_dim, suffix_dim, elem_size);
    return cudaErrorInvalidValue;
  }

  // Calculate total number of bytes to copy
  size_t total_bytes = (size_t)prefix_dim * num_block_pairs * suffix_dim * elem_size;

  // Calculate number of cache-line sized chunks
  size_t bytes_per_suffix = (size_t)suffix_dim * elem_size;
  size_t chunks_per_suffix = (bytes_per_suffix + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE;
  size_t total_chunks = prefix_dim * num_block_pairs * chunks_per_suffix;

  // Adjust grid size to account for multiple elements per thread
  int total_threads = (total_chunks + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
  int cuda_block_size = 256;
  int grid_size = (total_threads + cuda_block_size - 1) / cuda_block_size;

  // Validate grid size
  if (grid_size <= 0) {
    fprintf(stderr, "Invalid grid size: %d\n", grid_size);
    return cudaErrorInvalidValue;
  }

  // Launch kernel on specified stream
  copy_blocks_kernel<<<grid_size, cuda_block_size, 0, stream>>>(
      src_data, dst_data, h_src_block_ids, h_dst_block_ids, num_block_pairs, prefix_dim, suffix_dim, elem_size,
      src_prefix_stride, src_block_stride, src_suffix_stride, dst_prefix_stride, dst_block_stride, dst_suffix_stride);

  // Check for kernel launch errors immediately
  cudaError_t kernel_error = cudaGetLastError();
  if (kernel_error != cudaSuccess) {
    fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(kernel_error));
    return kernel_error;
  }

  // Record event on the stream (for synchronization by caller)
  if (event != NULL) {
    CUDA_CHECK(cudaEventRecord(event, stream));
  }

  return cudaSuccess;
}

// New function for 3D tensor copy blocks operation
extern "C" cudaError_t
copy_blocks_3d(
    const void* src_data, void* dst_data, const int* h_src_block_ids, const int* h_dst_block_ids, int num_block_pairs,
    int prefix_dim, int src_blocks, int dst_blocks, int suffix_dim, int elem_size)
{
  // Calculate row-major strides internally
  size_t src_suffix_stride = elem_size;
  size_t dst_suffix_stride = elem_size;

  size_t src_block_stride = suffix_dim * src_suffix_stride;
  size_t dst_block_stride = suffix_dim * dst_suffix_stride;

  size_t src_prefix_stride = src_blocks * src_block_stride;
  size_t dst_prefix_stride = dst_blocks * dst_block_stride;

  // Optional debug output
  printf(
      "Tensor dims: prefix=%d, src_blocks=%d, dst_blocks=%d, suffix=%d, elem_size=%d\n", prefix_dim, src_blocks,
      dst_blocks, suffix_dim, elem_size);
  printf(
      "Calculated strides: src_prefix=%zu, src_block=%zu, src_suffix=%zu\n", src_prefix_stride, src_block_stride,
      src_suffix_stride);

  // Allocate device memory for block IDs
  int* d_src_blocks_ids = NULL;
  int* d_dst_blocks_ids = NULL;

  CUDA_CHECK(cudaMalloc(&d_src_blocks_ids, num_block_pairs * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_dst_blocks_ids, num_block_pairs * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_src_blocks_ids, h_src_block_ids, num_block_pairs * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dst_blocks_ids, h_dst_block_ids, num_block_pairs * sizeof(int), cudaMemcpyHostToDevice));

  // Create CUDA event
  cudaEvent_t event;
  CUDA_CHECK(cudaEventCreate(&event));

  // Launch kernel with explicit strides
  cudaError_t result = copy_blocks_launcher_3d(
      src_data, dst_data, h_src_block_ids, h_dst_block_ids, num_block_pairs, prefix_dim, suffix_dim, elem_size,
      src_prefix_stride, src_block_stride, src_suffix_stride, dst_prefix_stride, dst_block_stride, dst_suffix_stride,
      d_src_blocks_ids, d_dst_blocks_ids, event, 0);

  // Handle errors from kernel launch
  if (result != cudaSuccess) {
    cudaFree(d_src_blocks_ids);
    cudaFree(d_dst_blocks_ids);
    cudaEventDestroy(event);
    return result;
  }

  // Wait for completion
  CUDA_CHECK(cudaEventSynchronize(event));

  // Clean up
  cudaFree(d_src_blocks_ids);
  cudaFree(d_dst_blocks_ids);
  cudaEventDestroy(event);

  printf("3D tensor block copy completed successfully\n");
  return cudaSuccess;
}

// TODO: Refactor the driver code to take pointers for the device block_id arrays
// TODO: Maintain a blocking driver, but then also provide a non-blocking driver
//
// We will have N copies of the BlockCopyControl struct which we will put in a reusable
// pool. Acquiring a BlockCopyControl will let you perform a copy for a kv attention layer.
//
// From rust or python we'll execute this on a thread allowed to block. We'll await the
// cuda event for completion and report the return code on the driver.
//
// TODO: decide whether or not we need a pool of streams or use a single stream.
//
// We should be able to decouple this from the forward pass. The only condition is that
// a new forward pass can not start until the last copy has completed.
//
// To that end, we might want to tie this copy kernel to the stream used for the forward pass.
struct BlockCopyControl {
  int* d_src_blocks;
  int* d_dst_blocks;
  cudaEvent_t start_event;
  cudaEvent_t stop_event;

  BlockCopyControl(int num_blocks);
  ~BlockCopyControl();

  void reset();
};

BlockCopyControl::BlockCopyControl(int num_blocks)
{
  cudaError_t status;
  status = cudaMalloc(&d_src_blocks, num_blocks * sizeof(int));
  if (status != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
    return;
  }

  status = cudaMalloc(&d_dst_blocks, num_blocks * sizeof(int));
  if (status != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
    cudaFree(d_src_blocks);
    return;
  }

  status = cudaEventCreateWithFlags(&start_event, cudaEventDisableTiming);
  if (status != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
    cudaFree(d_src_blocks);
    cudaFree(d_dst_blocks);
  }

  status = cudaEventCreateWithFlags(&stop_event, cudaEventDisableTiming);
  if (status != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
    cudaFree(d_src_blocks);
    cudaFree(d_dst_blocks);
  }
}

BlockCopyControl::~BlockCopyControl()
{
  cudaFree(d_src_blocks);
  cudaFree(d_dst_blocks);
  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);
}


extern "C" {
int cuda_malloc_host(void** ptr, size_t size);
int cuda_free_host(void* ptr);
int cuda_memcpy_async(void* dst, const void* src, size_t count, cudaStream_t stream);

int
cuda_malloc_host(void** ptr, size_t size)
{
  CUDA_CHECK(cudaMallocHost(ptr, size));
  return cudaSuccess;
}

int
cuda_free_host(void* ptr)
{
  CUDA_CHECK(cudaFreeHost(ptr));
  return cudaSuccess;
}

int
cuda_memcpy_async(void* dst, const void* src, size_t count, cudaStream_t stream)
{
  CUDA_CHECK(cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream));
  return cudaSuccess;
}
}
