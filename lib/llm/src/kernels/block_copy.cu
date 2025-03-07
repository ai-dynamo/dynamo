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

// Optimized kernel that processes multiple elements per thread
__global__ void
copy_blocks_kernel(
    const void* src_data, void* dst_data, const int* src_block_ids, const int* dst_block_ids, int num_blocks,
    int kv_size, int block_size, int heads_per_rank, int head_size, int elem_size, size_t src_tp_stride,
    size_t src_block_stride, size_t src_pos_stride, size_t src_head_stride, size_t dst_tp_stride,
    size_t dst_block_stride, size_t dst_pos_stride, size_t dst_head_stride)
{
  // Get global thread index
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate total elements per block
  int elements_per_block = block_size * heads_per_rank * head_size;

  // Total elements to process
  int total_elements = elements_per_block * num_blocks * kv_size;

  // Calculate the starting element index for this thread
  int start_element = thread_idx * ELEMENTS_PER_THREAD;

  // Process multiple elements per thread
  for (int e = 0; e < ELEMENTS_PER_THREAD; e++) {
    // Current element index
    int elem_idx = start_element + e;

    // Check if this element is within bounds
    if (elem_idx >= total_elements) {
      return;  // No more elements to process
    }

    // Calculate which element we're processing
    int combined_idx = elem_idx / elements_per_block;
    int element_offset = elem_idx % elements_per_block;

    // Split combined_idx into rank and block
    int rank = combined_idx / num_blocks;       // KV rank (0 or 1)
    int block_idx = combined_idx % num_blocks;  // Block index in the mapping list

    // Bounds check on block indices
    if (block_idx < 0 || block_idx >= num_blocks) {
      continue;  // Skip this element
    }

    // Get source and destination block IDs
    int src_block_id = src_block_ids[block_idx];
    int dst_block_id = dst_block_ids[block_idx];

    // Calculate position indices
    int pos = element_offset / (heads_per_rank * head_size);
    int remaining = element_offset % (heads_per_rank * head_size);
    int head = remaining / head_size;
    int head_offset = remaining % head_size;

    // Bounds check on all indices
    if (pos >= block_size || head >= heads_per_rank || head_offset >= head_size) {
      continue;  // Skip this element
    }

    // Calculate source offset using provided strides
    size_t src_offset = rank * src_tp_stride + src_block_id * src_block_stride + pos * src_pos_stride +
                        head * src_head_stride + head_offset;

    // Calculate destination offset
    size_t dst_offset = rank * dst_tp_stride + dst_block_id * dst_block_stride + pos * dst_pos_stride +
                        head * dst_head_stride + head_offset;

    // Perform type-optimized copy based on element size
    if (elem_size == 2) {
      // For 16-bit elements (half/bfloat16/uint16)
      const uint16_t* src_ptr = (const uint16_t*)src_data + src_offset;
      uint16_t* dst_ptr = (uint16_t*)dst_data + dst_offset;
      *dst_ptr = *src_ptr;
    } else if (elem_size == 4) {
      // For 32-bit elements (float/int32)
      const uint32_t* src_ptr = (const uint32_t*)src_data + src_offset;
      uint32_t* dst_ptr = (uint32_t*)dst_data + dst_offset;
      *dst_ptr = *src_ptr;
    } else if (elem_size == 8) {
      // For 64-bit elements (double/int64)
      const uint64_t* src_ptr = (const uint64_t*)src_data + src_offset;
      uint64_t* dst_ptr = (uint64_t*)dst_data + dst_offset;
      *dst_ptr = *src_ptr;
    } else {
      // For other element sizes, copy byte by byte
      const char* src_bytes = (const char*)src_data + src_offset * elem_size;
      char* dst_bytes = (char*)dst_data + dst_offset * elem_size;

      // Copy element using proper size
      for (int i = 0; i < elem_size; i++) {
        dst_bytes[i] = src_bytes[i];
      }
    }
  }
}

// Host-callable function
extern "C" cudaError_t
copy_blocks(
    const void* src_data, void* dst_data, const int* src_block_ids, int num_src_blocks, const int* dst_block_ids,
    int num_dst_blocks, int src_n_blocks, int dst_n_blocks, int kv_size, int block_size, int heads_per_rank,
    int head_size, int elem_size, size_t src_tp_stride, size_t src_block_stride, size_t src_pos_stride,
    size_t src_head_stride, size_t dst_tp_stride, size_t dst_block_stride, size_t dst_pos_stride,
    size_t dst_head_stride)
{
  // Validate inputs
  if (src_data == NULL || dst_data == NULL) {
    fprintf(stderr, "NULL data pointers\n");
    return cudaErrorInvalidValue;
  }

  if (src_block_ids == NULL || dst_block_ids == NULL) {
    fprintf(stderr, "NULL block ID pointers\n");
    return cudaErrorInvalidValue;
  }

  if (num_src_blocks != num_dst_blocks || num_src_blocks <= 0) {
    fprintf(stderr, "Block list issue: src=%d, dst=%d\n", num_src_blocks, num_dst_blocks);
    return cudaErrorInvalidValue;
  }

  if (kv_size <= 0 || block_size <= 0 || heads_per_rank <= 0 || head_size <= 0 || elem_size <= 0) {
    fprintf(
        stderr, "Invalid dimensions: tp=%d, block=%d, heads=%d, head_size=%d, elem=%d\n", kv_size, block_size,
        heads_per_rank, head_size, elem_size);
    return cudaErrorInvalidValue;
  }

  // Copy block IDs to device
  int* d_src_blocks = NULL;
  int* d_dst_blocks = NULL;

  // TODO: Create class/struct with preallocated memory for block_ids and two cuda events
  CUDA_CHECK(cudaMalloc(&d_src_blocks, num_src_blocks * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_dst_blocks, num_dst_blocks * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_src_blocks, src_block_ids, num_src_blocks * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dst_blocks, dst_block_ids, num_dst_blocks * sizeof(int), cudaMemcpyHostToDevice));

  // Calculate grid dimensions with ELEMENTS_PER_THREAD adjustment
  int elements_per_block = block_size * heads_per_rank * head_size;
  int total_elements = elements_per_block * num_src_blocks * kv_size;

  // Adjust grid size to account for multiple elements per thread
  int total_threads = (total_elements + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
  int cuda_block_size = 256;
  int grid_size = (total_threads + cuda_block_size - 1) / cuda_block_size;

  // Validate grid size
  if (grid_size <= 0) {
    fprintf(stderr, "Invalid grid size: %d\n", grid_size);
    cudaFree(d_src_blocks);
    cudaFree(d_dst_blocks);
    return cudaErrorInvalidValue;
  }

  // Print debug information
  printf("Starting kernel: blocks=%d, threads=%d, total elements=%d\n", grid_size, cuda_block_size, total_elements);
  printf("Elements per thread: %d, Total threads: %d\n", ELEMENTS_PER_THREAD, total_threads);
  printf(
      "Dimensions: tp=%d, blocks=%d, size=%d, heads=%d, headsize=%d, elemsize=%d\n", kv_size, num_src_blocks,
      block_size, heads_per_rank, head_size, elem_size);

  // Launch kernel
  copy_blocks_kernel<<<grid_size, cuda_block_size>>>(
      src_data, dst_data, d_src_blocks, d_dst_blocks, num_src_blocks, kv_size, block_size, heads_per_rank, head_size,
      elem_size, src_tp_stride, src_block_stride, src_pos_stride, src_head_stride, dst_tp_stride, dst_block_stride,
      dst_pos_stride, dst_head_stride);

  // Check for kernel errors immediately
  cudaError_t kernel_error = cudaGetLastError();
  if (kernel_error != cudaSuccess) {
    fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(kernel_error));
    cudaFree(d_src_blocks);
    cudaFree(d_dst_blocks);
    return kernel_error;
  }

  // Wait for completion
  CUDA_CHECK(cudaDeviceSynchronize());

  // Clean up
  cudaFree(d_src_blocks);
  cudaFree(d_dst_blocks);

  printf("Kernel execution completed successfully\n");
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

  status = cudaEventCreate(&start_event);
  if (status != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
    cudaFree(d_src_blocks);
    cudaFree(d_dst_blocks);
  }

  status = cudaEventCreate(&stop_event);
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
}
