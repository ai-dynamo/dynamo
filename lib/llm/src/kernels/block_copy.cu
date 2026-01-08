// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstring>
#include <memory>
#include <vector>

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
      // } else if (elem_size == 1 && (elements_to_copy % 4 == 0)) {
      //   // Use 32-bit loads/stores for 8-bit data when possible (half precision)
      //   const uint32_t* src_ptr = (const uint32_t*)((const char*)src_data + src_byte_offset);
      //   uint32_t* dst_ptr = (uint32_t*)((char*)dst_data + dst_byte_offset);

      //   for (size_t i = 0; i < elements_to_copy / 4; i++) {
      //     dst_ptr[i] = src_ptr[i];
      //   }
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
    const void* src_data, void* dst_data, const int* d_src_block_ids, const int* d_dst_block_ids, int num_block_pairs,
    int prefix_dim, int suffix_dim, int elem_size, int src_block_dim, int dst_block_dim, cudaStream_t stream)
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

  // Calculate row-major strides internally
  size_t src_suffix_stride = elem_size;
  size_t dst_suffix_stride = elem_size;

  size_t src_block_stride = suffix_dim * src_suffix_stride;
  size_t dst_block_stride = suffix_dim * dst_suffix_stride;

  size_t src_prefix_stride = src_block_dim * src_block_stride;
  size_t dst_prefix_stride = dst_block_dim * dst_block_stride;

  // // Optional debug output
  // printf(
  //     "Tensor dims: prefix=%d, src_blocks=%d, dst_blocks=%d, suffix=%d, elem_size=%d\n", prefix_dim, src_blocks_dim,
  //     dst_blocks_dim, suffix_dim, elem_size);
  // printf(
  //     "Calculated strides: src_prefix=%zu, src_block=%zu, src_suffix=%zu\n", src_prefix_stride, src_block_stride,
  //     src_suffix_stride);

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
      src_data, dst_data, d_src_block_ids, d_dst_block_ids, num_block_pairs, prefix_dim, suffix_dim, elem_size,
      src_prefix_stride, src_block_stride, src_suffix_stride, dst_prefix_stride, dst_block_stride, dst_suffix_stride);

  // Check for kernel launch errors immediately
  cudaError_t kernel_error = cudaGetLastError();
  if (kernel_error != cudaSuccess) {
    fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(kernel_error));
    return kernel_error;
  }

  return cudaSuccess;
}


extern "C" cudaError_t
copy_blocks_memcpy_3d(
    const void* src_data, void* dst_data, const int* h_src_block_ids, const int* h_dst_block_ids, int num_block_pairs,
    int prefix_dim, int suffix_dim, int elem_size, int src_block_dim, int dst_block_dim, cudaStream_t stream)
{
  // Validate inputs
  if (src_data == NULL || dst_data == NULL) {
    fprintf(stderr, "NULL data pointers\n");
    return cudaErrorInvalidValue;
  }

  if (h_src_block_ids == NULL || h_dst_block_ids == NULL) {
    fprintf(stderr, "NULL host block ID pointers\n");
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

  // Calculate row-major strides for source and destination
  size_t suffix_size_bytes = suffix_dim * elem_size;
  size_t src_block_stride = suffix_size_bytes;
  size_t dst_block_stride = suffix_size_bytes;
  size_t src_prefix_stride = src_block_dim * src_block_stride;
  size_t dst_prefix_stride = dst_block_dim * dst_block_stride;

  size_t count = 0;

  // Loop through all prefix dimensions and block pairs
  for (int prefix_idx = 0; prefix_idx < prefix_dim; prefix_idx++) {
    for (int pair_idx = 0; pair_idx < num_block_pairs; pair_idx++) {
      int src_block_id = h_src_block_ids[pair_idx];
      int dst_block_id = h_dst_block_ids[pair_idx];

      // Calculate byte offsets
      size_t src_offset = prefix_idx * src_prefix_stride + src_block_id * src_block_stride;
      size_t dst_offset = prefix_idx * dst_prefix_stride + dst_block_id * dst_block_stride;

      // Copy the suffix data in one call (it's contiguous)
      const void* src_ptr = static_cast<const char*>(src_data) + src_offset;
      void* dst_ptr = static_cast<char*>(dst_data) + dst_offset;

      cudaError_t error = cudaMemcpyAsync(dst_ptr, src_ptr, suffix_size_bytes, cudaMemcpyDefault, stream);
      if (error != cudaSuccess) {
        return error;
      }

      count += suffix_size_bytes;
    }
  }

  return cudaSuccess;
}


// New function for 3D tensor copy blocks operation
extern "C" cudaError_t
copy_blocks_3d(
    const void* src_data, void* dst_data, const int* h_src_block_ids, const int* h_dst_block_ids, int num_block_pairs,
    int prefix_dim, int src_blocks_dim, int dst_blocks_dim, int suffix_dim, int elem_size)
{
#ifdef USE_KERNEL
  // Allocate device memory for block IDs
  int* d_src_block_ids = NULL;
  int* d_dst_block_ids = NULL;

  CUDA_CHECK(cudaMalloc(&d_src_block_ids, num_block_pairs * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_dst_block_ids, num_block_pairs * sizeof(int)));

  CUDA_CHECK(
      cudaMemcpyAsync(d_src_block_ids, h_src_block_ids, num_block_pairs * sizeof(int), cudaMemcpyHostToDevice, 0));
  CUDA_CHECK(
      cudaMemcpyAsync(d_dst_block_ids, h_dst_block_ids, num_block_pairs * sizeof(int), cudaMemcpyHostToDevice, 0));

  // Launch kernel with explicit strides
  cudaError_t result = copy_blocks_launcher_3d(
      src_data, dst_data, d_src_block_ids, d_dst_block_ids, num_block_pairs, prefix_dim, suffix_dim, elem_size,
      src_blocks_dim, dst_blocks_dim, 0);

  // Handle errors from kernel launch
  if (result != cudaSuccess) {
    cudaFree(d_src_block_ids);
    cudaFree(d_dst_block_ids);
    return result;
  }
#else
  cudaError_t result = copy_blocks_memcpy_3d(
      src_data, dst_data, h_src_block_ids, h_dst_block_ids, num_block_pairs, prefix_dim, suffix_dim, elem_size,
      src_blocks_dim, dst_blocks_dim, 0);
#endif
  // Wait for completion
  CUDA_CHECK(cudaStreamSynchronize(0));

#ifdef USE_KERNEL
  // Clean up
  cudaFree(d_src_block_ids);
  cudaFree(d_dst_block_ids);
#endif

  return cudaSuccess;
}


// TODO: Refactor the driver code to take pointers for the device block_id arrays
// TODO: Maintain a blocking driver, but then also provide a non-blocking driver
//
// We will have N copies of the CopyStream struct which we will put in a reusable
// pool. Acquiring a CopyStream will let you perform a copy for a kv attention layer.
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
struct CopyStream {
  // Device block arrays
  int* d_src_blocks;
  int* d_dst_blocks;

  // Host copies of block arrays
  int* h_src_blocks;
  int* h_dst_blocks;

  int num_blocks;

  cudaStream_t stream;
  cudaEvent_t start_event;
  cudaEvent_t stop_event;

  CopyStream(int num_layers, int num_blocks);
  ~CopyStream();

  void reset();
};

CopyStream::CopyStream(int num_layers, int num_blocks)
{
  cudaError_t status;

  // Allocate device memory
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

  // Allocate host memory
  h_src_blocks = (int*)malloc(num_blocks * sizeof(int));
  h_dst_blocks = (int*)malloc(num_blocks * sizeof(int));
  if (!h_src_blocks || !h_dst_blocks) {
    fprintf(stderr, "Host memory allocation failed\n");
    if (h_src_blocks)
      free(h_src_blocks);
    cudaFree(d_src_blocks);
    cudaFree(d_dst_blocks);
    return;
  }

  status = cudaStreamCreate(&stream);
  if (status != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
    free(h_src_blocks);
    free(h_dst_blocks);
    cudaFree(d_src_blocks);
    cudaFree(d_dst_blocks);
    return;
  }

  // Create events
  status = cudaEventCreateWithFlags(&start_event, cudaEventDisableTiming);
  if (status != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
    free(h_src_blocks);
    free(h_dst_blocks);
    cudaFree(d_src_blocks);
    cudaFree(d_dst_blocks);
  }

  status = cudaEventCreateWithFlags(&stop_event, cudaEventDisableTiming);
  if (status != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
    free(h_src_blocks);
    free(h_dst_blocks);
    cudaFree(d_src_blocks);
    cudaFree(d_dst_blocks);
  }
}

CopyStream::~CopyStream()
{
  free(h_src_blocks);
  free(h_dst_blocks);
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
copy_stream_create(CopyStream** stream, int num_layers, int num_blocks)
{
  *stream = new CopyStream(num_layers, num_blocks);
  return 0;
}

int
copy_stream_destroy(CopyStream* stream)
{
  delete stream;
  return 0;
}


int
copy_stream_prepare_block_ids(CopyStream* cs, int* src_block_ids, int* dst_block_ids, int num_blocks)
{
  // Make host copies
  memcpy(cs->h_src_blocks, src_block_ids, num_blocks * sizeof(int));
  memcpy(cs->h_dst_blocks, dst_block_ids, num_blocks * sizeof(int));

  // Copy to device (for kernel-based implementation)
  CUDA_CHECK(
      cudaMemcpyAsync(cs->d_src_blocks, src_block_ids, num_blocks * sizeof(int), cudaMemcpyHostToDevice, cs->stream));
  CUDA_CHECK(
      cudaMemcpyAsync(cs->d_dst_blocks, dst_block_ids, num_blocks * sizeof(int), cudaMemcpyHostToDevice, cs->stream));

  cs->num_blocks = num_blocks;

  return 0;
}

int
copy_stream_launch(
    CopyStream* cs, const void* src_data, void* dst_data, int prefix_dim, int suffix_dim, int elem_size,
    int src_block_dim, int dst_block_dim)
{
  return copy_blocks_launcher_3d(
      src_data, dst_data, cs->d_src_blocks, cs->d_dst_blocks, cs->num_blocks, prefix_dim, suffix_dim, elem_size,
      src_block_dim, dst_block_dim, cs->stream);
}

int
copy_stream_memcpy(
    CopyStream* cs, const void* src_data, void* dst_data, int prefix_dim, int suffix_dim, int elem_size,
    int src_block_dim, int dst_block_dim)
{
  return copy_blocks_memcpy_3d(
      src_data, dst_data, cs->h_src_blocks, cs->h_dst_blocks, cs->num_blocks, prefix_dim, suffix_dim, elem_size,
      src_block_dim, dst_block_dim, cs->stream);
}

int
copy_stream_sync(CopyStream* cs)
{
  // sync on the event
  CUDA_CHECK(cudaStreamSynchronize(cs->stream));
  return cudaSuccess;
}

int
cuda_malloc_host(void** ptr, size_t size)
{
  CUDA_CHECK(cudaHostAlloc(ptr, size, cudaHostAllocDefault));
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

int
cuda_memcpy_sync(void* dst, const void* src, size_t count)
{
  CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDefault));
  return cudaSuccess;
}
}

/// This accepts a 6D tensor with dimensions that represent a tensor to be distributed
/// across tensor parallel ranks.
///
/// The dimensions of the source tensor are expected to be:
/// dims[0]: kv or block (depending on KvLayout)
/// dims[1]: block or kv (depending on KvLayout)
/// dims[2]: block_size (sequence length) # aka bs
/// dims[3]: scatter_factor (dst_tp_size / src_tp_size)
/// dims[4]: num_heads / (src_tp_size * scatter_factor) # aka dst_num_heads or dnh
/// dims[5]: head_size # aka hs
///
/// The permutation applied is (3, 0, 1, 2, 4, 5) which transforms
/// the tensor:
///  - from: [kv/block, block/kv, bs, scatter_factor, dnh, hs] to
///  - to:   [scatter_factor, kv/block, block/kv, bs, dnh, hs].
///
/// This transformation effectively distributes the heads dimension across
/// tensor parallel ranks, where we transform from src_tp_size to dst_tp_size,
/// with dst_tp_size > src_tp_size.
int
permute_scatter_memcpy(
    const void* src,           // source data
    void* dst,                 // destination data
    const uint32_t* dims,      // 6d dimensions of source tensor
    uint32_t num_dims,         // semi-redundant, size of the dims array, must be 6
    uint32_t elem_size,        // element size in bytes
    uint32_t block_dim_index,  // which dimension represents blocks
    uint32_t src_block_dim,    // the dimension of the source blocks
    uint32_t dst_block_dim,    // the dimension of the destination blocks
    int* src_block_ids,        // from state: the block IDs to copy
    int* dst_block_ids,        // from state: the block IDs to copy
    uint32_t num_blocks,       // from state: the number of blocks to copy
    cudaStream_t stream        // from state: the stream to use
)
{
  if (num_dims != 6) {
    printf("ERROR: num_dims must be 6\n");
    return -1;
  }

  if (block_dim_index != 0 && block_dim_index != 1) {
    printf("ERROR: block_dim_index must be 0 or 1\n");
    return -2;
  }

  uint32_t kv_dim_index = block_dim_index == 0 ? 1 : 0;

  // expect dims[block_dim_index] == src_block_dim
  // expect dims[kv_dim_index] == 2
  if (dims[block_dim_index] != src_block_dim) {
    printf("ERROR: dims[block_dim_index] must be equal to src_block_dim\n");
    return -3;
  }

  if (dims[kv_dim_index] != 2) {
    printf("ERROR: dims[kv_dim_index] must be 2\n");
    return -4;
  }

  size_t src_shape[5];
  size_t dst_shape[5];

  src_shape[block_dim_index] = src_block_dim;
  src_shape[kv_dim_index] = dims[kv_dim_index];
  src_shape[2] = dims[2];
  src_shape[3] = dims[3];
  src_shape[4] = dims[4] * dims[5];

  dst_shape[0] = dims[3];  // scatter factor
  dst_shape[block_dim_index + 1] = dst_block_dim;
  dst_shape[kv_dim_index + 1] = dims[kv_dim_index];
  dst_shape[3] = dims[2];  // block size
  dst_shape[4] = dims[4] * dims[5];

  size_t src_strides[5];
  size_t dst_strides[5];

  src_strides[4] = elem_size;
  dst_strides[4] = elem_size;

  // Compute source strides recursively (row-major order)
  for (int i = 3; i >= 0; i--) {
    src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
  }

  // Compute destination strides based on permuted dimensions
  for (int i = 3; i >= 0; i--) {
    dst_strides[i] = dst_strides[i + 1] * dst_shape[i + 1];
  }

#ifdef DEBUG
  printf("src_shape: ");
  for (int i = 0; i < 5; i++) {
    printf("%zu ", src_shape[i]);
  }
  printf("\n");

  printf("src_strides: ");
  for (int i = 0; i < 5; i++) {
    printf("%zu ", src_strides[i]);
  }
  printf("\n");

  printf("dst_shape: ");
  for (int i = 0; i < 5; i++) {
    printf("%zu ", dst_shape[i]);
  }
  printf("\n");

  printf("dst_strides: ");
  for (int i = 0; i < 5; i++) {
    printf("%zu ", dst_strides[i]);
  }
  printf("\n");
#endif

  size_t copy_size_bytes = dims[4] * dims[5] * elem_size;

  // we will start by computing the full offsets for each inner copy blocks
  size_t src_idx[5];
  size_t dst_idx[5];

  // notes:
  // - in the outer two loops, the index for the dst is shifted by one since we moved the
  //   scatter dimension to the front [0]

  const char* src_ptr = (const char*)src;
  char* dst_ptr = (char*)dst;

  // loop over blocks
  for (int block = 0; block < num_blocks; block++) {
    src_idx[block_dim_index] = block;
    dst_idx[block_dim_index + 1] = block;
    // loop over the kv dimension
    for (int kv = 0; kv < src_shape[kv_dim_index]; kv++) {
      src_idx[kv_dim_index] = kv;
      dst_idx[kv_dim_index + 1] = kv;
      // loop over block size
      for (int block_size = 0; block_size < src_shape[2]; block_size++) {
        src_idx[2] = block_size;
        dst_idx[3] = block_size;
        // loop over scatter factor
        for (int scatter = 0; scatter < src_shape[3]; scatter++) {
          src_idx[3] = scatter;
          dst_idx[0] = scatter;

          src_idx[4] = 0;
          dst_idx[4] = 0;

          size_t src_offset = 0;
          size_t dst_offset = 0;

          for (int i = 0; i < 5; i++) {
            src_offset += src_idx[i] * src_strides[i];
            dst_offset += dst_idx[i] * dst_strides[i];
          }

          auto rc =
              cudaMemcpyAsync(dst_ptr + dst_offset, src_ptr + src_offset, copy_size_bytes, cudaMemcpyDefault, stream);

          if (rc != cudaSuccess) {
            printf("ERROR: cudaMemcpyAsync failed with error code %d\n", rc);
            return -5;
          }
        }
      }
    }
  }

  return 0;
}

// Updated C API wrapper for the permutation function
extern "C" int
copy_stream_scatter(
    CopyStream* cs,            // the copy stream
    const void* src_data,      // the source data (single layer)
    void* dst_data,            // the destination data (single layer)
    const uint32_t* dims,      // 6d dimensions of source tensor
    uint32_t num_dims,         // semi-redundant, size of the dims array, must be 6
    uint32_t elem_size,        // element size in bytes
    uint32_t block_dim_index,  // which dimension represents blocks; either 0 or 1
    uint32_t src_block_dim,    // number of blocks in the src tensor (should match dims[block_dim_index])
    uint32_t dst_block_dim     // number of blocks in the dst tensor
)
{
  return permute_scatter_memcpy(
      src_data,          //
      dst_data,          //
      dims,              //
      num_dims,          //
      elem_size,         //
      block_dim_index,   //
      src_block_dim,     //
      dst_block_dim,     //
      cs->h_src_blocks,  //
      cs->h_dst_blocks,  //
      cs->num_blocks,    //
      cs->stream         //
  );
}
