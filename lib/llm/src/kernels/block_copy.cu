// block_copy.cu
#include <cuda_runtime.h>
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

// CRITICAL FIX: Ensure proper bounds checking in the kernel
__global__ void
copy_blocks_kernel(
    const void* src_data, void* dst_data, const int* src_block_ids, const int* dst_block_ids, int num_blocks,
    int kv_size, int block_size, int heads_per_rank, int head_size, int elem_size, size_t src_tp_stride,
    size_t src_block_stride, size_t src_pos_stride, size_t src_head_stride, size_t dst_tp_stride,
    size_t dst_block_stride, size_t dst_pos_stride, size_t dst_head_stride)
{
  // Get global thread index
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate total elements per block
  int elements_per_block = block_size * heads_per_rank * head_size;

  // Total elements to process
  int total_elements = elements_per_block * num_blocks * kv_size;

  // Check if this thread is within bounds
  if (idx >= total_elements) {
    return;
  }

  // Calculate which element we're processing
  // FIX: Ensure proper division
  int combined_idx = idx / elements_per_block;
  int element_offset = idx % elements_per_block;

  // Split combined_idx into rank and block
  int rank = combined_idx / num_blocks;       // KV rank (0 or 1)
  int block_idx = combined_idx % num_blocks;  // Block index in the mapping list

  // CRITICAL FIX: Bounds check on block indices
  if (block_idx < 0 || block_idx >= num_blocks) {
    return;  // Out of bounds
  }

  // Get source and destination block IDs
  int src_block_id = src_block_ids[block_idx];
  int dst_block_id = dst_block_ids[block_idx];

  // Calculate position indices
  int pos = element_offset / (heads_per_rank * head_size);
  int remaining = element_offset % (heads_per_rank * head_size);
  int head = remaining / head_size;
  int head_offset = remaining % head_size;

  // CRITICAL FIX: Ensure bounds checks on all indices
  if (pos >= block_size || head >= heads_per_rank || head_offset >= head_size) {
    return;  // Out of bounds
  }

  // Calculate source offset using provided strides
  size_t src_offset = rank * src_tp_stride + src_block_id * src_block_stride + pos * src_pos_stride +
                      head * src_head_stride + head_offset;

  // Calculate destination offset
  size_t dst_offset = rank * dst_tp_stride + dst_block_id * dst_block_stride + pos * dst_pos_stride +
                      head * dst_head_stride + head_offset;

  // Convert to byte pointers
  const char* src_bytes = (const char*)src_data + src_offset * elem_size;
  char* dst_bytes = (char*)dst_data + dst_offset * elem_size;

  // Copy element using proper size
  // FIX: Use memcpy for safety
  for (int i = 0; i < elem_size; i++) {
    dst_bytes[i] = src_bytes[i];
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
  // CRITICAL FIX: Validate inputs
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

  CUDA_CHECK(cudaMalloc(&d_src_blocks, num_src_blocks * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_dst_blocks, num_dst_blocks * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_src_blocks, src_block_ids, num_src_blocks * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dst_blocks, dst_block_ids, num_dst_blocks * sizeof(int), cudaMemcpyHostToDevice));

  // Calculate grid dimensions
  int elements_per_block = block_size * heads_per_rank * head_size;
  int total_elements = elements_per_block * num_src_blocks * kv_size;

  int cuda_block_size = 256;
  int grid_size = (total_elements + cuda_block_size - 1) / cuda_block_size;

  // CRITICAL FIX: Validate grid size
  if (grid_size <= 0) {
    fprintf(stderr, "Invalid grid size: %d\n", grid_size);
    cudaFree(d_src_blocks);
    cudaFree(d_dst_blocks);
    return cudaErrorInvalidValue;
  }

  // CRITICAL FIX: Print debug information for troubleshooting
  printf("Starting kernel: blocks=%d, threads=%d, total=%d\n", grid_size, cuda_block_size, total_elements);
  printf(
      "Dimensions: tp=%d, blocks=%d, size=%d, heads=%d, headsize=%d, elemsize=%d\n", kv_size, num_src_blocks,
      block_size, heads_per_rank, head_size, elem_size);

  // Launch kernel
  copy_blocks_kernel<<<grid_size, cuda_block_size>>>(
      src_data, dst_data, d_src_blocks, d_dst_blocks, num_src_blocks, kv_size, block_size, heads_per_rank, head_size,
      elem_size, src_tp_stride, src_block_stride, src_pos_stride, src_head_stride, dst_tp_stride, dst_block_stride,
      dst_pos_stride, dst_head_stride);

  // CRITICAL FIX: Check for kernel errors immediately
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
