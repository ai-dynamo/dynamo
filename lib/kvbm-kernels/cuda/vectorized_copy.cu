#include <cuda_runtime.h>

#include <cstdint>

extern "C" __global__ void
vectorised_copy(void** src_ptrs, void** dst_ptrs, size_t copy_size_in_bytes, int num_pairs)
{
  int pair_id = blockIdx.x;
  int block_stride = gridDim.x;

  // Determine the alignment case for the entire block
  // To avoid thread divergence, checking alignment for the entire block
  bool vector_copy_16 =
      (uintptr_t(src_ptrs[pair_id]) % 16 == 0 && uintptr_t(dst_ptrs[pair_id]) % 16 == 0 && copy_size_in_bytes >= 16);
  bool vector_copy_8 = !vector_copy_16 && (uintptr_t(src_ptrs[pair_id]) % 8 == 0 &&
                                           uintptr_t(dst_ptrs[pair_id]) % 8 == 0 && copy_size_in_bytes >= 8);
  bool vector_copy_4 =
      !vector_copy_8 && !vector_copy_16 &&
      (uintptr_t(src_ptrs[pair_id]) % 4 == 0 && uintptr_t(dst_ptrs[pair_id]) % 4 == 0 && copy_size_in_bytes >= 4);

  for (; pair_id < num_pairs; pair_id += block_stride) {
    char* src = static_cast<char*>(src_ptrs[pair_id]);
    char* dst = static_cast<char*>(dst_ptrs[pair_id]);
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (vector_copy_16) {
      // Vectorized copy: copy as int4 (16-byte units)
      for (int i = tid; i < copy_size_in_bytes / 16; i += block_size) {
        reinterpret_cast<int4*>(dst)[i] = reinterpret_cast<const int4*>(src)[i];
      }
    } else if (vector_copy_8) {
      // Vectorized copy: copy as int2 (8-byte units)
      for (size_t i = tid; i < copy_size_in_bytes / 8; i += block_size) {
        reinterpret_cast<int2*>(dst)[i] = reinterpret_cast<int2*>(src)[i];
      }
    } else if (vector_copy_4) {
      // Vectorized copy: copy as uint32_t (4-byte units)
      for (size_t i = tid; i < copy_size_in_bytes / 4; i += block_size) {
        reinterpret_cast<uint32_t*>(dst)[i] = reinterpret_cast<uint32_t*>(src)[i];
      }
    }

    // Unified handling of remaining bytes (leftover from vectorized + scalar fallback)
    size_t vectorized_bytes = 0;
    if (vector_copy_16) {
      vectorized_bytes = (copy_size_in_bytes / 16) * 16;
    } else if (vector_copy_8) {
      vectorized_bytes = (copy_size_in_bytes / 8) * 8;
    } else if (vector_copy_4) {
      vectorized_bytes = (copy_size_in_bytes / 4) * 4;
    }

    // Copy remaining bytes (either leftover from vectorized or entire data if scalar)
    for (size_t i = tid; i < copy_size_in_bytes - vectorized_bytes; i += block_size) {
      dst[vectorized_bytes + i] = src[vectorized_bytes + i];
    }
  }
}
