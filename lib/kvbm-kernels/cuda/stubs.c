// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Stub implementations for CUDA kernel functions.
// These are used when nvcc is not available, allowing the library to be built
// without CUDA. The stubs abort() when called, but the binary can be moved to
// an environment with the real .so and work correctly via LD_LIBRARY_PATH.

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// cudaError_t equivalent - cudaSuccess = 0
typedef int cudaError_t;

// cudaStream_t is an opaque pointer
typedef void* cudaStream_t;

#define STUB_ABORT(name)                                                 \
  do {                                                                   \
    fprintf(                                                             \
        stderr,                                                          \
        "FATAL: %s called but CUDA kernels not available.\n"             \
        "This binary was built with stub kernels. To use CUDA:\n"        \
        "  1. Build with nvcc available, or\n"                           \
        "  2. Set LD_LIBRARY_PATH to include real libkvbm_kernels.so\n", \
        name);                                                           \
    abort();                                                             \
  } while (0)

cudaError_t
kvbm_kernels_launch_universal_from_block(
    void* const* universal_ptrs, const void* const* block_ptrs, size_t num_blocks, size_t nh, size_t nl, size_t no,
    size_t nt, size_t hd, int dtype_value, int layout_value, cudaStream_t stream)
{
  (void)universal_ptrs;
  (void)block_ptrs;
  (void)num_blocks;
  (void)nh;
  (void)nl;
  (void)no;
  (void)nt;
  (void)hd;
  (void)dtype_value;
  (void)layout_value;
  (void)stream;
  STUB_ABORT("kvbm_kernels_launch_universal_from_block");
  return 1;  // Unreachable, but silences compiler warning
}

cudaError_t
kvbm_kernels_launch_block_from_universal(
    const void* const* universal_ptrs, void* const* block_ptrs, size_t num_blocks, size_t nh, size_t nl, size_t no,
    size_t nt, size_t hd, int dtype_value, int layout_value, cudaStream_t stream)
{
  (void)universal_ptrs;
  (void)block_ptrs;
  (void)num_blocks;
  (void)nh;
  (void)nl;
  (void)no;
  (void)nt;
  (void)hd;
  (void)dtype_value;
  (void)layout_value;
  (void)stream;
  STUB_ABORT("kvbm_kernels_launch_block_from_universal");
  return 1;  // Unreachable
}

cudaError_t
kvbm_kernels_launch_vectorized_copy(
    void** src_ptrs, void** dst_ptrs, size_t copy_size_bytes, int num_pairs, cudaStream_t stream)
{
  (void)src_ptrs;
  (void)dst_ptrs;
  (void)copy_size_bytes;
  (void)num_pairs;
  (void)stream;
  STUB_ABORT("kvbm_kernels_launch_vectorized_copy");
  return 1;  // Unreachable
}

// This function is safe to call even with stubs - it just returns false
// indicating that batch async is not available.
bool
kvbm_kernels_has_memcpy_batch_async(void)
{
  return false;
}

// Stub for memcpy_batch - returns not supported since we can't do CUDA ops
cudaError_t
kvbm_kernels_memcpy_batch(
    const void* const* src_ptrs, void* const* dst_ptrs, size_t size_per_copy, size_t num_copies, int mode,
    cudaStream_t stream)
{
  (void)src_ptrs;
  (void)dst_ptrs;
  (void)size_per_copy;
  (void)num_copies;
  (void)mode;
  (void)stream;
  STUB_ABORT("kvbm_kernels_memcpy_batch");
  return 1;  // Unreachable
}

// Returns true if this is the stub library (no real CUDA kernels).
// Downstream crates can use this to skip CUDA tests at runtime.
bool
kvbm_kernels_is_stub_build(void)
{
  return true;
}
