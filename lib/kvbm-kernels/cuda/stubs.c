// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define STUB_ABORT(name)                                                   \
  do {                                                                     \
    fprintf(                                                               \
        stderr,                                                            \
        "FATAL: %s called but CUDA kernels not available.\n"               \
        "This binary was built with stub kernels. To use CUDA:\n"          \
        "  1. Build with nvcc available, or\n"                             \
        "  2. Set LD_LIBRARY_PATH to include real libtensor_kernels.so\n", \
        name);                                                             \
    abort();                                                               \
  } while (0)

cudaError_t
launch_universal_from_block(
    void* const* universal_ptrs_device, const void* const* block_ptrs_device, size_t num_blocks, size_t nh, size_t nl,
    size_t no, size_t nt, size_t hd, int dtype_value, int layout_value, cudaStream_t stream)
{
  (void)universal_ptrs_device;
  (void)block_ptrs_device;
  (void)num_blocks;
  (void)nh;
  (void)nl;
  (void)no;
  (void)nt;
  (void)hd;
  (void)dtype_value;
  (void)layout_value;
  (void)stream;
  STUB_ABORT("launch_universal_from_block");
  return 1;  // Unreachable, but silences compiler warning
}

cudaError_t
launch_block_from_universal(
    const void* const* universal_ptrs_device, void* const* block_ptrs_device, size_t num_blocks, size_t nh, size_t nl,
    size_t no, size_t nt, size_t hd, int dtype_value, int layout_value, cudaStream_t stream)
{
  (void)universal_ptrs_device;
  (void)block_ptrs_device;
  (void)num_blocks;
  (void)nh;
  (void)nl;
  (void)no;
  (void)nt;
  (void)hd;
  (void)dtype_value;
  (void)layout_value;
  (void)stream;
  STUB_ABORT("launch_block_from_universal");
  return 1;  // Unreachable
}

cudaError_t
launch_operational_copy(
    const void* const* block_ptrs_host, const void* const* block_ptrs_device, void* const* operational_ptrs_host,
    void* const* operational_ptrs_device, size_t num_blocks, size_t nl, size_t no, size_t inner, size_t elem_size,
    int dtype_value, int direction_value, int backend_value, cudaStream_t stream)
{
  (void)block_ptrs_host;
  (void)block_ptrs_device;
  (void)operational_ptrs_host;
  (void)operational_ptrs_device;
  (void)num_blocks;
  (void)nl;
  (void)no;
  (void)inner;
  (void)elem_size;
  (void)dtype_value;
  (void)direction_value;
  (void)backend_value;
  (void)stream;
  STUB_ABORT("launch_operational_copy");
  return 1;  // Unreachable
}

// This function is safe to call even with stubs - it just returns false
// indicating that batch async is not available.
bool
has_memcpy_batch_async(void)
{
  return false;
}

// Returns true if this is the stub library (no real CUDA kernels).
// Downstream crates can use this to skip CUDA tests at runtime.
bool
is_stub_build(void)
{
  return true;
}
