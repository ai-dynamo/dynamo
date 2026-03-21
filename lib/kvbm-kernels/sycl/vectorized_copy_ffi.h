// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// C FFI interface for the SYCL vectorized_copy kernel.
//
// This header defines the C-linkage API that Rust calls via FFI.
// The implementation creates a sycl::queue sharing the caller's
// Level Zero context (via SYCL L0 interop) and dispatches the
// VectorizedCopyKernel functor through the SYCL runtime.
//
// Build:
//   icpx -fsycl -shared -fPIC -o libvectorized_copy_sycl.so \
//        vectorized_copy_ffi.cpp

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to the SYCL state (sycl::queue + cached resources).
typedef struct SyclVcState SyclVcState;

// Create a SYCL queue sharing the given Level Zero context and device.
//
// `ze_context` and `ze_device` are raw Level Zero handles (ze_context_handle_t,
// ze_device_handle_t) cast to void*.
//
// Returns an opaque handle, or NULL on failure.
SyclVcState* sycl_vc_init(void* ze_context, void* ze_device);

// Submit the vectorized_copy kernel and block until completion.
//
// `src_addrs_dev` / `dst_addrs_dev` are DEVICE pointers to arrays of
// uint64_t addresses, already uploaded by the caller.
// `copy_size` is bytes to copy per pair.
// `num_pairs` is the number of (src, dst) pairs.
//
// Returns 0 on success, non-zero on failure.
int sycl_vc_run(SyclVcState* state,
                uint64_t src_addrs_dev,
                uint64_t dst_addrs_dev,
                uint64_t copy_size,
                int      num_pairs);

// Destroy the SYCL state and release resources.
void sycl_vc_destroy(SyclVcState* state);

#ifdef __cplusplus
}
#endif
