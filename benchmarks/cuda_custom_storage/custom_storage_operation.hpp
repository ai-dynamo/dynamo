/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>
#include <sys/types.h>

#if !defined(CUDA_VERSION) || CUDA_VERSION < 13040
#error "CUDA CustomStorage benchmark requires CUDA 13.4 or newer headers"
#endif

namespace cuda_custom_storage_benchmark {

// Owns one benchmark checkpoint or restore operation. Storage transfer and
// persistence remain the caller's responsibility.
class CustomStorageOperation {
 public:
  CustomStorageOperation() = default;
  CustomStorageOperation(const CustomStorageOperation&) = delete;
  CustomStorageOperation& operator=(const CustomStorageOperation&) = delete;

  [[nodiscard]] CUresult BeginCheckpoint(pid_t pid);
  [[nodiscard]] CUresult BeginRestore(
      pid_t pid, CUcheckpointGpuPair* gpu_pairs = nullptr, unsigned int gpu_pair_count = 0);
  [[nodiscard]] CUresult Complete();

  [[nodiscard]] const CUcheckpointCustomStorageInfo* storage_info() const { return storage_info_; }
  [[nodiscard]] bool has_started() const { return has_started_; }

 private:
  CUresult Adopt(CUresult status, CUcheckpointCustomStorageInfo* storage_info);

  CUcheckpointCustomStorageInfo* storage_info_ = nullptr;
  bool has_started_ = false;
  bool has_valid_storage_shape_ = false;
  bool has_attempted_completion_ = false;
};

}  // namespace cuda_custom_storage_benchmark
