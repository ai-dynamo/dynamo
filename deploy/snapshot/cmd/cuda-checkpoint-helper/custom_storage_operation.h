/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>
#include <sys/types.h>

#include "cuda_checkpoint_compat.h"

namespace cuda_checkpoint_custom_storage {

// Owns only the CUDA operation state. Storage transfer, persistence, and policy
// remain the caller's responsibility. An unfinished operation is deliberately
// not completed on destruction: after a transfer failure, acknowledging partial
// storage would be unsafe, so a long-lived caller must observe fatal() and exit.
class Operation {
 public:
  Operation() = default;
  Operation(const Operation&) = delete;
  Operation& operator=(const Operation&) = delete;
  Operation(Operation&& other) noexcept;
  Operation& operator=(Operation&&) = delete;

  [[nodiscard]] CUresult BeginCheckpoint(pid_t pid);
  [[nodiscard]] CUresult BeginRestore(
      pid_t pid, CUcheckpointGpuPair* gpu_pairs = nullptr, unsigned int gpu_pair_count = 0);
  [[nodiscard]] CUresult Complete(cuda_checkpoint_compat::OperationCompleteFn complete);

  [[nodiscard]] const cuda_checkpoint_compat::StorageInfo* storage_info() const { return storage_info_; }
  [[nodiscard]] bool fatal() const { return driver_call_succeeded_ && !completion_succeeded_; }

 private:
  CUresult Adopt(CUresult status, cuda_checkpoint_compat::StorageInfo* storage_info);

  cuda_checkpoint_compat::StorageInfo* storage_info_ = nullptr;
  bool driver_call_succeeded_ = false;
  bool completion_attempted_ = false;
  bool completion_succeeded_ = false;
};

}  // namespace cuda_checkpoint_custom_storage
