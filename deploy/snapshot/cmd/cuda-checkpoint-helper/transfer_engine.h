/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>

#include <atomic>
#include <cstddef>
#include <string>
#include <vector>

#include "transfer_config.h"

namespace cuda_checkpoint_transfer {

enum class TransferOperation {
  kCheckpoint,
  kRestore,
};

struct StorageFileMetrics {
  size_t bytes = 0;
  double storage_seconds = 0.0;
  double fsync_seconds = 0.0;
};

struct TransferMetrics {
  size_t bytes = 0;
  double setup_seconds = 0.0;
  double pipeline_seconds = 0.0;
  double storage_seconds = 0.0;
  double cuda_wait_seconds = 0.0;
  double fsync_seconds = 0.0;
  double cleanup_seconds = 0.0;
  double total_seconds = 0.0;
  std::vector<StorageFileMetrics> files;
};

class TransferCancellation {
 public:
  void Cancel() { cancelled_.store(true, std::memory_order_relaxed); }
  bool IsCancelled() const { return cancelled_.load(std::memory_order_relaxed); }

 private:
  std::atomic<bool> cancelled_{false};
};

bool TransferExtent(
    CUdeviceptr device_ptr, size_t extent_size, CUstream stream, CUcontext context, const StorageLayout& storage,
    TransferOperation operation, const TransferOptions& options, TransferCancellation* cancellation,
    TransferMetrics* metrics, std::string* error);

}  // namespace cuda_checkpoint_transfer
