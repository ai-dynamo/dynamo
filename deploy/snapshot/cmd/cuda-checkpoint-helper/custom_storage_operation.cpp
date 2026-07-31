/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "custom_storage_operation.h"

#include <utility>

namespace cuda_checkpoint_custom_storage {
namespace {

bool
HasValidStorageShape(const cuda_checkpoint_compat::StorageInfo* info)
{
  return info != nullptr && info->handle != nullptr && (info->deviceCount == 0 || info->perDeviceData != nullptr);
}

}  // namespace

Operation::Operation(Operation&& other) noexcept
    : storage_info_(std::exchange(other.storage_info_, nullptr)),
      driver_call_succeeded_(std::exchange(other.driver_call_succeeded_, false)),
      completion_attempted_(std::exchange(other.completion_attempted_, false)),
      completion_succeeded_(std::exchange(other.completion_succeeded_, false))
{
}

CUresult
Operation::Adopt(CUresult status, cuda_checkpoint_compat::StorageInfo* storage_info)
{
  if (status != CUDA_SUCCESS) {
    return status;
  }
  driver_call_succeeded_ = true;
  storage_info_ = storage_info;
  return HasValidStorageShape(storage_info) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult
Operation::BeginCheckpoint(pid_t pid)
{
  if (pid <= 0 || driver_call_succeeded_ || completion_attempted_) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  cuda_checkpoint_compat::StorageInfo* storage_info = nullptr;
  cuda_checkpoint_compat::CheckpointArgs args{};
  args.customStorageInfo_out = &storage_info;
  const CUresult status = cuCheckpointProcessCheckpoint(pid, cuda_checkpoint_compat::NativeArgs(&args));
  return Adopt(status, storage_info);
}

CUresult
Operation::BeginRestore(pid_t pid, CUcheckpointGpuPair* gpu_pairs, unsigned int gpu_pair_count)
{
  if (pid <= 0 || driver_call_succeeded_ || completion_attempted_ || (gpu_pair_count != 0 && gpu_pairs == nullptr)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  cuda_checkpoint_compat::StorageInfo* storage_info = nullptr;
  cuda_checkpoint_compat::RestoreArgs args{};
  args.gpuPairs = gpu_pairs;
  args.gpuPairsCount = gpu_pair_count;
  args.customStorageInfo_out = &storage_info;
  const CUresult status = cuCheckpointProcessRestore(pid, cuda_checkpoint_compat::NativeArgs(&args));
  return Adopt(status, storage_info);
}

CUresult
Operation::Complete(cuda_checkpoint_compat::OperationCompleteFn complete)
{
  if (!driver_call_succeeded_ || storage_info_ == nullptr || storage_info_->handle == nullptr ||
      completion_attempted_ || complete == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  completion_attempted_ = true;
  const auto handle = storage_info_->handle;
  storage_info_ = nullptr;
  const CUresult status = complete(handle);
  completion_succeeded_ = status == CUDA_SUCCESS;
  return status;
}

}  // namespace cuda_checkpoint_custom_storage
