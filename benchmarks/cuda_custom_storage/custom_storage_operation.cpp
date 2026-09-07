/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "custom_storage_operation.hpp"

namespace cuda_custom_storage_benchmark {
namespace {

bool
HasValidStorageShape(const CUcheckpointCustomStorageInfo* info)
{
  return info != nullptr && info->handle != nullptr && (info->deviceCount == 0 || info->perDeviceData != nullptr);
}

}  // namespace

CUresult
CustomStorageOperation::Adopt(CUresult status, CUcheckpointCustomStorageInfo* storage_info)
{
  if (status != CUDA_SUCCESS) {
    return status;
  }
  has_started_ = true;
  storage_info_ = storage_info;
  has_valid_storage_shape_ = HasValidStorageShape(storage_info);
  return has_valid_storage_shape_ ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult
CustomStorageOperation::BeginCheckpoint(pid_t pid)
{
  if (pid <= 0 || has_started_ || has_attempted_completion_) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUcheckpointCustomStorageInfo* storage_info = nullptr;
  CUcheckpointCheckpointArgs args{};
  args.customStorageInfo_out = &storage_info;
  const CUresult status = cuCheckpointProcessCheckpoint(pid, &args);
  return Adopt(status, storage_info);
}

CUresult
CustomStorageOperation::BeginRestore(pid_t pid, CUcheckpointGpuPair* gpu_pairs, unsigned int gpu_pair_count)
{
  if (pid <= 0 || has_started_ || has_attempted_completion_ || (gpu_pair_count != 0 && gpu_pairs == nullptr)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUcheckpointCustomStorageInfo* storage_info = nullptr;
  CUcheckpointRestoreArgs args{};
  args.gpuPairs = gpu_pairs;
  args.gpuPairsCount = gpu_pair_count;
  args.customStorageInfo_out = &storage_info;
  const CUresult status = cuCheckpointProcessRestore(pid, &args);
  return Adopt(status, storage_info);
}

CUresult
CustomStorageOperation::Complete()
{
  if (!has_started_ || !has_valid_storage_shape_ || has_attempted_completion_) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  has_attempted_completion_ = true;
  const auto handle = storage_info_->handle;
  storage_info_ = nullptr;
  return cuCheckpointOperationComplete(handle);
}

}  // namespace cuda_custom_storage_benchmark
