/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>

#include <cstddef>
#include <type_traits>

namespace cuda_checkpoint_compat {

#if defined(CUDA_VERSION) && CUDA_VERSION >= 13040

using OperationHandle = CUcheckpointOperationHandle;
using PerDeviceData = CUcheckpointCustomStoragePerDeviceData;
using StorageInfo = CUcheckpointCustomStorageInfo;
using CheckpointArgs = CUcheckpointCheckpointArgs;
using RestoreArgs = CUcheckpointRestoreArgs;
using OperationCompleteFn = decltype(&cuCheckpointOperationComplete);

#else

// Public CUDA 13.4 (13040) custom-storage ABI used while the image builds
// against CUDA 13.0 headers. Keep these declarations local to this helper.
struct Operation;
using OperationHandle = Operation*;

struct PerDeviceData {
  CUdeviceptr devPtr;
  size_t size;
  CUstream stream;
};

struct StorageInfo {
  OperationHandle handle;
  PerDeviceData* perDeviceData;
  unsigned int deviceCount;
};

struct CheckpointArgs {
  StorageInfo** customStorageInfo_out;
  char reserved[64 - sizeof(StorageInfo*)];
};

struct RestoreArgs {
  CUcheckpointGpuPair* gpuPairs;
  unsigned int gpuPairsCount;
  unsigned int padding0;
  StorageInfo** customStorageInfo_out;
  char reserved[64 - sizeof(CUcheckpointGpuPair*) - 2 * sizeof(unsigned int) - sizeof(StorageInfo**)];
};

using OperationCompleteFn = CUresult(CUDAAPI*)(OperationHandle);

#endif

inline CUcheckpointCheckpointArgs*
NativeArgs(CheckpointArgs* args)
{
  return reinterpret_cast<CUcheckpointCheckpointArgs*>(args);
}

inline CUcheckpointRestoreArgs*
NativeArgs(RestoreArgs* args)
{
  return reinterpret_cast<CUcheckpointRestoreArgs*>(args);
}

static_assert(sizeof(void*) == 8, "CUDA checkpoint custom storage requires a 64-bit ABI");
static_assert(std::is_standard_layout_v<PerDeviceData>);
static_assert(sizeof(PerDeviceData) == 24);
static_assert(alignof(PerDeviceData) == 8);
static_assert(offsetof(PerDeviceData, devPtr) == 0);
static_assert(offsetof(PerDeviceData, size) == 8);
static_assert(offsetof(PerDeviceData, stream) == 16);

static_assert(std::is_standard_layout_v<StorageInfo>);
static_assert(sizeof(StorageInfo) == 24);
static_assert(alignof(StorageInfo) == 8);
static_assert(offsetof(StorageInfo, handle) == 0);
static_assert(offsetof(StorageInfo, perDeviceData) == 8);
static_assert(offsetof(StorageInfo, deviceCount) == 16);

static_assert(sizeof(CUcheckpointCheckpointArgs) == 64);
static_assert(std::is_standard_layout_v<CheckpointArgs>);
static_assert(sizeof(CheckpointArgs) == sizeof(CUcheckpointCheckpointArgs));
static_assert(alignof(CheckpointArgs) == alignof(CUcheckpointCheckpointArgs));
static_assert(offsetof(CheckpointArgs, customStorageInfo_out) == 0);

static_assert(sizeof(CUcheckpointRestoreArgs) == 64);
static_assert(std::is_standard_layout_v<RestoreArgs>);
static_assert(sizeof(RestoreArgs) == sizeof(CUcheckpointRestoreArgs));
static_assert(alignof(RestoreArgs) == alignof(CUcheckpointRestoreArgs));
static_assert(offsetof(RestoreArgs, gpuPairs) == 0);
static_assert(offsetof(RestoreArgs, gpuPairsCount) == 8);
static_assert(offsetof(RestoreArgs, padding0) == 12);
static_assert(offsetof(RestoreArgs, customStorageInfo_out) == 16);

}  // namespace cuda_checkpoint_compat
