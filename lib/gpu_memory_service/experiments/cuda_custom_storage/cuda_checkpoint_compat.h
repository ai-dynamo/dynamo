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

// Temporary CUDA 13.4 ABI declarations for the forward-compatible nscale
// environment. Remove these declarations after the build image carries the
// released CUDA 13.4 headers.
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

inline OperationCompleteFn
ResolveOperationComplete()
{
  void* symbol = nullptr;
  CUdriverProcAddressQueryResult query_status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  const CUresult status =
      cuGetProcAddress("cuCheckpointOperationComplete", &symbol, 13040, CU_GET_PROC_ADDRESS_DEFAULT, &query_status);
  if (status != CUDA_SUCCESS || symbol == nullptr || query_status != CU_GET_PROC_ADDRESS_SUCCESS) {
    return nullptr;
  }
  return reinterpret_cast<OperationCompleteFn>(symbol);
}

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

static_assert(sizeof(void*) == 8, "CUDA CustomStorage requires a 64-bit ABI");
static_assert(std::is_standard_layout_v<PerDeviceData>);
static_assert(sizeof(PerDeviceData) == 24);
static_assert(offsetof(PerDeviceData, devPtr) == 0);
static_assert(offsetof(PerDeviceData, size) == 8);
static_assert(offsetof(PerDeviceData, stream) == 16);
static_assert(std::is_standard_layout_v<StorageInfo>);
static_assert(sizeof(StorageInfo) == 24);
static_assert(offsetof(StorageInfo, handle) == 0);
static_assert(offsetof(StorageInfo, perDeviceData) == 8);
static_assert(offsetof(StorageInfo, deviceCount) == 16);
static_assert(sizeof(CUcheckpointCheckpointArgs) == 64);
static_assert(sizeof(CheckpointArgs) == sizeof(CUcheckpointCheckpointArgs));
static_assert(offsetof(CheckpointArgs, customStorageInfo_out) == 0);
static_assert(sizeof(CUcheckpointRestoreArgs) == 64);
static_assert(sizeof(RestoreArgs) == sizeof(CUcheckpointRestoreArgs));
static_assert(offsetof(RestoreArgs, customStorageInfo_out) == 16);

}  // namespace cuda_checkpoint_compat
