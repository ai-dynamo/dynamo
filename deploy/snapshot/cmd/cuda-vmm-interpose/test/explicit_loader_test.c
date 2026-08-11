/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE

#include <cuda.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#undef cuMulticastBindMem

typedef CUresult(CUDAAPI* resolver_v2_fn)(
    const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
typedef CUresult(CUDAAPI* create_fn)(
    CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*,
    unsigned long long);
typedef CUresult(CUDAAPI* map_fn)(
    CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle,
    unsigned long long);
typedef CUresult(CUDAAPI* multicast_bind_fn)(
    CUmemGenericAllocationHandle, size_t, CUmemGenericAllocationHandle, size_t,
    size_t, unsigned long long);
#if CUDA_VERSION >= 13010
typedef CUresult(CUDAAPI* multicast_bind_v2_fn)(
    CUmemGenericAllocationHandle, CUdevice, size_t,
    CUmemGenericAllocationHandle, size_t, size_t, unsigned long long);
#endif

static void
require(int condition, const char* message)
{
  if (!condition) {
    fprintf(stderr, "%s\n", message);
    exit(1);
  }
}

int
main(void)
{
  void* cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  resolver_v2_fn resolve;
  void* real_create;
  void* entry;
  void* unknown;
  void* (*unknown_address)(void);
  unsigned int (*logical_forward_count)(void);
  unsigned int (*multicast_bind_count)(void);
  CUdriverProcAddressQueryResult status;
  CUmemAllocationProp properties;
  CUmemGenericAllocationHandle logical;

  require(cuda != NULL, dlerror());
  real_create = dlsym(cuda, "cuMemCreate");
  unknown_address = (void* (*)(void))dlsym(cuda, "fake_cuda_unknown_entry");
  logical_forward_count =
      (unsigned int (*)(void))dlsym(cuda, "fake_cuda_logical_forward_count");
  multicast_bind_count =
      (unsigned int (*)(void))dlsym(cuda, "fake_cuda_multicast_bind_count");
  resolve = (resolver_v2_fn)dlsym(cuda, "cuGetProcAddress_v2");
  require(
      real_create != NULL && unknown_address != NULL &&
          logical_forward_count != NULL && multicast_bind_count != NULL &&
          resolve != NULL,
      "explicit libcuda lookups failed");

  unknown = NULL;
  require(
      resolve("cuUnknown", &unknown, CUDA_VERSION, 0, &status) ==
              CUDA_SUCCESS &&
          status == CU_GET_PROC_ADDRESS_SUCCESS &&
          unknown == unknown_address(),
      "explicit resolver did not preserve an unrelated real PFN");

  entry = NULL;
  require(
      resolve("cuMemCreate", &entry, CUDA_VERSION, 0, &status) ==
              CUDA_SUCCESS &&
          status == CU_GET_PROC_ADDRESS_SUCCESS && entry != NULL &&
          entry != real_create,
      "cuda-python-style cached resolver did not return the managed create wrapper");
  memset(&properties, 0, sizeof(properties));
  properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  properties.location.id = 0;
  properties.requestedHandleTypes =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  require(
      ((create_fn)entry)(&logical, 4096, &properties, 0) == CUDA_SUCCESS &&
          ((uint64_t)logical & UINT64_C(0xffff000000000000)) ==
              UINT64_C(0xd94d000000000000),
      "cached managed create PFN did not return a logical handle");

  require(
      resolve("cuMemMap", &entry, CUDA_VERSION, 0, &status) == CUDA_SUCCESS &&
          ((map_fn)entry)(0x1000, 4096, 0, logical, 0) == CUDA_SUCCESS &&
          logical_forward_count() == 0,
      "cached managed map PFN bypassed logical-handle translation");

  require(
      resolve("cuMulticastBindMem", &entry, 12010, 0, &status) ==
              CUDA_SUCCESS &&
          ((multicast_bind_fn)entry)(0x8000, 0, logical, 0, 4096, 0) ==
              CUDA_ERROR_NOT_SUPPORTED &&
          multicast_bind_count() == 0 && logical_forward_count() == 0,
      "managed multicast bind reached the fake UMD");
  require(
      ((multicast_bind_fn)entry)(0x8000, 0, 0x9000, 0, 4096, 0) ==
              CUDA_SUCCESS &&
          multicast_bind_count() == 1,
      "unmanaged multicast bind was not forwarded unchanged");

#if CUDA_VERSION >= 13010
  require(
      resolve("cuMulticastBindMem", &entry, 13010, 0, &status) ==
              CUDA_SUCCESS &&
          ((multicast_bind_v2_fn)entry)(
              0x8000, 0, 0, logical, 0, 4096, 0) ==
              CUDA_ERROR_NOT_SUPPORTED &&
          multicast_bind_count() == 1 && logical_forward_count() == 0,
      "cuda-python-style managed multicast bind v2 reached the fake UMD");
#endif
  require(dlclose(cuda) == 0, "cannot close explicit libcuda handle");
  return 0;
}
