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
#include <unistd.h>

#undef cuMulticastAddDevice
#undef cuMulticastBindAddr
#undef cuMulticastBindMem
#undef cuMulticastCreate
#undef cuMulticastUnbind

typedef CUresult(CUDAAPI* resolver_v2_fn)(
    const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
typedef CUresult(CUDAAPI* create_fn)(
    CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*,
    unsigned long long);
typedef CUresult(CUDAAPI* map_fn)(
    CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle,
    unsigned long long);
typedef CUresult(CUDAAPI* export_fn)(
    void*, CUmemGenericAllocationHandle, CUmemAllocationHandleType,
    unsigned long long);
typedef CUresult(CUDAAPI* multicast_create_fn)(
    CUmemGenericAllocationHandle*, const CUmulticastObjectProp*);
typedef CUresult(CUDAAPI* multicast_add_fn)(
    CUmemGenericAllocationHandle, CUdevice);
typedef CUresult(CUDAAPI* multicast_bind_fn)(
    CUmemGenericAllocationHandle, size_t, CUmemGenericAllocationHandle, size_t,
    size_t, unsigned long long);
typedef CUresult(CUDAAPI* multicast_bind_addr_fn)(
    CUmemGenericAllocationHandle, size_t, CUdeviceptr, size_t,
    unsigned long long);
#if CUDA_VERSION >= 13010
typedef CUresult(CUDAAPI* multicast_bind_addr_v2_fn)(
    CUmemGenericAllocationHandle, CUdevice, size_t, CUdeviceptr, size_t,
    unsigned long long);
#endif
typedef CUresult(CUDAAPI* multicast_unbind_fn)(
    CUmemGenericAllocationHandle, CUdevice, size_t, size_t);
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
  unsigned int (*multicast_bind_addr_count)(void);
  unsigned int (*multicast_create_count)(void);
  unsigned int (*multicast_add_count)(void);
  unsigned int (*multicast_unbind_count)(void);
  CUdevice (*multicast_device)(unsigned int);
  CUdriverProcAddressQueryResult status;
  CUmemAllocationProp properties;
  CUmulticastObjectProp multicast_properties;
  CUmemGenericAllocationHandle logical;
  CUmemGenericAllocationHandle multicast;
  int capability_fd;

  require(cuda != NULL, dlerror());
  real_create = dlsym(cuda, "cuMemCreate");
  unknown_address = (void* (*)(void))dlsym(cuda, "fake_cuda_unknown_entry");
  logical_forward_count =
      (unsigned int (*)(void))dlsym(cuda, "fake_cuda_logical_forward_count");
  multicast_bind_count =
      (unsigned int (*)(void))dlsym(cuda, "fake_cuda_multicast_bind_count");
  multicast_bind_addr_count = (unsigned int (*)(void))dlsym(
      cuda, "fake_cuda_multicast_bind_addr_count");
  multicast_create_count = (unsigned int (*)(void))dlsym(
      cuda, "fake_cuda_multicast_create_count");
  multicast_add_count = (unsigned int (*)(void))dlsym(
      cuda, "fake_cuda_multicast_add_count");
  multicast_unbind_count = (unsigned int (*)(void))dlsym(
      cuda, "fake_cuda_multicast_unbind_count");
  multicast_device =
      (CUdevice(*)(unsigned int))dlsym(cuda, "fake_cuda_multicast_device");
  resolve = (resolver_v2_fn)dlsym(cuda, "cuGetProcAddress_v2");
  require(
      real_create != NULL && unknown_address != NULL &&
          logical_forward_count != NULL && multicast_bind_count != NULL &&
          multicast_bind_addr_count != NULL &&
          multicast_create_count != NULL && multicast_add_count != NULL &&
          multicast_unbind_count != NULL && multicast_device != NULL &&
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
      resolve(
          "cuMemExportToShareableHandle", &entry, CUDA_VERSION, 0,
          &status) == CUDA_SUCCESS &&
          ((export_fn)entry)(
              &capability_fd, logical,
              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) == CUDA_SUCCESS,
      "cached managed export PFN did not export a capability");
  close(capability_fd);
  memset(&multicast_properties, 0, sizeof(multicast_properties));
  multicast_properties.handleTypes =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  multicast_properties.size = 4096;
  multicast_properties.numDevices = 1;
  require(
      resolve("cuMulticastCreate", &entry, CUDA_VERSION, 0, &status) ==
              CUDA_SUCCESS &&
          ((multicast_create_fn)entry)(
              &multicast, &multicast_properties) == CUDA_SUCCESS &&
          multicast != 0x1235 && multicast_create_count() == 1,
      "cached multicast create PFN did not return a logical handle");
  require(
      resolve("cuMulticastAddDevice", &entry, CUDA_VERSION, 0, &status) ==
              CUDA_SUCCESS &&
          ((multicast_add_fn)entry)(multicast, 0) == CUDA_SUCCESS &&
          multicast_add_count() == 1 && multicast_device(0) == 0,
      "cached multicast add PFN did not translate the logical handle");
  require(
      resolve("cuMulticastBindMem", &entry, 12010, 0, &status) ==
              CUDA_SUCCESS &&
          ((multicast_bind_fn)entry)(
              multicast, 0, logical, 0, 4096, 0) == CUDA_SUCCESS &&
          multicast_bind_count() == 1 && logical_forward_count() == 0,
      "cached multicast bind PFN did not translate logical handles");
  require(
      resolve("cuMulticastUnbind", &entry, CUDA_VERSION, 0, &status) ==
              CUDA_SUCCESS &&
          ((multicast_unbind_fn)entry)(multicast, 0, 0, 4096) ==
              CUDA_SUCCESS &&
          multicast_unbind_count() == 1 && logical_forward_count() == 0,
      "cached multicast unbind PFN did not translate the logical handle");
  require(
      resolve("cuMulticastBindAddr", &entry, 12010, 0, &status) ==
              CUDA_SUCCESS &&
          ((multicast_bind_addr_fn)entry)(
              UINT64_C(0x8000), 0, 0x2000, 4096, 0) == CUDA_SUCCESS &&
          multicast_bind_addr_count() == 1,
      "unmanaged multicast address bind was not forwarded unchanged");
  require(
      ((multicast_bind_addr_fn)entry)(
          multicast, 0, 0x2000, 4096, 0) == CUDA_ERROR_NOT_SUPPORTED &&
          multicast_bind_addr_count() == 1 &&
          logical_forward_count() == 0,
      "managed multicast address bind reached the fake UMD");

#if CUDA_VERSION >= 13010
  require(
      resolve("cuMulticastBindAddr", &entry, 13010, 0, &status) ==
              CUDA_SUCCESS &&
          ((multicast_bind_addr_v2_fn)entry)(
              UINT64_C(0x8000), 0, 0, 0x2000, 4096, 0) == CUDA_SUCCESS &&
          multicast_bind_addr_count() == 2,
      "unmanaged multicast address bind v2 was not forwarded unchanged");
  require(
      ((multicast_bind_addr_v2_fn)entry)(
          multicast, 0, 0, 0x2000, 4096, 0) ==
              CUDA_ERROR_NOT_SUPPORTED &&
          multicast_bind_addr_count() == 2 &&
          logical_forward_count() == 0,
      "managed multicast address bind v2 reached the fake UMD");
#endif

  require(
      resolve("cuMulticastBindMem", &entry, 12010, 0, &status) ==
              CUDA_SUCCESS &&
          ((multicast_bind_fn)entry)(0x8000, 0, logical, 0, 4096, 0) ==
              CUDA_ERROR_INVALID_HANDLE &&
          multicast_bind_count() == 1 && logical_forward_count() == 0,
      "managed multicast bind reached the fake UMD");
  require(
      ((multicast_bind_fn)entry)(0x8000, 0, 0x9000, 0, 4096, 0) ==
              CUDA_SUCCESS &&
          multicast_bind_count() == 2,
      "unmanaged multicast bind was not forwarded unchanged");

#if CUDA_VERSION >= 13010
  require(
      resolve("cuMulticastBindMem", &entry, 13010, 0, &status) ==
              CUDA_SUCCESS &&
          ((multicast_bind_v2_fn)entry)(
              0x8000, 0, 0, logical, 0, 4096, 0) ==
              CUDA_ERROR_INVALID_HANDLE &&
          multicast_bind_count() == 2 && logical_forward_count() == 0,
      "cuda-python-style managed multicast bind v2 reached the fake UMD");
#endif
  require(dlclose(cuda) == 0, "cannot close explicit libcuda handle");
  return 0;
}
