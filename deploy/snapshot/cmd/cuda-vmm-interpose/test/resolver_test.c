/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "../protocol.h"

#undef cuGetProcAddress
#undef cudaGetDriverEntryPoint
#undef cudaGetDriverEntryPointByVersion

CUresult CUDAAPI cuGetProcAddress(const char*, void**, int, cuuint64_t);
CUresult CUDAAPI cuGetProcAddress_v2(
    const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
CUresult CUDAAPI cuGetProcAddress_v2_ptsz(
    const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
CUresult CUDAAPI cuMemRetainAllocationHandle(
    CUmemGenericAllocationHandle*, void*);
CUresult CUDAAPI cuMemGetAllocationPropertiesFromHandle(
    CUmemAllocationProp*, CUmemGenericAllocationHandle);
cudaError_t CUDARTAPI cudaGetDriverEntryPoint(
    const char*, void**, unsigned long long,
    enum cudaDriverEntryPointQueryResult*);
cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion(
    const char*, void**, unsigned int, unsigned long long,
    enum cudaDriverEntryPointQueryResult*);
cudaError_t CUDARTAPI cudaGetDriverEntryPoint_ptsz(
    const char*, void**, unsigned long long,
    enum cudaDriverEntryPointQueryResult*);
cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion_ptsz(
    const char*, void**, unsigned int, unsigned long long,
    enum cudaDriverEntryPointQueryResult*);
void* fake_cuda_unknown_entry(void);
cuuint64_t fake_cuda_last_driver_flags(void);
unsigned long long fake_cuda_last_runtime_flags(void);
unsigned int fake_cuda_last_runtime_version(void);
unsigned int fake_cuda_create_count(void);
unsigned int fake_cuda_import_count(void);
unsigned int fake_cuda_release_count(void);
unsigned int fake_cuda_map_count(void);
unsigned int fake_cuda_export_count(void);
unsigned int fake_cuda_logical_forward_count(void);
CUmemGenericAllocationHandle fake_cuda_last_consumed_handle(void);
void* fake_cuda_ipc_entry(void);

static void
require(int condition, const char* message)
{
  if (!condition) {
    fprintf(stderr, "%s\n", message);
    exit(1);
  }
}

static void
test_resolver(void)
{
  void* entry = NULL;
  CUdriverProcAddressQueryResult driver_status;
  enum cudaDriverEntryPointQueryResult runtime_status;

  require(
      cuGetProcAddress("cuUnknown", &entry, CUDA_VERSION, 0) == CUDA_SUCCESS &&
          entry == fake_cuda_unknown_entry(),
      "unmanaged CUDA resolver result was not forwarded");
  require(
      cuGetProcAddress("cuMemCreate", &entry, CUDA_VERSION, 0) == CUDA_SUCCESS &&
          entry == (void*)&cuMemCreate,
      "managed CUDA resolver result was not interposed");
  require(
      cuGetProcAddress_v2(
          "cuMemRetainAllocationHandle", &entry, CUDA_VERSION, 0,
          &driver_status) == CUDA_SUCCESS &&
          entry == (void*)&cuMemRetainAllocationHandle,
      "retained-handle poison wrapper was not returned by the resolver");
  require(
      cudaGetDriverEntryPoint(
          "cuUnknown", &entry, 0, &runtime_status) == cudaSuccess &&
          runtime_status == cudaDriverEntryPointSuccess &&
          entry == fake_cuda_unknown_entry(),
      "unmanaged runtime resolver result was not forwarded");
  require(
      cudaGetDriverEntryPoint(
          "cuMemCreate", &entry, 0, &runtime_status) == cudaSuccess &&
          entry == (void*)&cuMemCreate,
      "managed runtime resolver result was not interposed");
  require(
      cudaGetDriverEntryPointByVersion(
          "cuMemCreate", &entry, 12030, 0, &runtime_status) == cudaSuccess &&
          entry == (void*)&cuMemCreate &&
          fake_cuda_last_runtime_version() == 12030,
      "versioned runtime resolver was not forwarded");
  require(
      cuGetProcAddress(
          "cuIpcGetMemHandle", &entry, CUDA_VERSION, 0) == CUDA_SUCCESS &&
          entry == fake_cuda_ipc_entry(),
      "legacy CUDA IPC resolver entry was substituted");
}

static void
test_ptsz_resolvers(void)
{
  void* entry = NULL;
  CUdriverProcAddressQueryResult driver_status;
  enum cudaDriverEntryPointQueryResult runtime_status;

  require(
      cuGetProcAddress_v2_ptsz(
          "cuMemCreate", &entry, CUDA_VERSION, 0, &driver_status) ==
          CUDA_SUCCESS &&
          entry == (void*)&cuMemCreate &&
          (fake_cuda_last_driver_flags() &
           CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) != 0,
      "driver _ptsz resolver did not use the public resolver with PTDS");
  require(
      cuGetProcAddress_v2_ptsz(
          "cuUnknown", &entry, CUDA_VERSION,
          CU_GET_PROC_ADDRESS_LEGACY_STREAM, &driver_status) == CUDA_SUCCESS &&
          fake_cuda_last_driver_flags() == CU_GET_PROC_ADDRESS_LEGACY_STREAM,
      "driver _ptsz resolver overwrote an explicit stream mode");
  require(
      cudaGetDriverEntryPoint_ptsz(
          "cuMemCreate", &entry, 0, &runtime_status) == cudaSuccess &&
          entry == (void*)&cuMemCreate &&
          (fake_cuda_last_runtime_flags() &
           cudaEnablePerThreadDefaultStream) != 0,
      "runtime _ptsz resolver did not use the public resolver with PTDS");
  require(
      cudaGetDriverEntryPointByVersion_ptsz(
          "cuMemCreate", &entry, 12040, cudaEnableLegacyStream,
          &runtime_status) == cudaSuccess &&
          entry == (void*)&cuMemCreate &&
          fake_cuda_last_runtime_version() == 12040 &&
          fake_cuda_last_runtime_flags() == cudaEnableLegacyStream,
      "versioned runtime _ptsz resolver did not preserve stream flags");
}

static void
test_unavailable_resolver_results(void)
{
  void* entry = (void*)&cuMemCreate;
  CUdriverProcAddressQueryResult driver_status;
  enum cudaDriverEntryPointQueryResult runtime_status;

  require(
      cuGetProcAddress_v2(
          "cuMemCreate", &entry, 222, 0, &driver_status) == CUDA_SUCCESS &&
          entry == NULL,
      "null driver PFN was replaced with a wrapper");
  require(
      cudaGetDriverEntryPointByVersion(
          "cuMemCreate", &entry, 222, 0, &runtime_status) == cudaSuccess &&
          entry == NULL,
      "null runtime PFN was replaced with a wrapper");
  require(
      cuGetProcAddress_v2(
          "cuMemCreate", &entry, 111, 0, &driver_status) == CUDA_SUCCESS &&
          driver_status == (CUdriverProcAddressQueryResult)17 &&
          entry == fake_cuda_unknown_entry(),
      "driver availability status or PFN was not preserved");
  require(
      cudaGetDriverEntryPointByVersion(
          "cuMemCreate", &entry, 111, 0, &runtime_status) == cudaSuccess &&
          runtime_status == (enum cudaDriverEntryPointQueryResult)17 &&
          entry == fake_cuda_unknown_entry(),
      "runtime availability status or PFN was not preserved");
}

static void
test_logical_handles(void)
{
  CUmemAllocationProp properties;
  CUmemAllocationProp queried;
  CUmemGenericAllocationHandle created;
  CUmemGenericAllocationHandle imported;
  CUmemGenericAllocationHandle unknown =
      (CUmemGenericAllocationHandle)UINT64_C(0xd94d00000000ffff);
  int fd;
  int export_fd;

  memset(&properties, 0, sizeof(properties));
  properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  properties.location.id = 0;
  properties.requestedHandleTypes =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  require(
      cuMemCreate(&created, 4096, &properties, 0) == CUDA_SUCCESS &&
          created != 0x1234,
      "cuMemCreate did not return a logical handle");
  require(
      cuMemMap(0x1000, 4096, 0, created, 0) == CUDA_SUCCESS &&
          fake_cuda_last_consumed_handle() == 0x1234,
      "cuMemMap did not translate the logical handle");
  require(
      cuMemGetAllocationPropertiesFromHandle(&queried, created) ==
              CUDA_SUCCESS &&
          fake_cuda_last_consumed_handle() == 0x1234,
      "allocation properties query did not translate the logical handle");
  require(
      cuMemExportToShareableHandle(
          &export_fd, created, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) ==
              CUDA_SUCCESS &&
          fake_cuda_last_consumed_handle() == 0x1234,
      "POSIX export did not translate the logical handle");
  fd = dup(export_fd);
  require(fd >= 0, "cannot duplicate fake POSIX sharing FD");
  require(
      cuMemImportFromShareableHandle(
          &imported, (void*)(uintptr_t)fd,
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == CUDA_SUCCESS &&
          imported != created,
      "POSIX import did not return a distinct logical handle");
  close(fd);
  close(export_fd);
  require(
      cuMemMap(0x2000, 4096, 0, imported, 0) == CUDA_SUCCESS,
      "imported logical handle did not map");
  require(
      cuMemRelease(imported) == CUDA_SUCCESS &&
          cuMemRelease(created) == CUDA_SUCCESS &&
          fake_cuda_release_count() == 2,
      "logical handles did not release their real handles exactly once");
  require(
      cuMemMap(0x3000, 4096, 0, unknown, 0) ==
              CUDA_ERROR_INVALID_HANDLE &&
          cuMemExportToShareableHandle(
              &export_fd, unknown,
              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) ==
              CUDA_ERROR_INVALID_HANDLE &&
          cuMemGetAllocationPropertiesFromHandle(&queried, unknown) ==
              CUDA_ERROR_INVALID_HANDLE &&
          cuMemRelease(unknown) == CUDA_ERROR_INVALID_HANDLE,
      "unknown tagged logical handle did not fail closed");
  require(
      fake_cuda_create_count() >= 1 && fake_cuda_import_count() == 1 && fake_cuda_map_count() == 2 &&
          fake_cuda_export_count() == 2 && fake_cuda_logical_forward_count() == 0,
      "logical token reached the fake UMD");
}

static void
test_unshared_create_passthrough(void)
{
  CUmemAllocationProp properties;
  CUmemGenericAllocationHandle handle = 0;

  memset(&properties, 0, sizeof(properties));
  properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  require(
      cuMemCreate(&handle, 4096, &properties, 0) == CUDA_SUCCESS &&
          ((uint64_t)handle & UINT64_C(0xffff000000000000)) !=
              UINT64_C(0xd94d000000000000) &&
          cuMemMap(0x8000, 4096, 0, handle, 0) == CUDA_SUCCESS &&
          cuMemSetAccess(0x8000, 4096, NULL, 0) == CUDA_SUCCESS &&
          cuMemRelease(handle) == CUDA_SUCCESS,
      "unshared real allocation did not pass through");
  require(
      fake_cuda_logical_forward_count() == 0,
      "unshared create passthrough forwarded a logical token");
}

static void
test_admission_succeeds_without_managed_resources(void)
{
  const char* control = getenv("DYN_SNAPSHOT_CONTROL_DIR");
  struct sockaddr_un address = {.sun_family = AF_UNIX};
  struct dyn_vmm_header request = {
      .magic = DYN_VMM_MAGIC,
      .version = DYN_VMM_VERSION,
      .operation = DYN_VMM_INSPECT,
  };
  struct dyn_vmm_header response;
  int client;

  require(
      snprintf(
          address.sun_path, sizeof(address.sun_path), "%s/%s%d.sock", control,
          DYN_VMM_SOCKET_PREFIX, getpid()) < (int)sizeof(address.sun_path),
      "socket path is too long");
  client = socket(AF_UNIX, SOCK_STREAM, 0);
  require(
      client >= 0 &&
          connect(client, (const struct sockaddr*)&address, sizeof(address)) == 0 &&
          write(client, &request, sizeof(request)) == (ssize_t)sizeof(request) &&
          read(client, &response, sizeof(response)) == (ssize_t)sizeof(response),
      "VMM no-resource control exchange failed");
  close(client);
  require(
      response.status == 0 && response.count == 0,
      "unshared allocation poisoned VMM checkpoint admission");
}

static void
test_retained_handle_admission(void)
{
  const char* control = getenv("DYN_SNAPSHOT_CONTROL_DIR");
  CUmemGenericAllocationHandle handle = 0;
  struct sockaddr_un address = {.sun_family = AF_UNIX};
  struct dyn_vmm_header request = {
      .magic = DYN_VMM_MAGIC,
      .version = DYN_VMM_VERSION,
      .operation = DYN_VMM_INSPECT,
  };
  struct dyn_vmm_header response;
  int client;

  require(
      cuMemRetainAllocationHandle(&handle, (void*)(uintptr_t)0x1000) ==
              CUDA_SUCCESS &&
          handle == 0x5678,
      "real cuMemRetainAllocationHandle result was not forwarded");
  require(
      snprintf(
          address.sun_path, sizeof(address.sun_path), "%s/%s%d.sock", control,
          DYN_VMM_SOCKET_PREFIX, getpid()) < (int)sizeof(address.sun_path),
      "socket path is too long");
  client = socket(AF_UNIX, SOCK_STREAM, 0);
  require(
      client >= 0 &&
          connect(client, (const struct sockaddr*)&address, sizeof(address)) == 0 &&
          write(client, &request, sizeof(request)) == (ssize_t)sizeof(request) &&
          read(client, &response, sizeof(response)) == (ssize_t)sizeof(response),
      "VMM control exchange failed");
  close(client);
  require(
      response.status != 0 &&
          strstr(response.message, "retained handles are unsupported") != NULL,
      "successful retained generic handle did not poison checkpoint admission");
}

int
main(int argc, char** argv)
{
  if (argc == 2 && strcmp(argv[1], "unshared") == 0) {
    test_unshared_create_passthrough();
    test_admission_succeeds_without_managed_resources();
    return 0;
  }
  if (argc == 2 && strcmp(argv[1], "retained") == 0) {
    test_retained_handle_admission();
    return 0;
  }
  test_resolver();
  test_ptsz_resolvers();
  test_unavailable_resolver_results();
  test_logical_handles();
  return 0;
}
