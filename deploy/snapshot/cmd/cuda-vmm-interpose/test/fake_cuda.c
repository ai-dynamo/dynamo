/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#undef cuGetProcAddress
#undef cuMulticastAddDevice
#undef cuMulticastCreate
#undef cuMulticastBindMem
#undef cuMulticastUnbind
#undef cudaGetDriverEntryPoint
#undef cudaGetDriverEntryPointByVersion

static void
unknown_entry(void)
{}

static cuuint64_t last_driver_flags;
static unsigned long long last_runtime_flags;
static unsigned int last_runtime_version;
static CUmemGenericAllocationHandle next_handle = 0x1234;
static CUmemGenericAllocationHandle last_consumed_handle;
static unsigned int create_count;
static unsigned int import_count;
static unsigned int release_count;
static unsigned int map_count;
static unsigned int export_count;
static unsigned int logical_forward_count;
static unsigned int active_handle_count;
static unsigned int active_mapping_count;
static unsigned int multicast_bind_count;
static unsigned int multicast_create_count;
static unsigned int multicast_add_count;
static unsigned int multicast_bind_addr_count;
static unsigned int multicast_unbind_count;
static unsigned int active_multicast_count;
static unsigned int active_multicast_bind_count;
static CUmulticastObjectProp multicast_properties[16];
static CUdevice multicast_devices[16];
static CUmemGenericAllocationHandle created_multicast_handles[16];
static CUmemGenericAllocationHandle multicast_handles[16];
static CUmemGenericAllocationHandle multicast_memory_handles[16];
static bool multicast_bind_active[16];
static size_t multicast_offsets[16];
static size_t multicast_memory_offsets[16];
static size_t multicast_sizes[16];
static unsigned long long multicast_flags[16];
static CUdevice multicast_bind_devices[16];
static CUmemGenericAllocationHandle multicast_unbind_handles[16];
static CUdevice multicast_unbind_devices[16];
static size_t multicast_unbind_offsets[16];
static size_t multicast_unbind_sizes[16];
static CUdeviceptr multicast_map_addresses[16];
static unsigned int multicast_map_address_count;
static char multicast_events[64];
static unsigned int multicast_event_count;
static int last_exported_fd = -1;
static int last_internal_export_alias = -1;
static int internal_export_aliases[32];
static CUmemAllocationHandleType handle_types[32];
static CUdevice handle_devices[32];
static CUdevice create_devices[32];
static CUdevice import_device;
static int last_imported_fd = -1;
static unsigned char last_imported_fabric[CU_IPC_HANDLE_SIZE];
static int colliding_export_source = -1;
static CUdeviceptr map_addresses[16];
static CUdeviceptr access_addresses[16];
static CUmemAccessDesc access_descriptors[16];
static size_t access_counts[16];
static unsigned int set_access_count;
static CUdeviceptr copy_destination;
static size_t copy_size;
static unsigned char copy_byte;
static int copy_uniform;
static CUmemGenericAllocationHandle released_handles[16];
static pthread_t initial_thread;
static _Thread_local CUcontext current_context;
static CUcontext last_context;
static int device_count = 1;
static CUdevice device_handles[16];
static CUdevice device_identities[16];
static unsigned int device_get_calls;
static unsigned int device_uuid_calls;
static unsigned int device_count_calls;

unsigned int
fake_cuda_multicast_create_count(void)
{
  return multicast_create_count;
}

unsigned int
fake_cuda_multicast_add_count(void)
{
  return multicast_add_count;
}

unsigned int
fake_cuda_multicast_bind_addr_count(void)
{
  return multicast_bind_addr_count;
}

CUdevice
fake_cuda_multicast_device(unsigned int index)
{
  return index < multicast_add_count ? multicast_devices[index] : -1;
}

unsigned int
fake_cuda_multicast_unbind_count(void)
{
  return multicast_unbind_count;
}

unsigned int
fake_cuda_active_multicast_count(void)
{
  return active_multicast_count;
}

unsigned int
fake_cuda_active_multicast_bind_count(void)
{
  return active_multicast_bind_count;
}

char
fake_cuda_multicast_event(unsigned int index)
{
  return index < multicast_event_count ? multicast_events[index] : '\0';
}

unsigned int
fake_cuda_multicast_event_count(void)
{
  return multicast_event_count;
}

CUmemGenericAllocationHandle
fake_cuda_multicast_handle(unsigned int index)
{
  return index < multicast_bind_count ? multicast_handles[index] : 0;
}

CUmemGenericAllocationHandle
fake_cuda_multicast_memory_handle(unsigned int index)
{
  return index < multicast_bind_count ? multicast_memory_handles[index] : 0;
}

size_t
fake_cuda_multicast_offset(unsigned int index)
{
  return index < multicast_bind_count ? multicast_offsets[index] : 0;
}

size_t
fake_cuda_multicast_memory_offset(unsigned int index)
{
  return index < multicast_bind_count ? multicast_memory_offsets[index] : 0;
}

size_t
fake_cuda_multicast_size(unsigned int index)
{
  return index < multicast_bind_count ? multicast_sizes[index] : 0;
}

unsigned long long
fake_cuda_multicast_flags(unsigned int index)
{
  return index < multicast_bind_count ? multicast_flags[index] : 0;
}

CUdevice
fake_cuda_multicast_bind_device(unsigned int index)
{
  return index < multicast_bind_count ? multicast_bind_devices[index] : -1;
}

CUmemGenericAllocationHandle
fake_cuda_multicast_unbind_handle(unsigned int index)
{
  return index < multicast_unbind_count
      ? multicast_unbind_handles[index]
      : 0;
}

CUdevice
fake_cuda_multicast_unbind_device(unsigned int index)
{
  return index < multicast_unbind_count ? multicast_unbind_devices[index] : -1;
}

size_t
fake_cuda_multicast_unbind_offset(unsigned int index)
{
  return index < multicast_unbind_count ? multicast_unbind_offsets[index] : 0;
}

size_t
fake_cuda_multicast_unbind_size(unsigned int index)
{
  return index < multicast_unbind_count ? multicast_unbind_sizes[index] : 0;
}

static void
record_multicast_event(char event)
{
  if (multicast_event_count < sizeof(multicast_events))
    multicast_events[multicast_event_count] = event;
  multicast_event_count++;
}

static int
known_multicast_handle(CUmemGenericAllocationHandle handle)
{
  size_t index;

  for (index = 0; index < multicast_create_count; index++) {
    if (created_multicast_handles[index] == handle)
      return 1;
  }
  for (index = 0; index < multicast_bind_count; index++) {
    if (multicast_handles[index] == handle)
      return 1;
  }
  return 0;
}

__attribute__((constructor)) static void
initialize_fake_cuda(void)
{
  size_t index;

  initial_thread = pthread_self();
  for (index = 0; index < sizeof(device_identities) / sizeof(device_identities[0]); index++) {
    device_handles[index] = (CUdevice)index;
    device_identities[index] = (CUdevice)index;
  }
}

static int
real_handle(CUmemGenericAllocationHandle handle)
{
  if (((uint64_t)handle & UINT64_C(0xffff000000000000)) ==
      UINT64_C(0xd94d000000000000)) {
    logical_forward_count++;
    return 0;
  }
  last_consumed_handle = handle;
  return 1;
}

unsigned int
fake_cuda_active_handle_count(void)
{
  return active_handle_count;
}

unsigned int
fake_cuda_active_mapping_count(void)
{
  return active_mapping_count;
}

unsigned int
fake_cuda_multicast_bind_count(void)
{
  return multicast_bind_count;
}

CUcontext
fake_cuda_last_context(void)
{
  return last_context;
}

int
fake_cuda_last_exported_fd(void)
{
  return last_exported_fd;
}

int
fake_cuda_last_internal_export_alias(void)
{
  return last_internal_export_alias;
}

int
fake_cuda_internal_export_alias(unsigned int index)
{
  return index < export_count &&
          index < sizeof(internal_export_aliases) / sizeof(internal_export_aliases[0])
      ? internal_export_aliases[index]
      : -1;
}

int
fake_cuda_last_imported_fd(void)
{
  return last_imported_fd;
}

unsigned char
fake_cuda_last_imported_fabric_byte(unsigned int index)
{
  return index < sizeof(last_imported_fabric) ? last_imported_fabric[index] : 0;
}

CUdeviceptr
fake_cuda_map_address(unsigned int index)
{
  return index < map_count ? map_addresses[index] : 0;
}

CUdeviceptr
fake_cuda_access_address(unsigned int index)
{
  return index < 16 ? access_addresses[index] : 0;
}

CUmemAccessDesc
fake_cuda_access_descriptor(unsigned int index)
{
  CUmemAccessDesc empty = {0};

  return index < 16 ? access_descriptors[index] : empty;
}

size_t
fake_cuda_access_count(unsigned int index)
{
  return index < 16 ? access_counts[index] : 0;
}

unsigned int
fake_cuda_set_access_count(void)
{
  return set_access_count;
}

CUdeviceptr
fake_cuda_copy_destination(void)
{
  return copy_destination;
}

size_t
fake_cuda_copy_size(void)
{
  return copy_size;
}

unsigned char
fake_cuda_copy_byte(void)
{
  return copy_byte;
}

int
fake_cuda_copy_uniform(void)
{
  return copy_uniform;
}

CUmemGenericAllocationHandle
fake_cuda_released_handle(unsigned int index)
{
  return index < release_count ? released_handles[index] : 0;
}

void*
fake_cuda_unknown_entry(void)
{
  return (void*)&unknown_entry;
}

cuuint64_t
fake_cuda_last_driver_flags(void)
{
  return last_driver_flags;
}

unsigned long long
fake_cuda_last_runtime_flags(void)
{
  return last_runtime_flags;
}

unsigned int
fake_cuda_last_runtime_version(void)
{
  return last_runtime_version;
}

unsigned int
fake_cuda_create_count(void)
{
  return create_count;
}

CUdevice
fake_cuda_create_device(unsigned int index)
{
  return index < create_count &&
          index < sizeof(create_devices) / sizeof(create_devices[0])
      ? create_devices[index]
      : -1;
}

void
fake_cuda_set_import_device(CUdevice device)
{
  import_device = device;
}

void
fake_cuda_set_device_count(int count)
{
  device_count = count;
}

void
fake_cuda_set_device_handle(int ordinal, CUdevice device)
{
  if (ordinal >= 0 && (size_t)ordinal < sizeof(device_handles) / sizeof(device_handles[0]))
    device_handles[ordinal] = device;
}

void
fake_cuda_set_device_identity(CUdevice device, CUdevice identity)
{
  if (device >= 0 && (size_t)device < sizeof(device_identities) / sizeof(device_identities[0]))
    device_identities[device] = identity;
}

unsigned int
fake_cuda_device_get_calls(void)
{
  return device_get_calls;
}

unsigned int
fake_cuda_device_uuid_calls(void)
{
  return device_uuid_calls;
}

unsigned int
fake_cuda_device_count_calls(void)
{
  return device_count_calls;
}

unsigned int
fake_cuda_import_count(void)
{
  return import_count;
}

unsigned int
fake_cuda_release_count(void)
{
  return release_count;
}

unsigned int
fake_cuda_map_count(void)
{
  return map_count;
}

unsigned int
fake_cuda_export_count(void)
{
  return export_count;
}

unsigned int
fake_cuda_logical_forward_count(void)
{
  return logical_forward_count;
}

CUmemGenericAllocationHandle
fake_cuda_last_consumed_handle(void)
{
  return last_consumed_handle;
}

CUresult CUDAAPI
cuCtxGetCurrent(CUcontext* context)
{
  if (current_context == NULL)
    current_context = pthread_equal(pthread_self(), initial_thread)
        ? (CUcontext)(uintptr_t)1
        : (CUcontext)(uintptr_t)0x77;
  *context = current_context;
  return CUDA_SUCCESS;
}

#if CUDA_VERSION >= 13010
CUresult CUDAAPI
cuMulticastBindAddr_v2(
    CUmemGenericAllocationHandle multicast_handle, CUdevice device,
    size_t multicast_offset, CUdeviceptr address, size_t size,
    unsigned long long flags)
{
  (void)device;
  return cuMulticastBindAddr(
      multicast_handle, multicast_offset, address, size, flags);
}
#endif

CUresult CUDAAPI
cuMulticastBindAddr(
    CUmemGenericAllocationHandle multicast_handle, size_t multicast_offset,
    CUdeviceptr address, size_t size, unsigned long long flags)
{
  (void)multicast_handle;
  (void)multicast_offset;
  (void)address;
  (void)size;
  (void)flags;
  multicast_bind_addr_count++;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuCtxSetCurrent(CUcontext context)
{
  current_context = context;
  last_context = context;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuDeviceGet(CUdevice* device, int ordinal)
{
  device_get_calls++;
  if (device == NULL || ordinal < 0 || ordinal >= device_count ||
      (size_t)ordinal >= sizeof(device_handles) / sizeof(device_handles[0]))
    return CUDA_ERROR_INVALID_DEVICE;
  *device = device_handles[ordinal];
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuDeviceGetUuid_v2(CUuuid* uuid, CUdevice device)
{
  CUdevice identity;
  size_t index;

  device_uuid_calls++;
  if (uuid == NULL || device < 0 || (size_t)device >= sizeof(device_identities) / sizeof(device_identities[0]))
    return CUDA_ERROR_INVALID_DEVICE;
  identity = device_identities[device];
  for (index = 0; index < sizeof(uuid->bytes); index++) uuid->bytes[index] = (char)(index + identity + 1);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuDeviceGetCount(int* count)
{
  device_count_calls++;
  if (count == NULL)
    return CUDA_ERROR_INVALID_VALUE;
  *count = device_count;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuMemCreate(
    CUmemGenericAllocationHandle* output, size_t size,
    const CUmemAllocationProp* properties, unsigned long long flags)
{
  size_t index;

  (void)size;
  (void)flags;
  if (create_count < sizeof(create_devices) / sizeof(create_devices[0]))
    create_devices[create_count] = properties->location.id;
  create_count++;
  active_handle_count++;
  *output = next_handle++;
  index = (size_t)(*output - UINT64_C(0x1234));
  if (index < sizeof(handle_types) / sizeof(handle_types[0])) {
    handle_types[index] = properties->requestedHandleTypes;
    handle_devices[index] = properties->location.id;
  }
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuMemRelease(CUmemGenericAllocationHandle handle)
{
  if (!real_handle(handle))
    return CUDA_ERROR_INVALID_HANDLE;
  if (release_count < sizeof(released_handles) / sizeof(released_handles[0]))
    released_handles[release_count] = handle;
  release_count++;
  active_handle_count--;
  if (known_multicast_handle(handle)) {
    for (size_t binding = 0; binding < multicast_bind_count; binding++) {
      if (multicast_bind_active[binding] &&
          multicast_handles[binding] == handle) {
        multicast_bind_active[binding] = false;
        active_multicast_bind_count--;
      }
    }
  }
  for (size_t index = 0; index < multicast_create_count; index++) {
    if (created_multicast_handles[index] == handle) {
      active_multicast_count--;
      record_multicast_event('r');
      break;
    }
  }
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuMemMap(
    CUdeviceptr address, size_t size, size_t offset,
    CUmemGenericAllocationHandle handle, unsigned long long flags)
{
  (void)address;
  (void)size;
  (void)offset;
  (void)flags;
  if (!real_handle(handle))
    return CUDA_ERROR_INVALID_HANDLE;
  if (map_count < sizeof(map_addresses) / sizeof(map_addresses[0]))
    map_addresses[map_count] = address;
  map_count++;
  active_mapping_count++;
  if (known_multicast_handle(handle)) {
    if (multicast_map_address_count <
        sizeof(multicast_map_addresses) /
            sizeof(multicast_map_addresses[0]))
      multicast_map_addresses[multicast_map_address_count] = address;
    multicast_map_address_count++;
    record_multicast_event('m');
  }
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuMemUnmap(CUdeviceptr address, size_t size)
{
  (void)address;
  (void)size;
  active_mapping_count--;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuMemSetAccess(
    CUdeviceptr address, size_t size, const CUmemAccessDesc* descriptors,
    size_t count)
{
  (void)size;

  if (set_access_count < sizeof(access_addresses) / sizeof(access_addresses[0])) {
    access_addresses[set_access_count] = address;
    access_counts[set_access_count] = count;
    if (count != 0)
      access_descriptors[set_access_count] = descriptors[0];
  }
  set_access_count++;
  for (size_t index = 0; index < multicast_map_address_count; index++) {
    if (multicast_map_addresses[index] == address) {
      record_multicast_event('x');
      break;
    }
  }
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuMemExportToShareableHandle(
    void* shareable, CUmemGenericAllocationHandle handle,
    CUmemAllocationHandleType type, unsigned long long flags)
{
  int fd;
  size_t index;

  (void)flags;
  if (!real_handle(handle))
    return CUDA_ERROR_INVALID_HANDLE;
  if (type == CU_MEM_HANDLE_TYPE_FABRIC) {
    for (index = 0; index < CU_IPC_HANDLE_SIZE; index++)
      ((unsigned char*)shareable)[index] = (unsigned char)(0xa0U + index);
    export_count++;
    return CUDA_SUCCESS;
  }
  if (type != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    return CUDA_ERROR_INVALID_HANDLE;
  if (getenv("FAKE_CUDA_COLLIDE_EXPORT_IDENTITY") != NULL) {
    if (colliding_export_source < 0)
      colliding_export_source = memfd_create("fake-cuda-vmm-collision", MFD_CLOEXEC);
    fd = colliding_export_source < 0 ? -1 : fcntl(colliding_export_source, F_DUPFD_CLOEXEC, 0);
  } else {
    fd = memfd_create("fake-cuda-vmm", MFD_CLOEXEC);
  }
  if (fd < 0)
    return CUDA_ERROR_UNKNOWN;
  last_internal_export_alias = fcntl(fd, F_DUPFD_CLOEXEC, 0);
  if (last_internal_export_alias < 0) {
    close(fd);
    return CUDA_ERROR_UNKNOWN;
  }
  *(int*)shareable = fd;
  last_exported_fd = fd;
  if (export_count < sizeof(internal_export_aliases) / sizeof(internal_export_aliases[0]))
    internal_export_aliases[export_count] = last_internal_export_alias;
  export_count++;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuMemImportFromShareableHandle(
    CUmemGenericAllocationHandle* output, void* os_handle,
    CUmemAllocationHandleType type)
{
  struct stat status;
  int fd = (int)(uintptr_t)os_handle;
  size_t index;

  if (type == CU_MEM_HANDLE_TYPE_FABRIC) {
    memcpy(last_imported_fabric, os_handle, sizeof(last_imported_fabric));
  } else if (type != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR || fstat(fd, &status) != 0) {
    return CUDA_ERROR_INVALID_HANDLE;
  } else {
    last_imported_fd = fd;
  }
  import_count++;
  active_handle_count++;
  *output = next_handle++;
  index = (size_t)(*output - UINT64_C(0x1234));
  if (index < sizeof(handle_types) / sizeof(handle_types[0])) {
    handle_types[index] = CU_MEM_HANDLE_TYPE_NONE;
    handle_devices[index] = import_device;
  }
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuMemGetAllocationPropertiesFromHandle(
    CUmemAllocationProp* properties, CUmemGenericAllocationHandle handle)
{
  size_t index = (size_t)(handle - UINT64_C(0x1234));

  if (!real_handle(handle))
    return CUDA_ERROR_INVALID_HANDLE;
  memset(properties, 0, sizeof(*properties));
  properties->type = CU_MEM_ALLOCATION_TYPE_PINNED;
  properties->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  properties->location.id =
      index < sizeof(handle_devices) / sizeof(handle_devices[0])
          ? handle_devices[index]
          : 0;
  properties->requestedHandleTypes =
      index < sizeof(handle_types) / sizeof(handle_types[0])
          ? handle_types[index]
          : CU_MEM_HANDLE_TYPE_NONE;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuMemcpyDtoH_v2(void* destination, CUdeviceptr source, size_t size)
{
  (void)source;
  memset(destination, 0x5a, size);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuMemcpyHtoD_v2(CUdeviceptr destination, const void* source, size_t size)
{
  const unsigned char* bytes = source;
  size_t index;

  copy_destination = destination;
  copy_size = size;
  copy_byte = size == 0 ? 0 : bytes[0];
  copy_uniform = 1;
  for (index = 1; index < size; index++) {
    if (bytes[index] != copy_byte) {
      copy_uniform = 0;
      break;
    }
  }
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuIpcGetMemHandle(CUipcMemHandle* handle, CUdeviceptr address)
{
  (void)address;
  memset(handle, 0x7b, sizeof(*handle));
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuMulticastCreate(
    CUmemGenericAllocationHandle* output,
    const CUmulticastObjectProp* properties)
{
  unsigned int index = multicast_create_count++;

  active_handle_count++;
  active_multicast_count++;
  *output = next_handle++;
  if (index < sizeof(multicast_properties) / sizeof(multicast_properties[0])) {
    multicast_properties[index] = *properties;
    created_multicast_handles[index] = *output;
  }
  record_multicast_event('c');
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuMulticastAddDevice(
    CUmemGenericAllocationHandle handle, CUdevice device)
{
  if (!real_handle(handle))
    return CUDA_ERROR_INVALID_HANDLE;
  if (multicast_add_count <
      sizeof(multicast_devices) / sizeof(multicast_devices[0]))
    multicast_devices[multicast_add_count] = device;
  multicast_add_count++;
  record_multicast_event('a');
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuMulticastBindMem(
    CUmemGenericAllocationHandle multicast_handle, size_t multicast_offset,
    CUmemGenericAllocationHandle memory_handle, size_t memory_offset,
    size_t size, unsigned long long flags)
{
  if (!real_handle(multicast_handle) || !real_handle(memory_handle))
    return CUDA_ERROR_INVALID_HANDLE;
  if (multicast_bind_count <
      sizeof(multicast_handles) / sizeof(multicast_handles[0])) {
    multicast_handles[multicast_bind_count] = multicast_handle;
    multicast_memory_handles[multicast_bind_count] = memory_handle;
    multicast_offsets[multicast_bind_count] = multicast_offset;
    multicast_memory_offsets[multicast_bind_count] = memory_offset;
    multicast_sizes[multicast_bind_count] = size;
    multicast_flags[multicast_bind_count] = flags;
    multicast_bind_devices[multicast_bind_count] = -1;
    multicast_bind_active[multicast_bind_count] = true;
  }
  multicast_bind_count++;
  active_multicast_bind_count++;
  record_multicast_event('b');
  return CUDA_SUCCESS;
}

#if CUDA_VERSION >= 13010
CUresult CUDAAPI
cuMulticastBindMem_v2(
    CUmemGenericAllocationHandle multicast_handle, CUdevice device,
    size_t multicast_offset, CUmemGenericAllocationHandle memory_handle,
    size_t memory_offset, size_t size, unsigned long long flags)
{
  CUresult result = cuMulticastBindMem(
      multicast_handle, multicast_offset, memory_handle, memory_offset, size,
      flags);
  if (result == CUDA_SUCCESS &&
      multicast_bind_count <=
          sizeof(multicast_bind_devices) / sizeof(multicast_bind_devices[0]))
    multicast_bind_devices[multicast_bind_count - 1] = device;
  return result;
}
#endif

CUresult CUDAAPI
cuMulticastUnbind(
    CUmemGenericAllocationHandle multicast_handle, CUdevice device,
    size_t multicast_offset, size_t size)
{
  (void)device;
  (void)multicast_offset;
  (void)size;
  if (!real_handle(multicast_handle))
    return CUDA_ERROR_INVALID_HANDLE;
  if (multicast_unbind_count <
      sizeof(multicast_unbind_handles) /
          sizeof(multicast_unbind_handles[0])) {
    multicast_unbind_handles[multicast_unbind_count] = multicast_handle;
    multicast_unbind_devices[multicast_unbind_count] = device;
    multicast_unbind_offsets[multicast_unbind_count] = multicast_offset;
    multicast_unbind_sizes[multicast_unbind_count] = size;
  }
  multicast_unbind_count++;
  for (size_t index = 0; index < multicast_bind_count; index++) {
    if (multicast_bind_active[index] &&
        multicast_handles[index] == multicast_handle) {
      multicast_bind_active[index] = false;
      active_multicast_bind_count--;
      break;
    }
  }
  record_multicast_event('u');
  return CUDA_SUCCESS;
}

void*
fake_cuda_ipc_entry(void)
{
  return (void*)&cuIpcGetMemHandle;
}

CUresult CUDAAPI
__attribute__((visibility("protected")))
cuGetProcAddress(
    const char* symbol, void** output, int version, cuuint64_t flags)
{
  (void)version;
  last_driver_flags = flags;
  if (version == 222) {
    *output = NULL;
    return CUDA_SUCCESS;
  }
  *output = strcmp(symbol, "cuIpcGetMemHandle") == 0
      ? (void*)&cuIpcGetMemHandle
      : (void*)&unknown_entry;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
__attribute__((visibility("protected")))
cuGetProcAddress_v2(
    const char* symbol, void** output, int version, cuuint64_t flags,
    CUdriverProcAddressQueryResult* status)
{
  last_driver_flags = flags;
  if (status != NULL) {
    *status = version == 111
        ? (CUdriverProcAddressQueryResult)17
        : CU_GET_PROC_ADDRESS_SUCCESS;
  }
  if (version == 222)
    *output = NULL;
  else if (version == 111)
    *output = (void*)&unknown_entry;
  else if (strcmp(symbol, "cuIpcGetMemHandle") == 0)
    *output = (void*)&cuIpcGetMemHandle;
  else if (strcmp(symbol, "cuMemCreate") == 0)
    *output = (void*)&cuMemCreate;
  else if (strcmp(symbol, "cuMemRelease") == 0)
    *output = (void*)&cuMemRelease;
  else if (strcmp(symbol, "cuMemMap") == 0)
    *output = (void*)&cuMemMap;
  else if (strcmp(symbol, "cuMemUnmap") == 0)
    *output = (void*)&cuMemUnmap;
  else if (strcmp(symbol, "cuMemSetAccess") == 0)
    *output = (void*)&cuMemSetAccess;
  else if (strcmp(symbol, "cuMemExportToShareableHandle") == 0)
    *output = (void*)&cuMemExportToShareableHandle;
  else if (strcmp(symbol, "cuMemImportFromShareableHandle") == 0)
    *output = (void*)&cuMemImportFromShareableHandle;
  else if (strcmp(symbol, "cuMemGetAllocationPropertiesFromHandle") == 0)
    *output = (void*)&cuMemGetAllocationPropertiesFromHandle;
  else if (strcmp(symbol, "cuCtxGetCurrent") == 0)
    *output = (void*)&cuCtxGetCurrent;
  else if (strcmp(symbol, "cuCtxSetCurrent") == 0)
    *output = (void*)&cuCtxSetCurrent;
  else if (strcmp(symbol, "cuDeviceGet") == 0)
    *output = (void*)&cuDeviceGet;
  else if (strcmp(symbol, "cuDeviceGetUuid_v2") == 0)
    *output = (void*)&cuDeviceGetUuid_v2;
  else if (strcmp(symbol, "cuDeviceGetCount") == 0)
    *output = (void*)&cuDeviceGetCount;
  else if (strcmp(symbol, "cuMemcpyDtoH_v2") == 0)
    *output = (void*)&cuMemcpyDtoH_v2;
  else if (strcmp(symbol, "cuMemcpyHtoD_v2") == 0)
    *output = (void*)&cuMemcpyHtoD_v2;
  else if (strcmp(symbol, "cuMulticastCreate") == 0)
    *output = (void*)&cuMulticastCreate;
  else if (strcmp(symbol, "cuMulticastAddDevice") == 0)
    *output = (void*)&cuMulticastAddDevice;
  else if (strcmp(symbol, "cuMulticastUnbind") == 0)
    *output = (void*)&cuMulticastUnbind;
  else if (strcmp(symbol, "cuMulticastBindAddr") == 0) {
#if CUDA_VERSION >= 13010
    if (version >= 13010)
      *output = (void*)&cuMulticastBindAddr_v2;
    else
#endif
      *output = (void*)&cuMulticastBindAddr;
  }
#if CUDA_VERSION >= 13010
  else if (strcmp(symbol, "cuMulticastBindAddr_v2") == 0)
    *output = (void*)&cuMulticastBindAddr_v2;
#endif
  else if (strcmp(symbol, "cuMulticastBindMem") == 0) {
#if CUDA_VERSION >= 13010
    if (version >= 13010)
      *output = (void*)&cuMulticastBindMem_v2;
    else
#endif
      *output = (void*)&cuMulticastBindMem;
  }
#if CUDA_VERSION >= 13010
  else if (strcmp(symbol, "cuMulticastBindMem_v2") == 0)
    *output = (void*)&cuMulticastBindMem_v2;
#endif
  else
    *output = (void*)&unknown_entry;
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuGetProcAddress_v2_ptsz(
    const char* symbol, void** output, int version, cuuint64_t flags,
    CUdriverProcAddressQueryResult* status)
{
  (void)symbol;
  (void)output;
  (void)version;
  (void)flags;
  (void)status;
  return CUDA_ERROR_NOT_SUPPORTED;
}

cudaError_t CUDARTAPI
cudaGetDriverEntryPoint(
    const char* symbol, void** output, unsigned long long flags,
    enum cudaDriverEntryPointQueryResult* status)
{
  (void)symbol;
  last_runtime_flags = flags;
  if (status != NULL)
    *status = cudaDriverEntryPointSuccess;
  *output = (void*)&unknown_entry;
  return cudaSuccess;
}

cudaError_t CUDARTAPI
cudaGetDriverEntryPointByVersion(
    const char* symbol, void** output, unsigned int version,
    unsigned long long flags, enum cudaDriverEntryPointQueryResult* status)
{
  last_runtime_version = version;
  last_runtime_flags = flags;
  if (status != NULL) {
    *status = version == 111
        ? (enum cudaDriverEntryPointQueryResult)17
        : cudaDriverEntryPointSuccess;
  }
  if (version == 222) {
    *output = NULL;
    return cudaSuccess;
  }
  (void)symbol;
  *output = (void*)&unknown_entry;
  return cudaSuccess;
}

cudaError_t CUDARTAPI
cudaGetDriverEntryPoint_ptsz(
    const char* symbol, void** output, unsigned long long flags,
    enum cudaDriverEntryPointQueryResult* status)
{
  (void)symbol;
  (void)output;
  (void)flags;
  (void)status;
  return cudaErrorNotSupported;
}

cudaError_t CUDARTAPI
cudaGetDriverEntryPointByVersion_ptsz(
    const char* symbol, void** output, unsigned int version,
    unsigned long long flags, enum cudaDriverEntryPointQueryResult* status)
{
  (void)symbol;
  (void)output;
  (void)version;
  (void)flags;
  (void)status;
  return cudaErrorNotSupported;
}

CUresult CUDAAPI
cuMemRetainAllocationHandle(
    CUmemGenericAllocationHandle* output, void* address)
{
  (void)address;
  *output = 0x5678;
  return CUDA_SUCCESS;
}
