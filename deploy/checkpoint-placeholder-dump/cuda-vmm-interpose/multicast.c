/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE

#include "multicast.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef CUresult(CUDAAPI* release_fn)(CUmemGenericAllocationHandle);
typedef CUresult(CUDAAPI* retain_fn)(CUmemGenericAllocationHandle*, void*);
typedef CUresult(CUDAAPI* map_fn)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
typedef CUresult(CUDAAPI* unmap_fn)(CUdeviceptr, size_t);
typedef CUresult(CUDAAPI* access_fn)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
typedef CUresult(CUDAAPI* export_fn)(
    void*, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long);
typedef CUresult(CUDAAPI* import_fn)(CUmemGenericAllocationHandle*, void*, CUmemAllocationHandleType);
typedef CUresult(CUDAAPI* properties_fn)(CUmemAllocationProp*, CUmemGenericAllocationHandle);
typedef CUresult(CUDAAPI* context_get_fn)(CUcontext*);
typedef CUresult(CUDAAPI* context_set_fn)(CUcontext);
typedef CUresult(CUDAAPI* context_device_fn)(CUdevice*);
typedef CUresult(CUDAAPI* create_fn)(CUmemGenericAllocationHandle*, const CUmulticastObjectProp*);
typedef CUresult(CUDAAPI* add_device_fn)(CUmemGenericAllocationHandle, CUdevice);
typedef CUresult(CUDAAPI* bind_mem_fn)(
    CUmemGenericAllocationHandle, size_t, CUmemGenericAllocationHandle, size_t, size_t, unsigned long long);
typedef CUresult(CUDAAPI* bind_address_fn)(
    CUmemGenericAllocationHandle, size_t, CUdeviceptr, size_t, unsigned long long);
#if CUDA_VERSION >= 13010
typedef CUresult(CUDAAPI* bind_mem_v2_fn)(
    CUmemGenericAllocationHandle, CUdevice, size_t, CUmemGenericAllocationHandle, size_t, size_t, unsigned long long);
typedef CUresult(CUDAAPI* bind_address_v2_fn)(
    CUmemGenericAllocationHandle, CUdevice, size_t, CUdeviceptr, size_t, unsigned long long);
#endif
typedef CUresult(CUDAAPI* unbind_fn)(CUmemGenericAllocationHandle, CUdevice, size_t, size_t);

struct multicast;

struct multicast_handle {
  CUmemGenericAllocationHandle logical;
  CUmemGenericAllocationHandle driver;
  bool live;
  struct multicast* multicast;
  struct multicast_handle* next;
};

struct multicast_device {
  CUdevice device;
  struct multicast_device* next;
};

struct multicast_binding {
  uint8_t member_id[SNAPSHOT_VMM_ALLOCATION_ID_SIZE];
  CUdeviceptr member_address;
  size_t multicast_offset;
  size_t member_offset;
  size_t size;
  unsigned long long flags;
  CUdevice device;
  uint8_t kind;
  uint8_t api_version;
  bool bound;
  bool checkpointed;
  struct multicast_binding* next;
};

struct multicast_mapping {
  CUdeviceptr address;
  size_t size;
  size_t offset;
  unsigned long long flags;
  CUmemAccessDesc access[SNAPSHOT_VMM_MAX_ACCESS];
  size_t access_count;
  bool mapped;
  bool checkpointed;
  struct multicast_mapping* next;
};

struct multicast {
  uint8_t id[SNAPSHOT_VMM_ALLOCATION_ID_SIZE];
  uint8_t authorization[SNAPSHOT_VMM_TOKEN_SIZE];
  char creator_participant[SNAPSHOT_VMM_ID_SIZE];
  char creator_endpoint[sizeof(((struct sockaddr_un*)0)->sun_path)];
  CUmulticastObjectProp properties;
  CUcontext context;
  CUmemGenericAllocationHandle restore_handle;
  bool creator;
  bool checkpointed;
  struct multicast_device* devices;
  struct multicast_binding* bindings;
  struct multicast_mapping* mappings;
  struct multicast* next;
};

struct context_scope {
  CUcontext previous;
  bool changed;
};

static struct snapshot_multicast_callbacks operations;
static const char* current_participant;
static const char* current_endpoint;
static struct multicast* multicasts;
static struct multicast_handle* handles;
static char failure[128];

static void*
symbol(const char* name)
{
  return operations.real_symbol == NULL ? NULL : operations.real_symbol(name);
}

static CUresult
unavailable(void)
{
  return CUDA_ERROR_NOT_INITIALIZED;
}

static int
fail(const char* operation, CUresult result)
{
  if (result == CUDA_SUCCESS)
    snprintf(failure, sizeof(failure), "%s", operation);
  else
    snprintf(failure, sizeof(failure), "%s failed: CUresult=%d", operation, (int)result);
  return -1;
}

static struct multicast*
find_multicast(const uint8_t id[SNAPSHOT_VMM_ALLOCATION_ID_SIZE])
{
  struct multicast* multicast;

  for (multicast = multicasts; multicast != NULL; multicast = multicast->next) {
    if (memcmp(multicast->id, id, SNAPSHOT_VMM_ALLOCATION_ID_SIZE) == 0)
      return multicast;
  }
  return NULL;
}

static struct multicast_handle*
find_handle(CUmemGenericAllocationHandle logical)
{
  struct multicast_handle* handle;

  for (handle = handles; handle != NULL; handle = handle->next) {
    if (handle->logical == logical)
      return handle;
  }
  return NULL;
}

static struct multicast_mapping*
find_mapping(CUdeviceptr address, size_t size)
{
  struct multicast* multicast;

  for (multicast = multicasts; multicast != NULL; multicast = multicast->next) {
    struct multicast_mapping* mapping;
    for (mapping = multicast->mappings; mapping != NULL; mapping = mapping->next) {
      if (mapping->mapped && mapping->address == address && mapping->size == size)
        return mapping;
    }
  }
  return NULL;
}

static struct multicast_mapping*
find_mapping_at(CUdeviceptr address)
{
  struct multicast* multicast;

  for (multicast = multicasts; multicast != NULL; multicast = multicast->next) {
    struct multicast_mapping* mapping;
    for (mapping = multicast->mappings; mapping != NULL; mapping = mapping->next) {
      if (mapping->mapped && address >= mapping->address && address < mapping->address + mapping->size)
        return mapping;
    }
  }
  return NULL;
}

static struct multicast*
mapping_owner(const struct multicast_mapping* expected)
{
  struct multicast* multicast;

  for (multicast = multicasts; multicast != NULL; multicast = multicast->next) {
    const struct multicast_mapping* mapping;
    for (mapping = multicast->mappings; mapping != NULL; mapping = mapping->next) {
      if (mapping == expected)
        return multicast;
    }
  }
  return NULL;
}

static size_t
live_handle_count(const struct multicast* multicast)
{
  const struct multicast_handle* handle;
  size_t count = 0;

  for (handle = handles; handle != NULL; handle = handle->next) {
    if (handle->live && handle->multicast == multicast)
      count++;
  }
  return count;
}

static CUmemGenericAllocationHandle
current_driver(const struct multicast* multicast)
{
  const struct multicast_handle* handle;

  if (multicast->restore_handle != 0)
    return multicast->restore_handle;
  for (handle = handles; handle != NULL; handle = handle->next) {
    if (handle->live && handle->multicast == multicast && handle->driver != 0)
      return handle->driver;
  }
  return 0;
}

static bool
driver_used(const struct multicast_handle* except, CUmemGenericAllocationHandle driver)
{
  const struct multicast_handle* handle;

  for (handle = handles; handle != NULL; handle = handle->next) {
    if (handle != except && handle->live && handle->driver == driver)
      return true;
  }
  return false;
}

static bool
matches_capability(const struct multicast* multicast, const struct snapshot_vmm_posix_capability* capability)
{
  return memcmp(multicast->authorization, capability->authorization, sizeof(multicast->authorization)) == 0 &&
         strcmp(multicast->creator_participant, capability->creator_participant) == 0 &&
         strcmp(multicast->creator_endpoint, capability->creator_endpoint) == 0 &&
         multicast->properties.numDevices == capability->num_devices &&
         multicast->properties.size == capability->allocation_size &&
         multicast->properties.handleTypes == capability->handle_types &&
         multicast->properties.flags == capability->object_flags;
}

static bool
active(const struct multicast* multicast)
{
  const struct multicast_binding* binding;
  const struct multicast_mapping* mapping;

  if (live_handle_count(multicast) != 0)
    return true;
  for (binding = multicast->bindings; binding != NULL; binding = binding->next) {
    if (binding->bound)
      return true;
  }
  for (mapping = multicast->mappings; mapping != NULL; mapping = mapping->next) {
    if (mapping->mapped)
      return true;
  }
  return false;
}

static int
add_handle(struct multicast* multicast, CUmemGenericAllocationHandle driver, CUmemGenericAllocationHandle* logical)
{
  struct multicast_handle* handle = calloc(1, sizeof(*handle));

  if (handle == NULL || operations.allocate_logical_handle == NULL ||
      operations.allocate_logical_handle(logical) != 0) {
    free(handle);
    return -1;
  }
  handle->logical = *logical;
  handle->driver = driver;
  handle->live = true;
  handle->multicast = multicast;
  handle->next = handles;
  handles = handle;
  return 0;
}

static void
install_driver(struct multicast* multicast, CUmemGenericAllocationHandle driver)
{
  struct multicast_handle* handle;

  for (handle = handles; handle != NULL; handle = handle->next) {
    if (handle->live && handle->multicast == multicast)
      handle->driver = driver;
  }
  multicast->restore_handle = driver;
}

static int
enter_context(CUcontext context, struct context_scope* scope)
{
  context_get_fn get_current = (context_get_fn)symbol("cuCtxGetCurrent");
  context_set_fn set_current = (context_set_fn)symbol("cuCtxSetCurrent");

  memset(scope, 0, sizeof(*scope));
  if (get_current == NULL || set_current == NULL || get_current(&scope->previous) != CUDA_SUCCESS)
    return -1;
  if (scope->previous != context) {
    if (set_current(context) != CUDA_SUCCESS)
      return -1;
    scope->changed = true;
  }
  return 0;
}

static int
leave_context(const struct context_scope* scope)
{
  context_set_fn set_current = (context_set_fn)symbol("cuCtxSetCurrent");

  return !scope->changed || (set_current != NULL && set_current(scope->previous) == CUDA_SUCCESS) ? 0 : -1;
}

static int
capture_context(CUcontext* context)
{
  context_get_fn get_current = (context_get_fn)symbol("cuCtxGetCurrent");

  return get_current != NULL && get_current(context) == CUDA_SUCCESS && *context != NULL ? 0 : -1;
}

static int
capture_device(CUdevice* device)
{
  context_device_fn get_device = (context_device_fn)symbol("cuCtxGetDevice");

  return get_device != NULL && get_device(device) == CUDA_SUCCESS ? 0 : -1;
}

static int
fill_capability(const struct multicast* multicast, struct snapshot_vmm_posix_capability* capability)
{
  memset(capability, 0, sizeof(*capability));
  capability->magic = SNAPSHOT_VMM_POSIX_CAPABILITY_MAGIC;
  capability->version = SNAPSHOT_VMM_POSIX_CAPABILITY_VERSION;
  capability->resource_kind = SNAPSHOT_VMM_RESOURCE_MULTICAST;
  snprintf(
      capability->creator_participant, sizeof(capability->creator_participant), "%s", multicast->creator_participant);
  memcpy(capability->allocation_id, multicast->id, sizeof(capability->allocation_id));
  snprintf(capability->creator_endpoint, sizeof(capability->creator_endpoint), "%s", multicast->creator_endpoint);
  memcpy(capability->authorization, multicast->authorization, sizeof(capability->authorization));
  capability->allocation_size = multicast->properties.size;
  capability->handle_types = multicast->properties.handleTypes;
  capability->object_flags = multicast->properties.flags;
  capability->num_devices = multicast->properties.numDevices;
  return 0;
}

static CUresult
bind_memory(
    CUmemGenericAllocationHandle driver, const struct multicast_binding* binding, CUmemGenericAllocationHandle member)
{
  CUresult result;

  /* Bind waits for the complete multicast team, so it must not block the creator endpoint under state_lock. */
#if CUDA_VERSION >= 13010
  if (binding->api_version == 2) {
    bind_mem_v2_fn function = (bind_mem_v2_fn)symbol("cuMulticastBindMem_v2");
    if (function == NULL)
      return unavailable();
    operations.release_state_lock();
    result = function(
        driver, binding->device, binding->multicast_offset, member, binding->member_offset, binding->size,
        binding->flags);
    operations.acquire_state_lock();
    return result;
  }
#endif
  {
    bind_mem_fn function = (bind_mem_fn)symbol("cuMulticastBindMem");
    if (function == NULL)
      return unavailable();
    operations.release_state_lock();
    result = function(driver, binding->multicast_offset, member, binding->member_offset, binding->size, binding->flags);
    operations.acquire_state_lock();
    return result;
  }
}

static CUresult
bind_address(CUmemGenericAllocationHandle driver, const struct multicast_binding* binding)
{
  CUresult result;

#if CUDA_VERSION >= 13010
  if (binding->api_version == 2) {
    bind_address_v2_fn function = (bind_address_v2_fn)symbol("cuMulticastBindAddr_v2");
    if (function == NULL)
      return unavailable();
    operations.release_state_lock();
    result = function(
        driver, binding->device, binding->multicast_offset, binding->member_address, binding->size, binding->flags);
    operations.acquire_state_lock();
    return result;
  }
#endif
  {
    bind_address_fn function = (bind_address_fn)symbol("cuMulticastBindAddr");
    if (function == NULL)
      return unavailable();
    operations.release_state_lock();
    result = function(driver, binding->multicast_offset, binding->member_address, binding->size, binding->flags);
    operations.acquire_state_lock();
    return result;
  }
}

void
snapshot_multicast_initialize(
    const struct snapshot_multicast_callbacks* callbacks, const char* participant_id, const char* endpoint)
{
  operations = *callbacks;
  current_participant = participant_id;
  current_endpoint = endpoint;
}

void
snapshot_multicast_reset(void)
{
  multicasts = NULL;
  handles = NULL;
  failure[0] = '\0';
}

bool
snapshot_multicast_is_handle(CUmemGenericAllocationHandle logical)
{
  return find_handle(logical) != NULL;
}

bool
snapshot_multicast_has_mapping(CUdeviceptr address, size_t size)
{
  return find_mapping(address, size) != NULL;
}

CUresult
snapshot_multicast_release(CUmemGenericAllocationHandle logical)
{
  release_fn release = (release_fn)symbol("cuMemRelease");
  struct multicast_handle* handle = find_handle(logical);
  CUresult result;

  if (handle == NULL || !handle->live)
    return CUDA_ERROR_INVALID_HANDLE;
  if (handle->driver == 0)
    return CUDA_ERROR_NOT_READY;
  result = CUDA_SUCCESS;
  if (!driver_used(handle, handle->driver))
    result = release == NULL ? unavailable() : release(handle->driver);
  if (result == CUDA_SUCCESS)
    handle->live = false;
  return result;
}

CUresult
snapshot_multicast_retain(CUmemGenericAllocationHandle* output, void* address)
{
  retain_fn retain = (retain_fn)symbol("cuMemRetainAllocationHandle");
  release_fn release = (release_fn)symbol("cuMemRelease");
  struct multicast_mapping* mapping = find_mapping_at((CUdeviceptr)(uintptr_t)address);
  struct multicast* multicast;
  CUmemGenericAllocationHandle driver = 0;
  CUmemGenericAllocationHandle existing;
  CUmemGenericAllocationHandle logical;
  CUresult result;

  if (output == NULL)
    return CUDA_ERROR_INVALID_VALUE;
  if (mapping == NULL)
    return CUDA_ERROR_INVALID_VALUE;
  multicast = mapping_owner(mapping);
  if (multicast == NULL || retain == NULL)
    return unavailable();
  existing = current_driver(multicast);
  result = retain(&driver, address);
  if (result != CUDA_SUCCESS)
    return result;
  if (existing != 0) {
    result = release == NULL ? unavailable() : release(driver);
    if (result != CUDA_SUCCESS)
      return result;
    driver = existing;
  }
  if (add_handle(multicast, driver, &logical) != 0) {
    if (existing == 0 && release != NULL)
      (void)release(driver);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  *output = logical;
  return CUDA_SUCCESS;
}

CUresult
snapshot_multicast_map(
    CUdeviceptr address, size_t size, size_t offset, CUmemGenericAllocationHandle logical, unsigned long long flags)
{
  map_fn map = (map_fn)symbol("cuMemMap");
  struct multicast_handle* handle = find_handle(logical);
  struct multicast_mapping* mapping;
  CUresult result;

  if (handle == NULL || !handle->live)
    return CUDA_ERROR_INVALID_HANDLE;
  if (handle->driver == 0)
    return CUDA_ERROR_NOT_READY;
  mapping = calloc(1, sizeof(*mapping));
  if (mapping == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;
  result = map == NULL ? unavailable() : map(address, size, offset, handle->driver, flags);
  if (result != CUDA_SUCCESS) {
    free(mapping);
    return result;
  }
  mapping->address = address;
  mapping->size = size;
  mapping->offset = offset;
  mapping->flags = flags;
  mapping->mapped = true;
  mapping->next = handle->multicast->mappings;
  handle->multicast->mappings = mapping;
  return CUDA_SUCCESS;
}

CUresult
snapshot_multicast_unmap(CUdeviceptr address, size_t size)
{
  unmap_fn unmap = (unmap_fn)symbol("cuMemUnmap");
  struct multicast_mapping* mapping = find_mapping(address, size);
  CUresult result;

  if (mapping == NULL)
    return CUDA_ERROR_INVALID_VALUE;
  result = unmap == NULL ? unavailable() : unmap(address, size);
  if (result == CUDA_SUCCESS)
    mapping->mapped = false;
  return result;
}

CUresult
snapshot_multicast_set_access(CUdeviceptr address, size_t size, const CUmemAccessDesc* descriptors, size_t count)
{
  access_fn set_access = (access_fn)symbol("cuMemSetAccess");
  struct multicast_mapping* mapping = find_mapping(address, size);
  CUresult result;

  if (mapping == NULL)
    return CUDA_ERROR_INVALID_VALUE;
  if (count > SNAPSHOT_VMM_MAX_ACCESS)
    return CUDA_ERROR_NOT_SUPPORTED;
  result = set_access == NULL ? unavailable() : set_access(address, size, descriptors, count);
  if (result == CUDA_SUCCESS) {
    memcpy(mapping->access, descriptors, count * sizeof(*descriptors));
    mapping->access_count = count;
  }
  return result;
}

CUresult
snapshot_multicast_export(
    void* shareable, CUmemGenericAllocationHandle logical, CUmemAllocationHandleType type, unsigned long long flags)
{
  struct snapshot_vmm_posix_capability capability;
  struct multicast_handle* handle = find_handle(logical);
  int capability_fd = -1;

  if (shareable == NULL)
    return CUDA_ERROR_INVALID_VALUE;
  if (type != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR || flags != 0)
    return CUDA_ERROR_NOT_SUPPORTED;
  if (handle == NULL || !handle->live || handle->driver == 0)
    return CUDA_ERROR_INVALID_HANDLE;
  if (!handle->multicast->creator)
    return CUDA_ERROR_NOT_SUPPORTED;
  fill_capability(handle->multicast, &capability);
  if (snapshot_vmm_posix_create_capability(&capability, &capability_fd) != 0)
    return CUDA_ERROR_OUT_OF_MEMORY;
  *(int*)shareable = capability_fd;
  return CUDA_SUCCESS;
}

CUresult
snapshot_multicast_import(
    CUmemGenericAllocationHandle* output, const struct snapshot_vmm_posix_capability* capability, int raw_fd)
{
  import_fn import_handle = (import_fn)symbol("cuMemImportFromShareableHandle");
  release_fn release = (release_fn)symbol("cuMemRelease");
  struct multicast* multicast;
  CUcontext context;
  bool created = false;
  bool acquired = true;
  CUmemGenericAllocationHandle driver = 0;
  CUmemGenericAllocationHandle logical;
  CUresult result;

  if (output == NULL || capability == NULL || raw_fd < 0)
    return CUDA_ERROR_INVALID_VALUE;
  if (capability->resource_kind != SNAPSHOT_VMM_RESOURCE_MULTICAST ||
      capability->handle_types != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    return CUDA_ERROR_INVALID_HANDLE;
  result = import_handle == NULL
               ? unavailable()
               : import_handle(&driver, (void*)(uintptr_t)raw_fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  if (result != CUDA_SUCCESS)
    return result;
  multicast = find_multicast(capability->allocation_id);
  if (multicast != NULL && !matches_capability(multicast, capability)) {
    if (release != NULL)
      (void)release(driver);
    return CUDA_ERROR_INVALID_HANDLE;
  }
  if (capture_context(&context) != 0 || (multicast != NULL && multicast->context != context)) {
    if (release != NULL)
      (void)release(driver);
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  if (multicast != NULL) {
    CUmemGenericAllocationHandle existing = current_driver(multicast);
    if (existing != 0) {
      result = release == NULL ? unavailable() : release(driver);
      if (result != CUDA_SUCCESS)
        return result;
      driver = existing;
      acquired = false;
    }
  }
  if (multicast == NULL) {
    multicast = calloc(1, sizeof(*multicast));
    if (multicast != NULL) {
      memcpy(multicast->id, capability->allocation_id, sizeof(multicast->id));
      memcpy(multicast->authorization, capability->authorization, sizeof(multicast->authorization));
      snprintf(
          multicast->creator_participant, sizeof(multicast->creator_participant), "%s",
          capability->creator_participant);
      snprintf(multicast->creator_endpoint, sizeof(multicast->creator_endpoint), "%s", capability->creator_endpoint);
      multicast->properties.numDevices = capability->num_devices;
      multicast->properties.size = capability->allocation_size;
      multicast->properties.handleTypes = capability->handle_types;
      multicast->properties.flags = capability->object_flags;
      multicast->context = context;
      multicast->creator = false;
      created = true;
    }
  }
  if (multicast == NULL || add_handle(multicast, driver, &logical) != 0) {
    if (acquired && release != NULL)
      (void)release(driver);
    if (created)
      free(multicast);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  if (created) {
    multicast->next = multicasts;
    multicasts = multicast;
  }
  *output = logical;
  return CUDA_SUCCESS;
}

CUresult
snapshot_multicast_get_properties(CUmemAllocationProp* properties, CUmemGenericAllocationHandle logical)
{
  properties_fn get_properties = (properties_fn)symbol("cuMemGetAllocationPropertiesFromHandle");
  struct multicast_handle* handle = find_handle(logical);

  if (handle == NULL || !handle->live)
    return CUDA_ERROR_INVALID_HANDLE;
  if (handle->driver == 0)
    return CUDA_ERROR_NOT_READY;
  return get_properties == NULL ? unavailable() : get_properties(properties, handle->driver);
}

CUresult
snapshot_multicast_create(CUmemGenericAllocationHandle* output, const CUmulticastObjectProp* properties)
{
  create_fn create = (create_fn)symbol("cuMulticastCreate");
  release_fn release = (release_fn)symbol("cuMemRelease");
  struct multicast* multicast;
  CUmemGenericAllocationHandle driver = 0;
  CUmemGenericAllocationHandle logical;
  CUresult result;

  if (output == NULL || properties == NULL)
    return CUDA_ERROR_INVALID_VALUE;
  if (properties->handleTypes != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    return CUDA_ERROR_NOT_SUPPORTED;
  result = create == NULL ? unavailable() : create(&driver, properties);
  if (result != CUDA_SUCCESS)
    return result;
  multicast = calloc(1, sizeof(*multicast));
  if (multicast == NULL || operations.random_bytes == NULL ||
      operations.random_bytes(multicast->id, sizeof(multicast->id)) != 0 ||
      operations.random_bytes(multicast->authorization, sizeof(multicast->authorization)) != 0 ||
      capture_context(&multicast->context) != 0 || add_handle(multicast, driver, &logical) != 0) {
    if (release != NULL)
      (void)release(driver);
    free(multicast);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  multicast->properties = *properties;
  multicast->creator = true;
  snprintf(multicast->creator_participant, sizeof(multicast->creator_participant), "%s", current_participant);
  snprintf(multicast->creator_endpoint, sizeof(multicast->creator_endpoint), "%s", current_endpoint);
  multicast->next = multicasts;
  multicasts = multicast;
  *output = logical;
  return CUDA_SUCCESS;
}

CUresult
snapshot_multicast_add_device(CUmemGenericAllocationHandle logical, CUdevice device)
{
  add_device_fn add_device = (add_device_fn)symbol("cuMulticastAddDevice");
  struct multicast_handle* handle = find_handle(logical);
  struct multicast_device* added = calloc(1, sizeof(*added));
  CUresult result;

  if (handle == NULL || !handle->live)
    return CUDA_ERROR_INVALID_HANDLE;
  if (handle->driver == 0)
    return CUDA_ERROR_NOT_READY;
  if (added == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;
  result = add_device == NULL ? unavailable() : add_device(handle->driver, device);
  if (result != CUDA_SUCCESS) {
    free(added);
    return result;
  }
  added->device = device;
  added->next = handle->multicast->devices;
  handle->multicast->devices = added;
  return CUDA_SUCCESS;
}

CUresult
snapshot_multicast_bind_mem(
    CUmemGenericAllocationHandle multicast_logical, CUdevice device, bool device_explicit, size_t multicast_offset,
    CUmemGenericAllocationHandle memory, size_t memory_offset, size_t size, unsigned long long flags)
{
  struct multicast_handle* handle = find_handle(multicast_logical);
  struct snapshot_multicast_member member;
  struct multicast_binding* binding;
  CUresult result;

  if (handle == NULL || !handle->live)
    return CUDA_ERROR_INVALID_HANDLE;
  if (handle->driver == 0)
    return CUDA_ERROR_NOT_READY;
  memset(&member, 0, sizeof(member));
  if (operations.member_from_handle == NULL || operations.member_from_handle(memory, &member) != 0)
    return CUDA_ERROR_NOT_SUPPORTED;
  if (!device_explicit)
    device = member.device;
  binding = calloc(1, sizeof(*binding));
  if (binding == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;
  memcpy(binding->member_id, member.id, sizeof(binding->member_id));
  binding->multicast_offset = multicast_offset;
  binding->member_offset = memory_offset;
  binding->size = size;
  binding->flags = flags;
  binding->device = device;
  binding->kind = SNAPSHOT_VMM_MULTICAST_BIND_MEM;
  binding->api_version = device_explicit ? 2 : 1;
  result = bind_memory(handle->driver, binding, member.handle);
  if (result != CUDA_SUCCESS) {
    free(binding);
    return result;
  }
  if (operations.mark_member_shared != NULL)
    operations.mark_member_shared(member.id);
  binding->bound = true;
  binding->next = handle->multicast->bindings;
  handle->multicast->bindings = binding;
  return CUDA_SUCCESS;
}

CUresult
snapshot_multicast_bind_address(
    CUmemGenericAllocationHandle multicast_logical, CUdevice device, bool device_explicit, size_t multicast_offset,
    CUdeviceptr memory, size_t size, unsigned long long flags)
{
  struct multicast_handle* handle = find_handle(multicast_logical);
  struct snapshot_multicast_member member;
  struct multicast_binding* binding;
  bool tracked_member = false;
  CUresult result;

  if (handle == NULL || !handle->live)
    return CUDA_ERROR_INVALID_HANDLE;
  if (handle->driver == 0)
    return CUDA_ERROR_NOT_READY;
  memset(&member, 0, sizeof(member));
  if (operations.member_from_address == NULL || operations.member_from_address(memory, size, &member) != 0) {
    if (operations.random_bytes == NULL || operations.random_bytes(member.id, sizeof(member.id)) != 0 ||
        (!device_explicit && capture_device(&device) != 0))
      return CUDA_ERROR_NOT_SUPPORTED;
    member.address = memory;
  } else {
    tracked_member = true;
    if (!device_explicit)
      device = member.device;
  }
  binding = calloc(1, sizeof(*binding));
  if (binding == NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;
  memcpy(binding->member_id, member.id, sizeof(binding->member_id));
  binding->member_address = memory;
  binding->multicast_offset = multicast_offset;
  binding->member_offset = member.allocation_offset;
  binding->size = size;
  binding->flags = flags;
  binding->device = device;
  binding->kind = SNAPSHOT_VMM_MULTICAST_BIND_ADDR;
  binding->api_version = device_explicit ? 2 : 1;
  result = bind_address(handle->driver, binding);
  if (result != CUDA_SUCCESS) {
    free(binding);
    return result;
  }
  if (tracked_member && operations.mark_member_shared != NULL)
    operations.mark_member_shared(member.id);
  binding->bound = true;
  binding->next = handle->multicast->bindings;
  handle->multicast->bindings = binding;
  return CUDA_SUCCESS;
}

CUresult
snapshot_multicast_unbind(CUmemGenericAllocationHandle logical, CUdevice device, size_t offset, size_t size)
{
  unbind_fn unbind = (unbind_fn)symbol("cuMulticastUnbind");
  struct multicast_handle* handle = find_handle(logical);
  struct multicast_binding* binding;
  CUresult result;

  if (handle == NULL || !handle->live)
    return CUDA_ERROR_INVALID_HANDLE;
  if (handle->driver == 0)
    return CUDA_ERROR_NOT_READY;
  for (binding = handle->multicast->bindings; binding != NULL; binding = binding->next) {
    if (binding->bound && binding->device == device && binding->multicast_offset == offset && binding->size == size)
      break;
  }
  if (binding == NULL)
    return CUDA_ERROR_NOT_SUPPORTED;
  result = unbind == NULL ? unavailable() : unbind(handle->driver, device, offset, size);
  if (result != CUDA_SUCCESS)
    return result;
  binding->bound = false;
  return CUDA_SUCCESS;
}

size_t
snapshot_multicast_record_count(void)
{
  const struct multicast* multicast;
  size_t count = 0;

  for (multicast = multicasts; multicast != NULL; multicast = multicast->next) {
    const struct multicast_binding* binding;
    const struct multicast_device* device;
    const struct multicast_mapping* mapping;
    if (!active(multicast))
      continue;
    count++;
    for (device = multicast->devices; device != NULL; device = device->next) count++;
    for (binding = multicast->bindings; binding != NULL; binding = binding->next) {
      if (binding->bound)
        count++;
    }
    for (mapping = multicast->mappings; mapping != NULL; mapping = mapping->next) {
      if (mapping->mapped)
        count++;
    }
  }
  return count;
}

int
snapshot_multicast_write_records(struct snapshot_vmm_record* records, size_t count)
{
  const struct multicast* multicast;
  size_t written = 0;

  for (multicast = multicasts; multicast != NULL; multicast = multicast->next) {
    const struct multicast_binding* binding;
    const struct multicast_device* device;
    const struct multicast_mapping* mapping;
    struct snapshot_vmm_record* record;
    size_t index;

    if (!active(multicast))
      continue;
    if (written == count)
      return -1;
    record = &records[written++];
    record->kind = SNAPSHOT_VMM_MULTICAST;
    record->flags = multicast->creator ? SNAPSHOT_VMM_CREATOR : 0;
    if (live_handle_count(multicast) != 0)
      record->flags |= SNAPSHOT_VMM_APPLICATION_HANDLE_LIVE;
    memcpy(record->allocation_id, multicast->id, sizeof(record->allocation_id));
    record->allocation_size = multicast->properties.size;
    record->application_handle_count = (uint32_t)live_handle_count(multicast);
    record->handle_types = multicast->properties.handleTypes;
    record->object_flags = multicast->properties.flags;
    record->num_devices = multicast->properties.numDevices;
    snprintf(record->creator_participant, sizeof(record->creator_participant), "%s", multicast->creator_participant);
    for (device = multicast->devices; device != NULL; device = device->next) {
      if (written == count)
        return -1;
      record = &records[written++];
      record->kind = SNAPSHOT_VMM_MULTICAST_DEVICE;
      memcpy(record->allocation_id, multicast->id, sizeof(record->allocation_id));
      record->device = device->device;
    }
    for (binding = multicast->bindings; binding != NULL; binding = binding->next) {
      if (!binding->bound)
        continue;
      if (written == count)
        return -1;
      record = &records[written++];
      record->kind = SNAPSHOT_VMM_MULTICAST_BINDING;
      memcpy(record->allocation_id, multicast->id, sizeof(record->allocation_id));
      memcpy(record->member_id, binding->member_id, sizeof(record->member_id));
      record->address = binding->member_address;
      record->size = binding->size;
      record->offset = binding->multicast_offset;
      record->member_offset = binding->member_offset;
      record->operation_flags = binding->flags;
      record->binding_kind = binding->kind;
      record->api_version = binding->api_version;
      record->device = binding->device;
    }
    for (mapping = multicast->mappings; mapping != NULL; mapping = mapping->next) {
      if (!mapping->mapped)
        continue;
      if (written == count)
        return -1;
      record = &records[written++];
      record->kind = SNAPSHOT_VMM_MULTICAST_MAPPING;
      memcpy(record->allocation_id, multicast->id, sizeof(record->allocation_id));
      record->address = mapping->address;
      record->size = mapping->size;
      record->offset = mapping->offset;
      record->operation_flags = mapping->flags;
      record->access_count = (uint32_t)mapping->access_count;
      for (index = 0; index < mapping->access_count; index++) {
        record->access[index].location_type = mapping->access[index].location.type;
        record->access[index].location_id = mapping->access[index].location.id;
        record->access[index].flags = mapping->access[index].flags;
      }
    }
  }
  return written == count ? 0 : -1;
}

int
snapshot_multicast_prepare(void)
{
  unmap_fn unmap = (unmap_fn)symbol("cuMemUnmap");
  unbind_fn unbind = (unbind_fn)symbol("cuMulticastUnbind");
  release_fn release = (release_fn)symbol("cuMemRelease");
  struct multicast* multicast;

  failure[0] = '\0';
  if (unmap == NULL || unbind == NULL || release == NULL)
    return fail("multicast teardown symbols are unavailable", CUDA_SUCCESS);
  for (multicast = multicasts; multicast != NULL; multicast = multicast->next) {
    if (active(multicast) && current_driver(multicast) == 0) {
      struct multicast_mapping* mapping;
      bool retainable = false;
      for (mapping = multicast->mappings; mapping != NULL; mapping = mapping->next) {
        if (mapping->mapped) {
          retainable = true;
          break;
        }
      }
      if (!retainable)
        return fail("multicast object has no handle or mapping", CUDA_SUCCESS);
    }
  }
  for (multicast = multicasts; multicast != NULL; multicast = multicast->next) {
    struct multicast_binding* binding;
    struct multicast_handle* handle;
    struct multicast_mapping* mapping;
    struct context_scope scope;
    CUmemGenericAllocationHandle driver;
    bool temporary = false;

    if (!active(multicast))
      continue;
    driver = current_driver(multicast);
    if (enter_context(multicast->context, &scope) != 0)
      return fail("cannot enter multicast context", CUDA_SUCCESS);
    if (driver == 0) {
      retain_fn retain = (retain_fn)symbol("cuMemRetainAllocationHandle");
      for (mapping = multicast->mappings; mapping != NULL; mapping = mapping->next) {
        if (mapping->mapped)
          break;
      }
      if (retain == NULL || mapping == NULL || retain(&driver, (void*)(uintptr_t)mapping->address) != CUDA_SUCCESS) {
        (void)leave_context(&scope);
        return fail("cannot retain multicast teardown handle", CUDA_SUCCESS);
      }
      temporary = true;
    }
    for (mapping = multicast->mappings; mapping != NULL; mapping = mapping->next) {
      CUresult result;
      if (!mapping->mapped)
        continue;
      result = unmap(mapping->address, mapping->size);
      if (result != CUDA_SUCCESS) {
        (void)leave_context(&scope);
        return fail("cuMemUnmap multicast mapping", result);
      }
      mapping->mapped = false;
      mapping->checkpointed = true;
    }
    for (binding = multicast->bindings; binding != NULL; binding = binding->next) {
      CUresult result;
      if (!binding->bound)
        continue;
      result = unbind(driver, binding->device, binding->multicast_offset, binding->size);
      if (result != CUDA_SUCCESS) {
        (void)leave_context(&scope);
        return fail("cuMulticastUnbind", result);
      }
      binding->bound = false;
      binding->checkpointed = true;
    }
    for (handle = handles; handle != NULL; handle = handle->next) {
      CUmemGenericAllocationHandle old;
      CUresult result;
      if (!handle->live || handle->multicast != multicast)
        continue;
      old = handle->driver;
      if (!driver_used(handle, old)) {
        result = release(old);
        if (result != CUDA_SUCCESS) {
          (void)leave_context(&scope);
          return fail("cuMemRelease multicast handle", result);
        }
      }
      handle->driver = 0;
    }
    if (temporary) {
      CUresult result = release(driver);
      if (result != CUDA_SUCCESS) {
        (void)leave_context(&scope);
        return fail("cuMemRelease multicast teardown handle", result);
      }
    }
    multicast->checkpointed = true;
    if (leave_context(&scope) != 0)
      return fail("cannot leave multicast context", CUDA_SUCCESS);
  }
  return 0;
}

int
snapshot_multicast_restore_creators(void)
{
  create_fn create = (create_fn)symbol("cuMulticastCreate");
  struct multicast* multicast;

  failure[0] = '\0';
  if (create == NULL)
    return fail("cuMulticastCreate is unavailable", CUDA_SUCCESS);
  for (multicast = multicasts; multicast != NULL; multicast = multicast->next) {
    struct context_scope scope;
    CUmemGenericAllocationHandle driver = 0;
    CUresult result;
    if (!multicast->checkpointed || !multicast->creator)
      continue;
    if (enter_context(multicast->context, &scope) != 0)
      return fail("cannot enter multicast creator context", CUDA_SUCCESS);
    result = create(&driver, &multicast->properties);
    if (result == CUDA_SUCCESS)
      install_driver(multicast, driver);
    if (leave_context(&scope) != 0)
      return fail("cannot leave multicast creator context", CUDA_SUCCESS);
    if (result != CUDA_SUCCESS)
      return fail("cuMulticastCreate", result);
  }
  return 0;
}

int
snapshot_multicast_restore_importers(void)
{
  import_fn import_handle = (import_fn)symbol("cuMemImportFromShareableHandle");
  release_fn release = (release_fn)symbol("cuMemRelease");
  struct multicast* multicast;

  failure[0] = '\0';
  if (import_handle == NULL || release == NULL)
    return fail("multicast import symbols are unavailable", CUDA_SUCCESS);
  for (multicast = multicasts; multicast != NULL; multicast = multicast->next) {
    struct context_scope scope;
    CUmemGenericAllocationHandle driver = current_driver(multicast);
    CUresult result = CUDA_SUCCESS;

    if (!multicast->checkpointed)
      continue;
    if (enter_context(multicast->context, &scope) != 0)
      return fail("cannot enter multicast importer context", CUDA_SUCCESS);
    if (!multicast->creator) {
      struct snapshot_vmm_posix_capability capability;
      char export_error[sizeof(failure)];
      int raw_fd = -1;

      fill_capability(multicast, &capability);
      if (snapshot_vmm_posix_request_export(&capability, &raw_fd, export_error, sizeof(export_error)) != 0) {
        (void)leave_context(&scope);
        snprintf(
            failure, sizeof(failure), "multicast creator export: %.96s",
            export_error[0] == '\0' ? "request failed" : export_error);
        return -1;
      }
      result = import_handle(&driver, (void*)(uintptr_t)raw_fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
      if (close(raw_fd) != 0 && result == CUDA_SUCCESS) {
        (void)release(driver);
        result = CUDA_ERROR_UNKNOWN;
      }
      if (result == CUDA_SUCCESS)
        install_driver(multicast, driver);
    }
    if (leave_context(&scope) != 0)
      return fail("cannot leave multicast importer context", CUDA_SUCCESS);
    if (result != CUDA_SUCCESS)
      return fail("multicast import", result);
  }
  return 0;
}

int
snapshot_multicast_restore_devices(void)
{
  add_device_fn add_device = (add_device_fn)symbol("cuMulticastAddDevice");
  struct multicast* multicast;

  failure[0] = '\0';
  if (add_device == NULL)
    return fail("cuMulticastAddDevice is unavailable", CUDA_SUCCESS);
  for (multicast = multicasts; multicast != NULL; multicast = multicast->next) {
    struct multicast_device* device;
    struct context_scope scope;
    CUmemGenericAllocationHandle driver;

    if (!multicast->checkpointed)
      continue;
    driver = current_driver(multicast);
    if (driver == 0)
      return fail("imported multicast handle is unavailable", CUDA_SUCCESS);
    if (enter_context(multicast->context, &scope) != 0)
      return fail("cannot enter multicast device context", CUDA_SUCCESS);
    for (device = multicast->devices; device != NULL; device = device->next) {
      CUresult result = add_device(driver, device->device);
      if (result != CUDA_SUCCESS) {
        (void)leave_context(&scope);
        return fail("cuMulticastAddDevice", result);
      }
    }
    if (leave_context(&scope) != 0)
      return fail("cannot leave multicast device context", CUDA_SUCCESS);
  }
  return 0;
}

int
snapshot_multicast_restore_topology(void)
{
  map_fn map = (map_fn)symbol("cuMemMap");
  access_fn set_access = (access_fn)symbol("cuMemSetAccess");
  release_fn release = (release_fn)symbol("cuMemRelease");
  struct multicast* multicast;

  failure[0] = '\0';
  if (map == NULL || set_access == NULL || release == NULL)
    return fail("multicast mapping symbols are unavailable", CUDA_SUCCESS);
  for (multicast = multicasts; multicast != NULL; multicast = multicast->next) {
    struct multicast_binding* binding;
    struct multicast_mapping* mapping;
    struct context_scope scope;
    CUmemGenericAllocationHandle driver;

    if (!multicast->checkpointed)
      continue;
    driver = current_driver(multicast);
    if (driver == 0)
      return fail("restored multicast handle is unavailable", CUDA_SUCCESS);
    if (enter_context(multicast->context, &scope) != 0)
      return fail("cannot enter multicast restore context", CUDA_SUCCESS);
    for (binding = multicast->bindings; binding != NULL; binding = binding->next) {
      struct snapshot_multicast_member member;
      CUresult result;

      if (!binding->checkpointed)
        continue;
      if (binding->kind == SNAPSHOT_VMM_MULTICAST_BIND_MEM) {
        memset(&member, 0, sizeof(member));
        if (operations.member_from_id == NULL || operations.member_from_id(binding->member_id, &member) != 0) {
          (void)leave_context(&scope);
          return fail("restored multicast member is unavailable", CUDA_SUCCESS);
        }
        result = bind_memory(driver, binding, member.handle);
        if (member.temporary_handle) {
          CUresult release_result = release(member.handle);
          if (result == CUDA_SUCCESS)
            result = release_result;
        }
      } else {
        result = bind_address(driver, binding);
      }
      if (result != CUDA_SUCCESS) {
        (void)leave_context(&scope);
        return fail("restore multicast binding", result);
      }
      binding->bound = true;
      binding->checkpointed = false;
    }
    for (mapping = multicast->mappings; mapping != NULL; mapping = mapping->next) {
      CUresult result;
      if (!mapping->checkpointed)
        continue;
      result = map(mapping->address, mapping->size, mapping->offset, driver, mapping->flags);
      if (result == CUDA_SUCCESS && mapping->access_count != 0)
        result = set_access(mapping->address, mapping->size, mapping->access, mapping->access_count);
      if (result != CUDA_SUCCESS) {
        (void)leave_context(&scope);
        return fail("restore multicast mapping or access", result);
      }
      mapping->mapped = true;
      mapping->checkpointed = false;
    }
    multicast->checkpointed = false;
    multicast->restore_handle = 0;
    if (live_handle_count(multicast) == 0) {
      CUresult result = release(driver);
      if (result != CUDA_SUCCESS) {
        (void)leave_context(&scope);
        return fail("cuMemRelease restored multicast handle", result);
      }
    }
    if (leave_context(&scope) != 0)
      return fail("cannot leave multicast restore context", CUDA_SUCCESS);
  }
  return snapshot_multicast_validate_restored();
}

int
snapshot_multicast_validate_restored(void)
{
  const struct multicast_handle* handle;
  const struct multicast* multicast;

  for (handle = handles; handle != NULL; handle = handle->next) {
    if (handle->live && handle->driver == 0)
      return fail("multicast logical handle was not restored", CUDA_SUCCESS);
  }
  for (multicast = multicasts; multicast != NULL; multicast = multicast->next) {
    const struct multicast_binding* binding;
    const struct multicast_mapping* mapping;
    if (multicast->checkpointed)
      return fail("multicast object remains checkpointed", CUDA_SUCCESS);
    for (binding = multicast->bindings; binding != NULL; binding = binding->next) {
      if (binding->checkpointed)
        return fail("multicast binding remains checkpointed", CUDA_SUCCESS);
    }
    for (mapping = multicast->mappings; mapping != NULL; mapping = mapping->next) {
      if (mapping->checkpointed)
        return fail("multicast mapping remains checkpointed", CUDA_SUCCESS);
    }
  }
  return 0;
}

CUresult
snapshot_multicast_export_raw(
    const uint8_t id[SNAPSHOT_VMM_ALLOCATION_ID_SIZE], const uint8_t authorization[SNAPSHOT_VMM_TOKEN_SIZE],
    int* output)
{
  export_fn export_handle = (export_fn)symbol("cuMemExportToShareableHandle");
  retain_fn retain = (retain_fn)symbol("cuMemRetainAllocationHandle");
  release_fn release = (release_fn)symbol("cuMemRelease");
  struct multicast* multicast = find_multicast(id);
  struct multicast_mapping* mapping;
  struct context_scope scope;
  CUmemGenericAllocationHandle driver;
  bool temporary = false;
  CUresult result;

  *output = -1;
  if (multicast == NULL || !multicast->creator ||
      memcmp(multicast->authorization, authorization, sizeof(multicast->authorization)) != 0)
    return CUDA_ERROR_INVALID_HANDLE;
  driver = current_driver(multicast);
  if (export_handle == NULL || enter_context(multicast->context, &scope) != 0)
    return CUDA_ERROR_INVALID_HANDLE;
  if (driver == 0) {
    for (mapping = multicast->mappings; mapping != NULL; mapping = mapping->next) {
      if (mapping->mapped)
        break;
    }
    if (mapping == NULL || retain == NULL || retain(&driver, (void*)(uintptr_t)mapping->address) != CUDA_SUCCESS) {
      (void)leave_context(&scope);
      return CUDA_ERROR_INVALID_HANDLE;
    }
    temporary = true;
  }
  result = export_handle(output, driver, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  if (temporary) {
    CUresult release_result = release == NULL ? CUDA_ERROR_NOT_INITIALIZED : release(driver);
    if (result == CUDA_SUCCESS)
      result = release_result;
  }
  if (result != CUDA_SUCCESS && *output >= 0) {
    close(*output);
    *output = -1;
  }
  if (leave_context(&scope) != 0) {
    if (*output >= 0) {
      close(*output);
      *output = -1;
    }
    return CUDA_ERROR_UNKNOWN;
  }
  return result;
}

const char*
snapshot_multicast_error(void)
{
  return failure[0] == '\0' ? "multicast operation failed" : failure;
}
