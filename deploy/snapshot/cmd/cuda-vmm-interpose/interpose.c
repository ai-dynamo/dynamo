/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE

#include <cuda.h>
#include <dlfcn.h>
#include <errno.h>
#include <link.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include "posix.h"
#include "protocol.h"
#include "util.h"

#undef cuGetProcAddress
#undef cuMulticastBindAddr
#undef cuMulticastBindMem

#define CONTROL_DIR "/snapshot-control"
#define CONTROL_TIMEOUT_SECONDS 30

CUresult CUDAAPI cuGetProcAddress(const char*, void**, int, cuuint64_t);
CUresult CUDAAPI cuGetProcAddress_v2(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
CUresult CUDAAPI cuGetProcAddress_v2_ptsz(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
CUresult CUDAAPI cuMemRetainAllocationHandle(CUmemGenericAllocationHandle*, void*);
CUresult CUDAAPI cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp*, CUmemGenericAllocationHandle);

typedef CUresult(CUDAAPI* create_fn)(
    CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
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

struct allocation;

struct handle {
  CUmemGenericAllocationHandle application;
  CUmemGenericAllocationHandle current;
  bool live;
  struct allocation* allocation;
  struct handle* next;
};

struct mapping {
  CUdeviceptr address;
  size_t size;
  size_t offset;
  CUmemAccessDesc access[SNAPSHOT_VMM_MAX_ACCESS];
  size_t access_count;
  bool mapped;
  bool checkpointed;
  struct allocation* allocation;
  struct mapping* next;
};

struct allocation {
  uint8_t id[SNAPSHOT_VMM_ALLOCATION_ID_SIZE];
  uint8_t authorization[SNAPSHOT_VMM_TOKEN_SIZE];
  char creator_participant[SNAPSHOT_VMM_ID_SIZE];
  char creator_endpoint[sizeof(((struct sockaddr_un*)0)->sun_path)];
  size_t size;
  CUmemAllocationProp properties;
  CUcontext context;
  CUmemGenericAllocationHandle carrier;
  bool creator;
  struct allocation* next;
};

enum phase {
  PHASE_ACTIVE,
  PHASE_CARRIERS,
  PHASE_PREPARED,
  PHASE_CREATORS_RESTORED,
  PHASE_FAILED,
};

struct context_scope {
  CUcontext previous;
  bool changed;
};

static pthread_mutex_t state_lock = PTHREAD_MUTEX_INITIALIZER;
static struct allocation* allocations;
static struct handle* handles;
static struct mapping* mappings;
static enum phase current_phase = PHASE_ACTIVE;
static bool enabled;
static char failure[96];
static char job_id[SNAPSHOT_VMM_ID_SIZE];
static char participant_id[SNAPSHOT_VMM_ID_SIZE];
static char control_directory[sizeof(((struct sockaddr_un*)0)->sun_path)];
static char socket_path[sizeof(((struct sockaddr_un*)0)->sun_path)];
static int listener = -1;
static bool endpoint_needs_initialization;
static pthread_once_t real_dlsym_once = PTHREAD_ONCE_INIT;
static void* (*real_dlsym_function)(void*, const char*);
static _Atomic(uintptr_t) explicit_libcuda_handle;
static _Atomic(uintptr_t) explicit_cu_get_proc_address;
static _Atomic(uintptr_t) explicit_cu_get_proc_address_v2;

static void* replacement(const char*, int);

static void
set_failure(const char* message)
{
  current_phase = PHASE_FAILED;
  snprintf(failure, sizeof(failure), "%s", message);
}

static void
set_importer_failure(const char* operation, CUresult result)
{
  current_phase = PHASE_FAILED;
  if (result == CUDA_SUCCESS)
    snprintf(failure, sizeof(failure), "importer restore: %s", operation);
  else
    snprintf(failure, sizeof(failure), "importer restore: %s failed: CUresult=%d", operation, (int)result);
}

static CUresult
unavailable(void)
{
  return CUDA_ERROR_NOT_INITIALIZED;
}

static void
initialize_real_dlsym(void)
{
  static const char* versions[] = {"GLIBC_2.2.5", "GLIBC_2.17", "GLIBC_2.34"};
  size_t index;

  for (index = 0; index < sizeof(versions) / sizeof(versions[0]); index++) {
    real_dlsym_function = (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", versions[index]);
    if (real_dlsym_function != NULL)
      return;
  }
}

static void*
real_dlsym(void* handle, const char* name)
{
  if (pthread_once(&real_dlsym_once, initialize_real_dlsym) != 0 || real_dlsym_function == NULL)
    return NULL;
  return real_dlsym_function(handle, name);
}

static void*
real_symbol(const char* name)
{
  void* symbol = real_dlsym(RTLD_NEXT, name);
  void* handle;

  if (symbol != NULL)
    return symbol;
  handle = (void*)atomic_load(&explicit_libcuda_handle);
  return handle == NULL ? NULL : real_dlsym(handle, name);
}

static bool
is_libcuda(void* handle, void* symbol)
{
  struct link_map* map;
  Dl_info info;
  const char* name;

  if (handle == NULL || handle == RTLD_NEXT || dlinfo(handle, RTLD_DI_LINKMAP, &map) != 0 || map == NULL ||
      dladdr(symbol, &info) == 0)
    return false;
  name = strrchr(map->l_name, '/');
  name = name == NULL ? map->l_name : name + 1;
  if (strncmp(name, "libcuda.so", 10) != 0)
    return false;
  name = strrchr(info.dli_fname, '/');
  name = name == NULL ? info.dli_fname : name + 1;
  return strncmp(name, "libcuda.so", 10) == 0;
}

void*
dlsym(void* handle, const char* name)
{
  void* symbol = real_dlsym(handle, name);
  void* entry;

  if (!enabled || symbol == NULL || !is_libcuda(handle, symbol))
    return symbol;
  atomic_store(&explicit_libcuda_handle, (uintptr_t)handle);
  if (strcmp(name, "cuGetProcAddress") == 0)
    atomic_store(&explicit_cu_get_proc_address, (uintptr_t)symbol);
  if (strcmp(name, "cuGetProcAddress_v2") == 0)
    atomic_store(&explicit_cu_get_proc_address_v2, (uintptr_t)symbol);
  entry = replacement(name, 0);
  return entry == NULL || entry == symbol ? symbol : entry;
}

static int
random_bytes(void* output, size_t size)
{
  unsigned char* current = output;

  while (size != 0) {
    ssize_t count = getrandom(current, size, 0);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      return -1;
    current += count;
    size -= (size_t)count;
  }
  return 0;
}

static int
random_id(char output[SNAPSHOT_VMM_ID_SIZE])
{
  uint8_t value[16];
  size_t index;

  if (random_bytes(value, sizeof(value)) != 0)
    return -1;
  for (index = 0; index < sizeof(value); index++)
    snprintf(output + index * 2, SNAPSHOT_VMM_ID_SIZE - index * 2, "%02x", value[index]);
  return 0;
}

static struct allocation*
find_allocation(const uint8_t id[SNAPSHOT_VMM_ALLOCATION_ID_SIZE])
{
  struct allocation* allocation;

  for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
    if (memcmp(allocation->id, id, SNAPSHOT_VMM_ALLOCATION_ID_SIZE) == 0)
      return allocation;
  }
  return NULL;
}

static struct handle*
find_handle(CUmemGenericAllocationHandle application)
{
  struct handle* handle;

  for (handle = handles; handle != NULL; handle = handle->next) {
    if (handle->live && handle->application == application)
      return handle;
  }
  return NULL;
}

static struct mapping*
find_mapping(CUdeviceptr address, size_t size)
{
  struct mapping* mapping;

  for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
    if (mapping->mapped && mapping->address == address && mapping->size == size)
      return mapping;
  }
  return NULL;
}

static struct mapping*
find_mapping_at(CUdeviceptr address)
{
  struct mapping* mapping;

  for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
    if (mapping->mapped && address >= mapping->address && address < mapping->address + mapping->size)
      return mapping;
  }
  return NULL;
}

static size_t
live_handle_count(const struct allocation* allocation)
{
  const struct handle* handle;
  size_t count = 0;

  for (handle = handles; handle != NULL; handle = handle->next) {
    if (handle->live && handle->allocation == allocation)
      count++;
  }
  return count;
}

static struct mapping*
first_mapping(const struct allocation* allocation)
{
  struct mapping* mapping;

  for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
    if (mapping->allocation == allocation && mapping->mapped)
      return mapping;
  }
  return NULL;
}

static struct handle*
first_live_handle(const struct allocation* allocation)
{
  struct handle* handle;

  for (handle = handles; handle != NULL; handle = handle->next) {
    if (handle->allocation == allocation && handle->live)
      return handle;
  }
  return NULL;
}

static int
add_handle(
    struct allocation* allocation, CUmemGenericAllocationHandle application, CUmemGenericAllocationHandle current)
{
  struct handle* handle = calloc(1, sizeof(*handle));

  if (handle == NULL)
    return -1;
  handle->application = application;
  handle->current = current;
  handle->live = true;
  handle->allocation = allocation;
  handle->next = handles;
  handles = handle;
  return 0;
}

static int
enter_context(CUcontext context, struct context_scope* scope)
{
  context_get_fn get_current = (context_get_fn)real_symbol("cuCtxGetCurrent");
  context_set_fn set_current = (context_set_fn)real_symbol("cuCtxSetCurrent");

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
  context_set_fn set_current = (context_set_fn)real_symbol("cuCtxSetCurrent");

  return !scope->changed || (set_current != NULL && set_current(scope->previous) == CUDA_SUCCESS) ? 0 : -1;
}

static int
current_context(CUcontext* context)
{
  context_get_fn get_current = (context_get_fn)real_symbol("cuCtxGetCurrent");

  return get_current != NULL && get_current(context) == CUDA_SUCCESS && *context != NULL ? 0 : -1;
}

static int
create_posix_capability(const struct allocation* allocation, int* output)
{
  struct snapshot_vmm_posix_capability capability;

  memset(&capability, 0, sizeof(capability));
  capability.magic = SNAPSHOT_VMM_POSIX_CAPABILITY_MAGIC;
  capability.version = SNAPSHOT_VMM_POSIX_CAPABILITY_VERSION;
  snprintf(capability.job_id, sizeof(capability.job_id), "%s", job_id);
  snprintf(
      capability.creator_participant, sizeof(capability.creator_participant), "%s", allocation->creator_participant);
  memcpy(capability.allocation_id, allocation->id, sizeof(capability.allocation_id));
  snprintf(capability.creator_endpoint, sizeof(capability.creator_endpoint), "%s", allocation->creator_endpoint);
  memcpy(capability.authorization, allocation->authorization, sizeof(capability.authorization));
  return snapshot_vmm_posix_create_capability(&capability, output);
}

static CUresult
export_raw(struct allocation* allocation, int* output)
{
  export_fn export_handle = (export_fn)real_symbol("cuMemExportToShareableHandle");
  retain_fn retain = (retain_fn)real_symbol("cuMemRetainAllocationHandle");
  release_fn release = (release_fn)real_symbol("cuMemRelease");
  struct context_scope scope;
  struct handle* handle;
  struct mapping* mapping;
  CUmemGenericAllocationHandle temporary = 0;
  CUresult result;

  *output = -1;
  if (!allocation->creator || export_handle == NULL || enter_context(allocation->context, &scope) != 0)
    return CUDA_ERROR_INVALID_HANDLE;
  handle = first_live_handle(allocation);
  if (allocation->carrier != 0)
    temporary = allocation->carrier;
  else if (handle != NULL)
    temporary = handle->current;
  else if ((mapping = first_mapping(allocation)) != NULL && retain != NULL) {
    result = retain(&temporary, (void*)(uintptr_t)mapping->address);
    if (result != CUDA_SUCCESS)
      goto done;
  } else {
    result = CUDA_ERROR_INVALID_HANDLE;
    goto done;
  }
  result = export_handle(output, temporary, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  if (temporary != allocation->carrier && (handle == NULL || temporary != handle->current) && release != NULL)
    (void)release(temporary);
done:
  if (leave_context(&scope) != 0) {
    if (*output >= 0) {
      close(*output);
      *output = -1;
    }
    return CUDA_ERROR_UNKNOWN;
  }
  return result;
}

static int
request_export(const struct snapshot_vmm_posix_capability* capability, int* output, char* error, size_t error_size)
{
  int result = -1;

  *output = -1;
  if (error != NULL && error_size != 0)
    error[0] = '\0';
  if (strcmp(capability->job_id, job_id) != 0) {
    if (error != NULL && error_size != 0)
      snprintf(error, error_size, "%s", "capability job mismatch");
    return -1;
  }
  if (strcmp(capability->creator_participant, participant_id) == 0) {
    struct allocation* allocation;
    CUresult export_result;

    pthread_mutex_lock(&state_lock);
    allocation = find_allocation(capability->allocation_id);
    if (allocation == NULL ||
        memcmp(allocation->authorization, capability->authorization, sizeof(allocation->authorization)) != 0) {
      if (error != NULL && error_size != 0)
        snprintf(error, error_size, "%s", "creator allocation is unavailable");
    } else if ((export_result = export_raw(allocation, output)) != CUDA_SUCCESS) {
      if (error != NULL && error_size != 0)
        snprintf(error, error_size, "creator export failed: CUresult=%d", (int)export_result);
    } else {
      result = 0;
    }
    pthread_mutex_unlock(&state_lock);
    return result;
  }
  return snapshot_vmm_posix_request_export(capability, output, error, error_size);
}

static void
fill_header(struct snapshot_vmm_header* header, uint16_t operation)
{
  memset(header, 0, sizeof(*header));
  header->magic = SNAPSHOT_VMM_MAGIC;
  header->version = SNAPSHOT_VMM_VERSION;
  header->operation = operation;
  snprintf(header->job_id, sizeof(header->job_id), "%s", job_id);
  snprintf(header->participant_id, sizeof(header->participant_id), "%s", participant_id);
}

static void
response_error(struct snapshot_vmm_header* response, const char* message)
{
  response->status = -1;
  snprintf(response->message, sizeof(response->message), "%s", message);
}

static struct snapshot_vmm_record*
inspect_records(uint32_t* count)
{
  struct snapshot_vmm_record* records;
  struct snapshot_vmm_record* record;
  struct allocation* allocation;
  struct mapping* mapping;
  size_t total = 0;
  size_t index;

  for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
    if (live_handle_count(allocation) != 0 || first_mapping(allocation) != NULL)
      total++;
  }
  for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
    if (mapping->mapped)
      total++;
  }
  if (total > SNAPSHOT_VMM_MAX_RECORDS)
    return NULL;
  records = calloc(total == 0 ? 1 : total, sizeof(*records));
  if (records == NULL)
    return NULL;
  record = records;
  for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
    size_t handles_count = live_handle_count(allocation);
    if (handles_count == 0 && first_mapping(allocation) == NULL)
      continue;
    record->kind = SNAPSHOT_VMM_ALLOCATION;
    record->flags = allocation->creator ? SNAPSHOT_VMM_CREATOR : 0;
    if (handles_count != 0)
      record->flags |= SNAPSHOT_VMM_APPLICATION_HANDLE_LIVE;
    memcpy(record->allocation_id, allocation->id, sizeof(record->allocation_id));
    record->allocation_size = allocation->size;
    record->allocation_type = allocation->properties.type;
    record->requested_handle_types = allocation->properties.requestedHandleTypes;
    record->allocation_location_type = allocation->properties.location.type;
    record->allocation_location_id = allocation->properties.location.id;
    record->application_handle_count = (uint32_t)handles_count;
    record++;
  }
  for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
    if (!mapping->mapped)
      continue;
    record->kind = SNAPSHOT_VMM_MAPPING;
    record->flags = mapping->allocation->creator ? SNAPSHOT_VMM_CREATOR : 0;
    memcpy(record->allocation_id, mapping->allocation->id, sizeof(record->allocation_id));
    record->address = mapping->address;
    record->size = mapping->size;
    record->offset = mapping->offset;
    record->access_count = (uint32_t)mapping->access_count;
    for (index = 0; index < mapping->access_count; index++) {
      record->access[index].location_type = mapping->access[index].location.type;
      record->access[index].location_id = mapping->access[index].location.id;
      record->access[index].flags = mapping->access[index].flags;
    }
    record++;
  }
  *count = (uint32_t)total;
  return records;
}

static bool
handle_current_used(const struct handle* except, CUmemGenericAllocationHandle current)
{
  const struct handle* handle;

  for (handle = handles; handle != NULL; handle = handle->next) {
    if (handle != except && handle->live && handle->current == current)
      return true;
  }
  return false;
}

static int
create_checkpoint_carriers(void)
{
  retain_fn retain = (retain_fn)real_symbol("cuMemRetainAllocationHandle");
  struct allocation* allocation;
  struct context_scope scope;

  if (current_phase != PHASE_ACTIVE || retain == NULL)
    return -1;
  for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
    struct handle* carrier_handle;
    struct mapping* carrier_mapping;
    CUresult result;

    if (!allocation->creator || (live_handle_count(allocation) == 0 && first_mapping(allocation) == NULL))
      continue;
    carrier_handle = first_live_handle(allocation);
    carrier_mapping = first_mapping(allocation);
    if (carrier_handle != NULL) {
      allocation->carrier = carrier_handle->current;
      continue;
    }
    if (carrier_mapping == NULL || enter_context(allocation->context, &scope) != 0)
      goto failed;
    result = retain(&allocation->carrier, (void*)(uintptr_t)carrier_mapping->address);
    if (leave_context(&scope) != 0 || result != CUDA_SUCCESS)
      goto failed;
  }
  current_phase = PHASE_CARRIERS;
  return 0;
failed:
  set_failure("cannot create CUDA VMM checkpoint carrier");
  return -1;
}

static int
prepare_topology(void)
{
  release_fn release = (release_fn)real_symbol("cuMemRelease");
  unmap_fn unmap = (unmap_fn)real_symbol("cuMemUnmap");
  struct allocation* allocation;
  struct context_scope scope;
  struct handle* handle;
  struct mapping* mapping;

  if (current_phase != PHASE_CARRIERS || release == NULL || unmap == NULL)
    return -1;
  for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
    if (allocation->creator && (live_handle_count(allocation) != 0 || first_mapping(allocation) != NULL) &&
        allocation->carrier == 0)
      goto failed;
  }
  current_phase = PHASE_PREPARED;
  for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
    if (live_handle_count(allocation) == 0 && first_mapping(allocation) == NULL && allocation->carrier == 0)
      continue;
    if (enter_context(allocation->context, &scope) != 0)
      goto failed;
    for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
      if (mapping->allocation == allocation && mapping->mapped) {
        if (unmap(mapping->address, mapping->size) != CUDA_SUCCESS) {
          (void)leave_context(&scope);
          goto failed;
        }
        mapping->mapped = false;
        mapping->checkpointed = true;
      }
    }
    for (handle = handles; handle != NULL; handle = handle->next) {
      CUmemGenericAllocationHandle old;

      if (!handle->live || handle->allocation != allocation)
        continue;
      old = handle->current;
      if (allocation->creator && old == allocation->carrier) {
        handle->current = allocation->carrier;
        continue;
      }
      if (!handle_current_used(handle, old) && release(old) != CUDA_SUCCESS) {
        (void)leave_context(&scope);
        goto failed;
      }
      handle->current = allocation->creator ? allocation->carrier : 0;
    }
    if (leave_context(&scope) != 0)
      goto failed;
  }
  return 0;
failed:
  set_failure("cannot prepare CUDA VMM topology");
  return -1;
}

static CUresult
restore_mappings(struct allocation* allocation, CUmemGenericAllocationHandle handle, const char** operation)
{
  map_fn map = (map_fn)real_symbol("cuMemMap");
  access_fn set_access = (access_fn)real_symbol("cuMemSetAccess");
  struct mapping* mapping;
  CUresult result;

  if (map == NULL || set_access == NULL) {
    *operation = "mapping symbols are unavailable";
    return CUDA_ERROR_NOT_INITIALIZED;
  }
  for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
    if (mapping->allocation != allocation || !mapping->checkpointed)
      continue;
    result = map(mapping->address, mapping->size, mapping->offset, handle, 0);
    if (result != CUDA_SUCCESS) {
      *operation = "cuMemMap";
      return result;
    }
    mapping->mapped = true;
    if (mapping->access_count != 0) {
      result = set_access(mapping->address, mapping->size, mapping->access, mapping->access_count);
      if (result != CUDA_SUCCESS) {
        *operation = "cuMemSetAccess";
        return result;
      }
    }
    mapping->checkpointed = false;
  }
  return CUDA_SUCCESS;
}

static int
restore_creators(void)
{
  release_fn release = (release_fn)real_symbol("cuMemRelease");
  struct allocation* allocation;
  struct context_scope scope;
  struct handle* handle;
  const char* mapping_operation;

  if ((current_phase != PHASE_PREPARED && current_phase != PHASE_FAILED) || release == NULL)
    return -1;
  for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
    if (!allocation->creator || allocation->carrier == 0)
      continue;
    if (enter_context(allocation->context, &scope) != 0)
      goto failed;
    if (restore_mappings(allocation, allocation->carrier, &mapping_operation) != CUDA_SUCCESS) {
      (void)leave_context(&scope);
      goto failed;
    }
    for (handle = handles; handle != NULL; handle = handle->next) {
      if (handle->live && handle->allocation == allocation)
        handle->current = allocation->carrier;
    }
    if (live_handle_count(allocation) == 0) {
      if (release(allocation->carrier) != CUDA_SUCCESS) {
        (void)leave_context(&scope);
        goto failed;
      }
      allocation->carrier = 0;
    }
    if (leave_context(&scope) != 0)
      goto failed;
  }
  current_phase = PHASE_CREATORS_RESTORED;
  failure[0] = '\0';
  return 0;
failed:
  set_failure("cannot restore creator CUDA VMM topology");
  return -1;
}

static int
restore_importers(void)
{
  import_fn import_handle = (import_fn)real_symbol("cuMemImportFromShareableHandle");
  release_fn release = (release_fn)real_symbol("cuMemRelease");
  struct allocation* allocation;
  struct context_scope scope;
  struct handle* handle;

  if (current_phase != PHASE_CREATORS_RESTORED || import_handle == NULL || release == NULL) {
    snprintf(failure, sizeof(failure), "%s", "importer restore: phase or symbols are not ready");
    return -1;
  }
  for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
    struct snapshot_vmm_posix_capability capability;
    CUmemGenericAllocationHandle imported = 0;
    CUresult cuda_result;
    char export_error[sizeof(failure)];
    int export_result;
    int raw_fd = -1;
    bool needed = false;
    const char* mapping_operation;

    if (allocation->creator)
      continue;
    for (handle = handles; handle != NULL; handle = handle->next) {
      if (handle->live && handle->allocation == allocation) {
        needed = true;
        break;
      }
    }
    if (!needed) {
      struct mapping* mapping;
      for (mapping = mappings; mapping != NULL; mapping = mapping->next) {
        if (mapping->allocation == allocation && mapping->checkpointed) {
          needed = true;
          break;
        }
      }
    }
    if (!needed)
      continue;
    memset(&capability, 0, sizeof(capability));
    capability.magic = SNAPSHOT_VMM_POSIX_CAPABILITY_MAGIC;
    capability.version = SNAPSHOT_VMM_POSIX_CAPABILITY_VERSION;
    snprintf(capability.job_id, sizeof(capability.job_id), "%s", job_id);
    snprintf(
        capability.creator_participant, sizeof(capability.creator_participant), "%s", allocation->creator_participant);
    memcpy(capability.allocation_id, allocation->id, sizeof(capability.allocation_id));
    snprintf(capability.creator_endpoint, sizeof(capability.creator_endpoint), "%s", allocation->creator_endpoint);
    memcpy(capability.authorization, allocation->authorization, sizeof(capability.authorization));
    if (enter_context(allocation->context, &scope) != 0) {
      set_importer_failure("enter context", CUDA_SUCCESS);
      return -1;
    }
    pthread_mutex_unlock(&state_lock);
    export_result = request_export(&capability, &raw_fd, export_error, sizeof(export_error));
    pthread_mutex_lock(&state_lock);
    if (export_result != 0) {
      (void)leave_context(&scope);
      current_phase = PHASE_FAILED;
      snprintf(
          failure, sizeof(failure), "importer restore: creator export: %.61s",
          export_error[0] != '\0' ? export_error : "request failed");
      return -1;
    }
    cuda_result = import_handle(&imported, (void*)(uintptr_t)raw_fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    if (cuda_result != CUDA_SUCCESS) {
      close(raw_fd);
      (void)leave_context(&scope);
      set_importer_failure("cuMemImportFromShareableHandle", cuda_result);
      return -1;
    }
    if (close(raw_fd) != 0) {
      (void)release(imported);
      (void)leave_context(&scope);
      set_importer_failure("raw FD close", CUDA_SUCCESS);
      return -1;
    }
    cuda_result = restore_mappings(allocation, imported, &mapping_operation);
    if (cuda_result != CUDA_SUCCESS) {
      (void)release(imported);
      (void)leave_context(&scope);
      set_importer_failure(mapping_operation, cuda_result);
      return -1;
    }
    for (handle = handles; handle != NULL; handle = handle->next) {
      if (handle->live && handle->allocation == allocation)
        handle->current = imported;
    }
    if (live_handle_count(allocation) == 0) {
      cuda_result = release(imported);
      if (cuda_result != CUDA_SUCCESS) {
        (void)leave_context(&scope);
        set_importer_failure("cuMemRelease imported handle", cuda_result);
        return -1;
      }
    }
    if (leave_context(&scope) != 0) {
      set_importer_failure("leave context", CUDA_SUCCESS);
      return -1;
    }
  }
  current_phase = PHASE_ACTIVE;
  failure[0] = '\0';
  return 0;
}

static bool
valid_control_request(const struct snapshot_vmm_header* request)
{
  return request->magic == SNAPSHOT_VMM_MAGIC && request->version == SNAPSHOT_VMM_VERSION && request->status == 0 &&
         request->count == 0 && request->payload_size == 0 &&
         ((request->operation == SNAPSHOT_VMM_IDENTIFY && request->job_id[0] == '\0') ||
          strcmp(request->job_id, job_id) == 0);
}

static void
serve(int client)
{
  struct snapshot_vmm_header request;
  struct snapshot_vmm_header response;
  struct snapshot_vmm_record* records = NULL;
  int passed_fd = -1;
  int exported_fd = -1;

  if (snapshot_vmm_receive_header(client, &request, &passed_fd) != 0)
    goto done;
  fill_header(&response, request.operation);
  if (passed_fd >= 0 || !valid_control_request(&request)) {
    response_error(&response, "invalid CUDA VMM control request");
    (void)snapshot_vmm_send_header(client, &response, -1);
    goto done;
  }
  pthread_mutex_lock(&state_lock);
  switch (request.operation) {
    case SNAPSHOT_VMM_IDENTIFY:
      if (current_phase == PHASE_FAILED)
        response_error(&response, failure);
      (void)snapshot_vmm_send_header(client, &response, -1);
      break;
    case SNAPSHOT_VMM_INSPECT:
      if (current_phase != PHASE_ACTIVE) {
        response_error(&response, "CUDA VMM topology is not active");
        (void)snapshot_vmm_send_header(client, &response, -1);
        break;
      }
      records = inspect_records(&response.count);
      if (records == NULL) {
        response_error(&response, "cannot inspect CUDA VMM topology");
        (void)snapshot_vmm_send_header(client, &response, -1);
        break;
      }
      response.payload_size = (uint64_t)response.count * sizeof(struct snapshot_vmm_record);
      if (snapshot_vmm_send_header(client, &response, -1) == 0 && response.payload_size != 0)
        (void)snapshot_vmm_write_all(client, records, (size_t)response.payload_size);
      break;
    case SNAPSHOT_VMM_PREPARE:
      if (prepare_topology() != 0)
        response_error(&response, failure);
      (void)snapshot_vmm_send_header(client, &response, -1);
      break;
    case SNAPSHOT_VMM_CREATE_CARRIERS:
      if (create_checkpoint_carriers() != 0)
        response_error(&response, failure);
      (void)snapshot_vmm_send_header(client, &response, -1);
      break;
    case SNAPSHOT_VMM_RESTORE_CREATORS:
      if (restore_creators() != 0)
        response_error(&response, failure);
      (void)snapshot_vmm_send_header(client, &response, -1);
      break;
    case SNAPSHOT_VMM_RESTORE_IMPORTERS:
      if (restore_importers() != 0)
        response_error(&response, failure);
      (void)snapshot_vmm_send_header(client, &response, -1);
      break;
    case SNAPSHOT_VMM_EXPORT: {
      struct allocation* allocation = find_allocation(request.allocation_id);
      CUresult export_result;
      if (strcmp(request.participant_id, participant_id) != 0 || allocation == NULL || !allocation->creator ||
          memcmp(allocation->authorization, request.authorization, sizeof(request.authorization)) != 0 ||
          (current_phase != PHASE_ACTIVE && current_phase != PHASE_CREATORS_RESTORED)) {
        response_error(&response, "creator allocation is unavailable");
        (void)snapshot_vmm_send_header(client, &response, -1);
        break;
      }
      export_result = export_raw(allocation, &exported_fd);
      if (export_result != CUDA_SUCCESS) {
        char message[sizeof(response.message)];
        snprintf(message, sizeof(message), "creator export failed: CUresult=%d", (int)export_result);
        response_error(&response, message);
        (void)snapshot_vmm_send_header(client, &response, -1);
        break;
      }
      memcpy(response.allocation_id, allocation->id, sizeof(response.allocation_id));
      (void)snapshot_vmm_send_header(client, &response, exported_fd);
      break;
    }
    default:
      response_error(&response, "unknown CUDA VMM control operation");
      (void)snapshot_vmm_send_header(client, &response, -1);
      break;
  }
  pthread_mutex_unlock(&state_lock);
done:
  if (passed_fd >= 0)
    close(passed_fd);
  if (exported_fd >= 0)
    close(exported_fd);
  free(records);
}

static void*
control_agent(void* unused)
{
  (void)unused;
  for (;;) {
    int client = accept4(listener, NULL, NULL, SOCK_CLOEXEC);
    if (client < 0) {
      if (errno == EINTR)
        continue;
      return NULL;
    }
    if (snapshot_vmm_set_socket_timeouts(client, CONTROL_TIMEOUT_SECONDS) == 0)
      serve(client);
    close(client);
  }
}

static int
format_socket_path(char* output, size_t size)
{
  int count = snprintf(output, size, "%s/%s%ld.sock", control_directory, SNAPSHOT_VMM_SOCKET_PREFIX, (long)getpid());
  return count >= 0 && (size_t)count < size ? 0 : -1;
}

static int
start_control_endpoint(void)
{
  struct sockaddr_un address = {.sun_family = AF_UNIX};
  pthread_t thread;

  if (format_socket_path(socket_path, sizeof(socket_path)) != 0) {
    socket_path[0] = '\0';
    return -1;
  }
  snprintf(address.sun_path, sizeof(address.sun_path), "%s", socket_path);
  listener = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (listener >= 0)
    unlink(socket_path);
  if (listener < 0 || bind(listener, (const struct sockaddr*)&address, sizeof(address)) != 0 ||
      listen(listener, 8) != 0 || pthread_create(&thread, NULL, control_agent, NULL) != 0) {
    if (listener >= 0)
      close(listener);
    listener = -1;
    unlink(socket_path);
    socket_path[0] = '\0';
    return -1;
  }
  pthread_detach(thread);
  return 0;
}

static void
fork_prepare(void)
{
  pthread_mutex_lock(&state_lock);
}

static void
fork_parent(void)
{
  pthread_mutex_unlock(&state_lock);
}

static void
fork_child(void)
{
  if (listener >= 0)
    close(listener);
  listener = -1;
  participant_id[0] = '\0';
  socket_path[0] = '\0';
  endpoint_needs_initialization = true;
  allocations = NULL;
  handles = NULL;
  mappings = NULL;
  current_phase = PHASE_ACTIVE;
  failure[0] = '\0';
  pthread_mutex_unlock(&state_lock);
}

static CUresult
ensure_process_endpoint(void)
{
  CUresult result = CUDA_SUCCESS;

  pthread_mutex_lock(&state_lock);
  if (endpoint_needs_initialization) {
    if (current_phase == PHASE_FAILED) {
      result = CUDA_ERROR_NOT_INITIALIZED;
    } else if (random_id(participant_id) != 0 || start_control_endpoint() != 0) {
      set_failure("cannot start forked CUDA VMM control endpoint");
      result = CUDA_ERROR_NOT_INITIALIZED;
    } else {
      endpoint_needs_initialization = false;
    }
  }
  pthread_mutex_unlock(&state_lock);
  return result;
}

__attribute__((constructor)) static void
initialize(void)
{
  const char* control;
  const char* configured_participant;
  const char* configured_job;

  enabled = getenv("DYN_SNAPSHOT_CUDA_VMM_INTERPOSE") != NULL;
  if (!enabled)
    return;
  configured_job = getenv("DYN_SNAPSHOT_JOB_ID");
  configured_participant = getenv("DYN_SNAPSHOT_PARTICIPANT_ID");
  if (!snapshot_vmm_is_lower_hex_id(configured_job) ||
      (configured_participant != NULL && !snapshot_vmm_is_lower_hex_id(configured_participant))) {
    set_failure("invalid CUDA VMM job or participant identity");
    return;
  }
  snprintf(job_id, sizeof(job_id), "%s", configured_job);
  if (configured_participant != NULL)
    snprintf(participant_id, sizeof(participant_id), "%s", configured_participant);
  else if (random_id(participant_id) != 0) {
    set_failure("cannot create CUDA VMM participant identity");
    return;
  }
  control = getenv("DYN_SNAPSHOT_CONTROL_DIR");
  if (control == NULL || control[0] == '\0')
    control = CONTROL_DIR;
  if (control[0] != '/' || strlen(control) >= sizeof(control_directory) ||
      snprintf(control_directory, sizeof(control_directory), "%s", control) >= (int)sizeof(control_directory)) {
    set_failure("invalid CUDA VMM control directory");
    return;
  }
  if (pthread_atfork(fork_prepare, fork_parent, fork_child) != 0) {
    set_failure("cannot register CUDA VMM fork handlers");
    return;
  }
  if (start_control_endpoint() != 0) {
    set_failure("cannot start CUDA VMM control endpoint");
    return;
  }
}

__attribute__((destructor)) static void
finalize(void)
{
  if (listener >= 0)
    close(listener);
  if (socket_path[0] != '\0')
    unlink(socket_path);
}

CUresult CUDAAPI
cuMemCreate(
    CUmemGenericAllocationHandle* output, size_t size, const CUmemAllocationProp* properties, unsigned long long flags)
{
  create_fn function = (create_fn)real_symbol("cuMemCreate");
  struct allocation* allocation;
  CUresult result;

  if (!enabled || properties == NULL || properties->requestedHandleTypes == 0)
    return function != NULL ? function(output, size, properties, flags) : unavailable();
  if ((result = ensure_process_endpoint()) != CUDA_SUCCESS)
    return result;
  if (properties->requestedHandleTypes != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    return CUDA_ERROR_NOT_SUPPORTED;
  if (function == NULL)
    return unavailable();
  result = function(output, size, properties, flags);
  if (result != CUDA_SUCCESS)
    return result;
  allocation = calloc(1, sizeof(*allocation));
  pthread_mutex_lock(&state_lock);
  if (allocation == NULL || current_phase != PHASE_ACTIVE || current_context(&allocation->context) != 0 ||
      random_bytes(allocation->id, sizeof(allocation->id)) != 0 ||
      random_bytes(allocation->authorization, sizeof(allocation->authorization)) != 0 ||
      add_handle(allocation, *output, *output) != 0) {
    release_fn release = (release_fn)real_symbol("cuMemRelease");
    if (release != NULL)
      (void)release(*output);
    free(allocation);
    pthread_mutex_unlock(&state_lock);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  allocation->size = size;
  allocation->properties = *properties;
  allocation->creator = true;
  snprintf(allocation->creator_participant, sizeof(allocation->creator_participant), "%s", participant_id);
  snprintf(allocation->creator_endpoint, sizeof(allocation->creator_endpoint), "%s", socket_path);
  allocation->next = allocations;
  allocations = allocation;
  pthread_mutex_unlock(&state_lock);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuMemRelease(CUmemGenericAllocationHandle application)
{
  release_fn function = (release_fn)real_symbol("cuMemRelease");
  struct handle* handle;
  CUresult result;

  if (!enabled)
    return function != NULL ? function(application) : unavailable();
  if ((result = ensure_process_endpoint()) != CUDA_SUCCESS)
    return result;
  pthread_mutex_lock(&state_lock);
  handle = find_handle(application);
  if (handle == NULL) {
    pthread_mutex_unlock(&state_lock);
    return function != NULL ? function(application) : unavailable();
  }
  if (current_phase != PHASE_ACTIVE || handle->current == 0) {
    pthread_mutex_unlock(&state_lock);
    return CUDA_ERROR_NOT_READY;
  }
  result = CUDA_SUCCESS;
  if (!handle_current_used(handle, handle->current))
    result = function != NULL ? function(handle->current) : unavailable();
  if (result == CUDA_SUCCESS)
    handle->live = false;
  pthread_mutex_unlock(&state_lock);
  return result;
}

CUresult CUDAAPI
cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* output, void* address)
{
  retain_fn function = (retain_fn)real_symbol("cuMemRetainAllocationHandle");
  struct mapping* mapping;
  CUresult result;

  if (enabled && (result = ensure_process_endpoint()) != CUDA_SUCCESS)
    return result;
  if (function == NULL)
    return unavailable();
  result = function(output, address);
  if (!enabled || result != CUDA_SUCCESS)
    return result;
  pthread_mutex_lock(&state_lock);
  mapping = find_mapping_at((CUdeviceptr)(uintptr_t)address);
  if (mapping != NULL && add_handle(mapping->allocation, *output, *output) != 0) {
    release_fn release = (release_fn)real_symbol("cuMemRelease");
    if (release != NULL)
      (void)release(*output);
    result = CUDA_ERROR_OUT_OF_MEMORY;
  }
  pthread_mutex_unlock(&state_lock);
  return result;
}

CUresult CUDAAPI
cuMemMap(
    CUdeviceptr address, size_t size, size_t offset, CUmemGenericAllocationHandle application, unsigned long long flags)
{
  map_fn function = (map_fn)real_symbol("cuMemMap");
  struct mapping* mapping;
  struct handle* handle;
  CUresult result;

  if (!enabled)
    return function != NULL ? function(address, size, offset, application, flags) : unavailable();
  if ((result = ensure_process_endpoint()) != CUDA_SUCCESS)
    return result;
  pthread_mutex_lock(&state_lock);
  handle = find_handle(application);
  if (handle == NULL) {
    pthread_mutex_unlock(&state_lock);
    return function != NULL ? function(address, size, offset, application, flags) : unavailable();
  }
  if (current_phase != PHASE_ACTIVE || handle->current == 0) {
    pthread_mutex_unlock(&state_lock);
    return CUDA_ERROR_NOT_READY;
  }
  result = function != NULL ? function(address, size, offset, handle->current, flags) : unavailable();
  if (result == CUDA_SUCCESS) {
    mapping = calloc(1, sizeof(*mapping));
    if (mapping == NULL) {
      unmap_fn unmap = (unmap_fn)real_symbol("cuMemUnmap");
      if (unmap != NULL)
        (void)unmap(address, size);
      result = CUDA_ERROR_OUT_OF_MEMORY;
    } else {
      mapping->address = address;
      mapping->size = size;
      mapping->offset = offset;
      mapping->mapped = true;
      mapping->allocation = handle->allocation;
      mapping->next = mappings;
      mappings = mapping;
    }
  }
  pthread_mutex_unlock(&state_lock);
  return result;
}

CUresult CUDAAPI
cuMemUnmap(CUdeviceptr address, size_t size)
{
  unmap_fn function = (unmap_fn)real_symbol("cuMemUnmap");
  struct mapping* mapping;
  CUresult result;

  if (!enabled)
    return function != NULL ? function(address, size) : unavailable();
  if ((result = ensure_process_endpoint()) != CUDA_SUCCESS)
    return result;
  pthread_mutex_lock(&state_lock);
  mapping = find_mapping(address, size);
  if (mapping == NULL) {
    pthread_mutex_unlock(&state_lock);
    return function != NULL ? function(address, size) : unavailable();
  }
  if (current_phase != PHASE_ACTIVE) {
    pthread_mutex_unlock(&state_lock);
    return CUDA_ERROR_NOT_READY;
  }
  result = function != NULL ? function(address, size) : unavailable();
  if (result == CUDA_SUCCESS)
    mapping->mapped = false;
  pthread_mutex_unlock(&state_lock);
  return result;
}

CUresult CUDAAPI
cuMemSetAccess(CUdeviceptr address, size_t size, const CUmemAccessDesc* descriptors, size_t count)
{
  access_fn function = (access_fn)real_symbol("cuMemSetAccess");
  struct mapping* mapping;
  CUresult result;

  if (!enabled)
    return function != NULL ? function(address, size, descriptors, count) : unavailable();
  if ((result = ensure_process_endpoint()) != CUDA_SUCCESS)
    return result;
  pthread_mutex_lock(&state_lock);
  mapping = find_mapping(address, size);
  if (mapping == NULL) {
    pthread_mutex_unlock(&state_lock);
    return function != NULL ? function(address, size, descriptors, count) : unavailable();
  }
  if (current_phase != PHASE_ACTIVE || count > SNAPSHOT_VMM_MAX_ACCESS) {
    pthread_mutex_unlock(&state_lock);
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  result = function != NULL ? function(address, size, descriptors, count) : unavailable();
  if (result == CUDA_SUCCESS) {
    memcpy(mapping->access, descriptors, count * sizeof(*descriptors));
    mapping->access_count = count;
  }
  pthread_mutex_unlock(&state_lock);
  return result;
}

CUresult CUDAAPI
cuMemExportToShareableHandle(
    void* shareable, CUmemGenericAllocationHandle application, CUmemAllocationHandleType type, unsigned long long flags)
{
  export_fn function = (export_fn)real_symbol("cuMemExportToShareableHandle");
  struct handle* handle;
  CUresult result;
  int capability = -1;

  if (!enabled)
    return function != NULL ? function(shareable, application, type, flags) : unavailable();
  if ((result = ensure_process_endpoint()) != CUDA_SUCCESS)
    return result;
  if (type != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    return CUDA_ERROR_NOT_SUPPORTED;
  pthread_mutex_lock(&state_lock);
  handle = find_handle(application);
  if (handle == NULL) {
    pthread_mutex_unlock(&state_lock);
    return function != NULL ? function(shareable, application, type, flags) : unavailable();
  }
  if (!handle->allocation->creator || current_phase != PHASE_ACTIVE ||
      create_posix_capability(handle->allocation, &capability) != 0) {
    pthread_mutex_unlock(&state_lock);
    return CUDA_ERROR_INVALID_HANDLE;
  }
  *(int*)shareable = capability;
  pthread_mutex_unlock(&state_lock);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* output, void* os_handle, CUmemAllocationHandleType type)
{
  import_fn function = (import_fn)real_symbol("cuMemImportFromShareableHandle");
  properties_fn get_properties;
  struct snapshot_vmm_posix_capability capability;
  struct allocation* allocation;
  CUmemGenericAllocationHandle imported;
  int raw_fd = -1;
  int capability_fd = (int)(uintptr_t)os_handle;
  CUresult result;

  if (!enabled)
    return function != NULL ? function(output, os_handle, type) : unavailable();
  if (type != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    return CUDA_ERROR_INVALID_HANDLE;
  if (snapshot_vmm_posix_read_capability(capability_fd, &capability) != 0)
    return function != NULL ? function(output, os_handle, type) : unavailable();
  get_properties = (properties_fn)real_symbol("cuMemGetAllocationPropertiesFromHandle");
  if (function == NULL || get_properties == NULL || strcmp(capability.job_id, job_id) != 0)
    return CUDA_ERROR_INVALID_HANDLE;
  if ((result = ensure_process_endpoint()) != CUDA_SUCCESS)
    return result;
  pthread_mutex_lock(&state_lock);
  if (current_phase != PHASE_ACTIVE) {
    pthread_mutex_unlock(&state_lock);
    return CUDA_ERROR_INVALID_HANDLE;
  }
  pthread_mutex_unlock(&state_lock);
  if (request_export(&capability, &raw_fd, NULL, 0) != 0)
    return CUDA_ERROR_INVALID_HANDLE;
  result = function(&imported, (void*)(uintptr_t)raw_fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  if (close(raw_fd) != 0 && result == CUDA_SUCCESS)
    result = CUDA_ERROR_UNKNOWN;
  if (result != CUDA_SUCCESS)
    return result;
  pthread_mutex_lock(&state_lock);
  if (current_phase != PHASE_ACTIVE) {
    release_fn release = (release_fn)real_symbol("cuMemRelease");
    if (release != NULL)
      (void)release(imported);
    pthread_mutex_unlock(&state_lock);
    return CUDA_ERROR_INVALID_HANDLE;
  }
  allocation = find_allocation(capability.allocation_id);
  if (allocation == NULL) {
    allocation = calloc(1, sizeof(*allocation));
    if (allocation != NULL) {
      memcpy(allocation->id, capability.allocation_id, sizeof(allocation->id));
      memcpy(allocation->authorization, capability.authorization, sizeof(allocation->authorization));
      snprintf(
          allocation->creator_participant, sizeof(allocation->creator_participant), "%s",
          capability.creator_participant);
      snprintf(allocation->creator_endpoint, sizeof(allocation->creator_endpoint), "%s", capability.creator_endpoint);
      allocation->creator = false;
      allocation->next = allocations;
      allocations = allocation;
    }
  }
  if (allocation == NULL || current_context(&allocation->context) != 0 ||
      get_properties(&allocation->properties, imported) != CUDA_SUCCESS ||
      add_handle(allocation, imported, imported) != 0) {
    release_fn release = (release_fn)real_symbol("cuMemRelease");
    if (release != NULL)
      (void)release(imported);
    pthread_mutex_unlock(&state_lock);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  *output = imported;
  pthread_mutex_unlock(&state_lock);
  return CUDA_SUCCESS;
}

CUresult CUDAAPI
cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp* properties, CUmemGenericAllocationHandle application)
{
  properties_fn function = (properties_fn)real_symbol("cuMemGetAllocationPropertiesFromHandle");
  struct handle* handle;
  CUresult result;

  if (!enabled)
    return function != NULL ? function(properties, application) : unavailable();
  if ((result = ensure_process_endpoint()) != CUDA_SUCCESS)
    return result;
  pthread_mutex_lock(&state_lock);
  handle = find_handle(application);
  result = handle == NULL
               ? (function != NULL ? function(properties, application) : unavailable())
               : (current_phase == PHASE_ACTIVE && handle->current != 0 ? function(properties, handle->current)
                                                                        : CUDA_ERROR_NOT_READY);
  pthread_mutex_unlock(&state_lock);
  return result;
}

CUresult CUDAAPI
cuMulticastCreate(CUmemGenericAllocationHandle* output, const CUmulticastObjectProp* properties)
{
  typedef CUresult(CUDAAPI * function_type)(CUmemGenericAllocationHandle*, const CUmulticastObjectProp*);
  function_type function = (function_type)real_symbol("cuMulticastCreate");

  return enabled ? CUDA_ERROR_NOT_SUPPORTED
                 : (function != NULL ? function(output, properties) : unavailable());
}

CUresult CUDAAPI
cuMulticastAddDevice(CUmemGenericAllocationHandle multicast, CUdevice device)
{
  typedef CUresult(CUDAAPI * function_type)(CUmemGenericAllocationHandle, CUdevice);
  function_type function = (function_type)real_symbol("cuMulticastAddDevice");

  return enabled ? CUDA_ERROR_NOT_SUPPORTED : (function != NULL ? function(multicast, device) : unavailable());
}

CUresult CUDAAPI
cuMulticastBindMem(
    CUmemGenericAllocationHandle multicast, size_t multicast_offset, CUmemGenericAllocationHandle memory,
    size_t memory_offset, size_t size, unsigned long long flags)
{
  typedef CUresult(CUDAAPI * function_type)(
      CUmemGenericAllocationHandle, size_t, CUmemGenericAllocationHandle, size_t, size_t, unsigned long long);
  function_type function = (function_type)real_symbol("cuMulticastBindMem");

  return enabled ? CUDA_ERROR_NOT_SUPPORTED
                 : (function != NULL
                        ? function(multicast, multicast_offset, memory, memory_offset, size, flags)
                        : unavailable());
}

CUresult CUDAAPI
cuMulticastBindMem_v2(
    CUmemGenericAllocationHandle multicast, CUdevice device, size_t multicast_offset,
    CUmemGenericAllocationHandle memory, size_t memory_offset, size_t size, unsigned long long flags)
{
  typedef CUresult(CUDAAPI * function_type)(
      CUmemGenericAllocationHandle, CUdevice, size_t, CUmemGenericAllocationHandle, size_t, size_t,
      unsigned long long);
  function_type function = (function_type)real_symbol("cuMulticastBindMem_v2");

  return enabled ? CUDA_ERROR_NOT_SUPPORTED
                 : (function != NULL
                        ? function(multicast, device, multicast_offset, memory, memory_offset, size, flags)
                        : unavailable());
}

CUresult CUDAAPI
cuMulticastBindAddr(
    CUmemGenericAllocationHandle multicast, size_t multicast_offset, CUdeviceptr memory, size_t size,
    unsigned long long flags)
{
  typedef CUresult(CUDAAPI * function_type)(
      CUmemGenericAllocationHandle, size_t, CUdeviceptr, size_t, unsigned long long);
  function_type function = (function_type)real_symbol("cuMulticastBindAddr");

  return enabled ? CUDA_ERROR_NOT_SUPPORTED
                 : (function != NULL ? function(multicast, multicast_offset, memory, size, flags) : unavailable());
}

CUresult CUDAAPI
cuMulticastBindAddr_v2(
    CUmemGenericAllocationHandle multicast, CUdevice device, size_t multicast_offset, CUdeviceptr memory,
    size_t size, unsigned long long flags)
{
  typedef CUresult(CUDAAPI * function_type)(
      CUmemGenericAllocationHandle, CUdevice, size_t, CUdeviceptr, size_t, unsigned long long);
  function_type function = (function_type)real_symbol("cuMulticastBindAddr_v2");

  return enabled
             ? CUDA_ERROR_NOT_SUPPORTED
             : (function != NULL ? function(multicast, device, multicast_offset, memory, size, flags) : unavailable());
}

CUresult CUDAAPI
cuMulticastGetGranularity(
    size_t* granularity, const CUmulticastObjectProp* properties, CUmulticastGranularity_flags option)
{
  typedef CUresult(CUDAAPI * function_type)(
      size_t*, const CUmulticastObjectProp*, CUmulticastGranularity_flags);
  function_type function = (function_type)real_symbol("cuMulticastGetGranularity");

  return enabled ? CUDA_ERROR_NOT_SUPPORTED
                 : (function != NULL ? function(granularity, properties, option) : unavailable());
}

CUresult CUDAAPI
cuMulticastUnbind(CUmemGenericAllocationHandle multicast, CUdevice device, size_t offset, size_t size)
{
  typedef CUresult(CUDAAPI * function_type)(CUmemGenericAllocationHandle, CUdevice, size_t, size_t);
  function_type function = (function_type)real_symbol("cuMulticastUnbind");

  return enabled ? CUDA_ERROR_NOT_SUPPORTED
                 : (function != NULL ? function(multicast, device, offset, size) : unavailable());
}

static void*
replacement(const char* symbol, int version)
{
#define ENTRY(name)               \
  if (strcmp(symbol, #name) == 0) \
  return (void*)&name
  if (symbol == NULL)
    return NULL;
  ENTRY(cuMemCreate);
  ENTRY(cuMemRelease);
  ENTRY(cuMemRetainAllocationHandle);
  ENTRY(cuMemMap);
  ENTRY(cuMemUnmap);
  ENTRY(cuMemSetAccess);
  ENTRY(cuMemExportToShareableHandle);
  ENTRY(cuMemImportFromShareableHandle);
  ENTRY(cuMemGetAllocationPropertiesFromHandle);
  ENTRY(cuMulticastCreate);
  ENTRY(cuMulticastAddDevice);
  ENTRY(cuMulticastBindMem);
  ENTRY(cuMulticastBindMem_v2);
  ENTRY(cuMulticastBindAddr);
  ENTRY(cuMulticastBindAddr_v2);
  ENTRY(cuMulticastGetGranularity);
  ENTRY(cuMulticastUnbind);
  ENTRY(cuGetProcAddress_v2);
  ENTRY(cuGetProcAddress_v2_ptsz);
#undef ENTRY
  if (strcmp(symbol, "cuGetProcAddress") == 0)
    return version >= 12000 ? (void*)&cuGetProcAddress_v2 : (void*)&cuGetProcAddress;
  return NULL;
}

CUresult CUDAAPI
cuGetProcAddress(const char* symbol, void** output, int version, cuuint64_t flags)
{
  typedef CUresult(CUDAAPI * function_type)(const char*, void**, int, cuuint64_t);
  function_type function = (function_type)atomic_load(&explicit_cu_get_proc_address);
  CUresult result;
  void* entry;

  if (function == NULL)
    function = (function_type)real_symbol("cuGetProcAddress");
  result = function != NULL ? function(symbol, output, version, flags) : unavailable();
  if (enabled && result == CUDA_SUCCESS && output != NULL && *output != NULL &&
      (entry = replacement(symbol, version)) != NULL)
    *output = entry;
  return result;
}

CUresult CUDAAPI
cuGetProcAddress_v2(
    const char* symbol, void** output, int version, cuuint64_t flags, CUdriverProcAddressQueryResult* status)
{
  typedef CUresult(CUDAAPI * function_type)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
  function_type function = (function_type)atomic_load(&explicit_cu_get_proc_address_v2);
  CUresult result;
  void* entry;

  if (function == NULL)
    function = (function_type)real_symbol("cuGetProcAddress_v2");
  result = function != NULL ? function(symbol, output, version, flags, status) : unavailable();
  if (enabled && result == CUDA_SUCCESS && output != NULL && *output != NULL &&
      (status == NULL || *status == CU_GET_PROC_ADDRESS_SUCCESS) && (entry = replacement(symbol, version)) != NULL)
    *output = entry;
  return result;
}

CUresult CUDAAPI
cuGetProcAddress_v2_ptsz(
    const char* symbol, void** output, int version, cuuint64_t flags, CUdriverProcAddressQueryResult* status)
{
  typedef CUresult(CUDAAPI * function_type)(const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
  function_type function = (function_type)atomic_load(&explicit_cu_get_proc_address_v2);
  cuuint64_t stream_flags = CU_GET_PROC_ADDRESS_LEGACY_STREAM | CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM;
  CUresult result;
  void* entry;

  if (function == NULL)
    function = (function_type)real_symbol("cuGetProcAddress_v2");
  if ((flags & stream_flags) == 0)
    flags |= CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM;
  result = function != NULL ? function(symbol, output, version, flags, status) : unavailable();
  if (enabled && result == CUDA_SUCCESS && output != NULL && *output != NULL &&
      (status == NULL || *status == CU_GET_PROC_ADDRESS_SUCCESS) && (entry = replacement(symbol, version)) != NULL)
    *output = entry;
  return result;
}
