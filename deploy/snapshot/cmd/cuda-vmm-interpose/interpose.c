/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <link.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include "protocol.h"

#undef cuGetProcAddress
#undef cuMulticastAddDevice
#undef cuMulticastBindAddr
#undef cuMulticastBindMem
#undef cuMulticastCreate
#undef cuMulticastUnbind
#undef cudaGetDriverEntryPoint
#undef cudaGetDriverEntryPointByVersion

#define CONTROL_DIR "/snapshot-control"
#define COPY_CHUNK (1U << 20)
#define BROKER_TIMEOUT_SECONDS 30
#define LOGICAL_HANDLE_TAG UINT64_C(0xd94d000000000000)
#define LOGICAL_HANDLE_TAG_MASK UINT64_C(0xffff000000000000)
#define LOGICAL_HANDLE_VALUE_MASK UINT64_C(0x0000ffffffffffff)

_Static_assert(CU_IPC_HANDLE_SIZE == DYN_VMM_FABRIC_HANDLE_SIZE, "CUDA IPC handle ABI changed");
_Static_assert(sizeof(CUmemFabricHandle) == DYN_VMM_FABRIC_HANDLE_SIZE, "CUDA FABRIC handle ABI changed");

CUresult CUDAAPI cuGetProcAddress(const char*, void**, int, cuuint64_t);
CUresult CUDAAPI cuGetProcAddress_v2(
    const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
CUresult CUDAAPI cuGetProcAddress_v2_ptsz(
    const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
CUresult CUDAAPI cuMulticastBindMem(
    CUmemGenericAllocationHandle, size_t, CUmemGenericAllocationHandle, size_t,
    size_t, unsigned long long);
CUresult CUDAAPI cuMulticastCreate(
    CUmemGenericAllocationHandle*, const CUmulticastObjectProp*);
CUresult CUDAAPI cuMulticastAddDevice(
    CUmemGenericAllocationHandle, CUdevice);
CUresult CUDAAPI cuMulticastBindAddr(
    CUmemGenericAllocationHandle, size_t, CUdeviceptr, size_t,
    unsigned long long);
CUresult CUDAAPI cuMulticastUnbind(
    CUmemGenericAllocationHandle, CUdevice, size_t, size_t);
#if CUDA_VERSION >= 13010
CUresult CUDAAPI cuMulticastBindMem_v2(
    CUmemGenericAllocationHandle, CUdevice, size_t,
    CUmemGenericAllocationHandle, size_t, size_t, unsigned long long);
CUresult CUDAAPI cuMulticastBindAddr_v2(
    CUmemGenericAllocationHandle, CUdevice, size_t, CUdeviceptr, size_t,
    unsigned long long);
#endif
CUresult CUDAAPI cuMemRetainAllocationHandle(
    CUmemGenericAllocationHandle*, void*);
CUresult CUDAAPI cuMemGetAllocationPropertiesFromHandle(
    CUmemAllocationProp*, CUmemGenericAllocationHandle);
cudaError_t CUDARTAPI cudaGetDriverEntryPoint_ptsz(
    const char*, void**, unsigned long long,
    enum cudaDriverEntryPointQueryResult*);
cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion_ptsz(
    const char*, void**, unsigned int, unsigned long long,
    enum cudaDriverEntryPointQueryResult*);

struct allocation {
  uint8_t allocation_uuid[DYN_VMM_ALLOCATION_UUID_SIZE];
  uint64_t object_id;
  CUmemGenericAllocationHandle logical_handle;
  CUmemGenericAllocationHandle real_handle;
  CUcontext context;
  CUdeviceptr address;
  size_t size;
  size_t offset;
  CUmemAllocationProp properties;
  CUmulticastObjectProp multicast_properties;
  CUmemAccessDesc* access;
  size_t access_count;
  uint32_t role;
  uint32_t object_kind;
  CUmemAllocationHandleType requested_handle_type;
  int source_device_ordinal;
  int device_ordinal;
  CUuuid gpu_uuid;
  bool exported;
  bool application_handle_live;
  bool mapped;
  bool detached;
  bool member_added;
  bool bound;
  bool temporary_restore_handle;
  CUdevice member_device;
  uint8_t backing_allocation_uuid[DYN_VMM_ALLOCATION_UUID_SIZE];
  uint64_t backing_object_id;
  uint32_t backing_role;
  uint32_t bind_api;
  size_t multicast_offset;
  size_t memory_offset;
  size_t bind_size;
  unsigned long long bind_flags;
  struct allocation* next;
};

struct access_update {
  struct allocation* allocation;
  CUmemAccessDesc* access;
};

struct broker_result {
  CUmemAllocationHandleType handle_type;
  uint32_t object_kind;
  int fd;
  uint8_t bytes[DYN_VMM_FABRIC_HANDLE_SIZE];
  size_t bytes_size;
  CUmulticastObjectProp multicast_properties;
  bool has_multicast_properties;
};

struct unmanaged_multicast {
  CUmemGenericAllocationHandle handle;
  struct unmanaged_multicast* next;
};

struct context_scope {
  CUcontext previous;
  bool changed;
};

typedef CUresult(CUDAAPI* create_fn)(
    CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long);
typedef CUresult(CUDAAPI* release_fn)(CUmemGenericAllocationHandle);
typedef CUresult(CUDAAPI* map_fn)(
    CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
typedef CUresult(CUDAAPI* unmap_fn)(CUdeviceptr, size_t);
typedef CUresult(CUDAAPI* access_fn)(CUdeviceptr, size_t, const CUmemAccessDesc*, size_t);
typedef CUresult(CUDAAPI* export_fn)(
    void*, CUmemGenericAllocationHandle, CUmemAllocationHandleType, unsigned long long);
typedef CUresult(CUDAAPI* import_fn)(
    CUmemGenericAllocationHandle*, void*, CUmemAllocationHandleType);
typedef CUresult(CUDAAPI* retain_fn)(CUmemGenericAllocationHandle*, void*);
typedef CUresult(CUDAAPI* properties_fn)(
    CUmemAllocationProp*, CUmemGenericAllocationHandle);
typedef CUresult(CUDAAPI* device_get_fn)(CUdevice*, int);
typedef CUresult(CUDAAPI* device_uuid_fn)(CUuuid*, CUdevice);
typedef CUresult(CUDAAPI* device_count_fn)(int*);

struct device_identity_api {
  device_get_fn get_device;
  device_uuid_fn get_uuid;
  device_count_fn get_count;
};

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static struct allocation* allocations;
static struct unmanaged_multicast* unmanaged_multicasts;
static bool enabled;
static bool cuda_seen;
static bool forked_after_cuda;
static bool failed;
static enum dyn_vmm_phase phase = DYN_VMM_ACTIVE;
static char failure[96];
static char participant_id[DYN_VMM_PARTICIPANT_ID_SIZE];
static uint64_t next_logical_handle = 1;
static int listener = -1;
static char control_directory[sizeof(((struct sockaddr_un*)0)->sun_path)];
static char socket_path[sizeof(((struct sockaddr_un*)0)->sun_path)];
static pthread_once_t real_dlsym_once = PTHREAD_ONCE_INIT;
static void* (*real_dlsym_function)(void*, const char*);
static _Atomic(uintptr_t) explicit_libcuda_handle;
static _Atomic(uintptr_t) explicit_cu_get_proc_address;
static _Atomic(uintptr_t) explicit_cu_get_proc_address_v2;

static int write_all(int, const void*, size_t);
static int pread_all(int, void*, size_t);
static int send_header(int, const struct dyn_vmm_header*, int);
static int close_owned_fd(int*, const char*);
static void set_error(struct dyn_vmm_header*, const char*);
static bool all_zero(const void*, size_t);
static int resolve_device_identity_api(struct device_identity_api*);
static struct allocation* find_logical_handle(
    CUmemGenericAllocationHandle);
static struct allocation* find_logical_resource(
    uint64_t, uint32_t, uint32_t);
static bool is_logical_handle(CUmemGenericAllocationHandle);
static CUresult unknown_logical_handle(void);
static CUresult unavailable(void);
static void fail(const char*);
static void fail_cleanup(const char*);
static int enter_context(CUcontext, struct context_scope*);
static int leave_context(const struct context_scope*);
static int send_broker_result(
    int, struct dyn_vmm_header*, const struct broker_result*);
static int cleanup_multicast_restore(void);
static bool test_response_send_failure(void);

static void
initialize_real_dlsym(void)
{
  static const char* versions[] = {
      "GLIBC_2.2.5",
      "GLIBC_2.17",
      "GLIBC_2.34",
  };
  size_t index;

  for (index = 0; index < sizeof(versions) / sizeof(versions[0]); index++) {
    real_dlsym_function =
        (void* (*)(void*, const char*))dlvsym(RTLD_NEXT, "dlsym", versions[index]);
    if (real_dlsym_function != NULL)
      return;
  }
}

static void*
real_dlsym(void* handle, const char* name)
{
  if (pthread_once(&real_dlsym_once, initialize_real_dlsym) != 0 ||
      real_dlsym_function == NULL)
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

static int
resolve_device_identity_api(struct device_identity_api* api)
{
  api->get_device = (device_get_fn)real_symbol("cuDeviceGet");
  api->get_uuid = (device_uuid_fn)real_symbol("cuDeviceGetUuid_v2");
  if (api->get_uuid == NULL)
    api->get_uuid = (device_uuid_fn)real_symbol("cuDeviceGetUuid");
  api->get_count = (device_count_fn)real_symbol("cuDeviceGetCount");
  return api->get_device != NULL && api->get_uuid != NULL &&
                 api->get_count != NULL
             ? 0
             : -1;
}

static CUresult
device_uuid(const struct device_identity_api* api, int ordinal, CUuuid* uuid)
{
  CUdevice device;
  CUresult result = api->get_device(&device, ordinal);

  return result == CUDA_SUCCESS ? api->get_uuid(uuid, device) : result;
}

static bool
is_libcuda_name(const char* path)
{
  const char* name;
  const char* current;
  static const char base[] = "libcuda.so";

  if (path == NULL)
    return false;
  name = strrchr(path, '/');
  name = name == NULL ? path : name + 1;
  if (strcmp(name, base) == 0)
    return true;
  if (strncmp(name, base, sizeof(base) - 1) != 0 ||
      name[sizeof(base) - 1] != '.')
    return false;
  current = name + sizeof(base);
  for (;;) {
    if (*current < '0' || *current > '9')
      return false;
    do {
      current++;
    } while (*current >= '0' && *current <= '9');
    if (*current == '\0')
      return true;
    if (*current != '.')
      return false;
    current++;
  }
}

static bool
is_explicit_libcuda_resolver(void* handle, void* symbol)
{
  struct link_map* map;
  Dl_info info;

  return handle != NULL && handle != RTLD_NEXT &&
      dlinfo(handle, RTLD_DI_LINKMAP, &map) == 0 && map != NULL &&
      is_libcuda_name(map->l_name) && dladdr(symbol, &info) != 0 &&
      is_libcuda_name(info.dli_fname);
}

void*
dlsym(void* handle, const char* name)
{
  void* symbol = real_dlsym(handle, name);

  if (!enabled || symbol == NULL)
    return symbol;
  if (strcmp(name, "cuGetProcAddress") == 0) {
    if (!is_explicit_libcuda_resolver(handle, symbol))
      return symbol;
    if (symbol == (void*)&cuGetProcAddress)
      return symbol;
    atomic_store(&explicit_libcuda_handle, (uintptr_t)handle);
    atomic_store(&explicit_cu_get_proc_address, (uintptr_t)symbol);
    return (void*)&cuGetProcAddress;
  }
  if (strcmp(name, "cuGetProcAddress_v2") == 0) {
    if (!is_explicit_libcuda_resolver(handle, symbol))
      return symbol;
    if (symbol == (void*)&cuGetProcAddress_v2)
      return symbol;
    atomic_store(&explicit_libcuda_handle, (uintptr_t)handle);
    atomic_store(&explicit_cu_get_proc_address_v2, (uintptr_t)symbol);
    return (void*)&cuGetProcAddress_v2;
  }
  if (strcmp(name, "cuGetProcAddress_v2_ptsz") == 0) {
    if (!is_explicit_libcuda_resolver(handle, symbol))
      return symbol;
    if (symbol == (void*)&cuGetProcAddress_v2_ptsz)
      return symbol;
    atomic_store(&explicit_libcuda_handle, (uintptr_t)handle);
    atomic_store(&explicit_cu_get_proc_address_v2, (uintptr_t)symbol);
    return (void*)&cuGetProcAddress_v2_ptsz;
  }
  return symbol;
}

static bool
valid_participant_id(const char value[DYN_VMM_PARTICIPANT_ID_SIZE])
{
  size_t index;

  for (index = 0; index < DYN_VMM_PARTICIPANT_ID_SIZE - 1; index++) {
    if (!((value[index] >= '0' && value[index] <= '9') || (value[index] >= 'a' && value[index] <= 'f')))
      return false;
  }
  return value[DYN_VMM_PARTICIPANT_ID_SIZE - 1] == '\0';
}

static bool
valid_fabric_participant_id(const char value[DYN_VMM_FABRIC_PARTICIPANT_ID_SIZE])
{
  size_t index;

  for (index = 0; index < DYN_VMM_FABRIC_PARTICIPANT_ID_SIZE; index++) {
    if (!((value[index] >= '0' && value[index] <= '9') || (value[index] >= 'a' && value[index] <= 'f')))
      return false;
  }
  return true;
}

static bool
supported_handle_type(CUmemAllocationHandleType type)
{
  return type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR || type == CU_MEM_HANDLE_TYPE_FABRIC;
}

static bool
valid_multicast_properties(const CUmulticastObjectProp* properties)
{
  return properties != NULL && properties->flags == 0 &&
      properties->size != 0 && properties->numDevices != 0 &&
      (properties->handleTypes ==
           (unsigned long long)CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR ||
       properties->handleTypes ==
           (unsigned long long)CU_MEM_HANDLE_TYPE_FABRIC);
}

static bool
valid_broker_shape(CUmemAllocationHandleType type, int fd, uint64_t bytes_size)
{
  return (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR && fd >= 0 && bytes_size == 0) ||
         (type == CU_MEM_HANDLE_TYPE_FABRIC && fd < 0 && bytes_size == DYN_VMM_FABRIC_HANDLE_SIZE);
}

static void
initialize_broker_result(
    struct broker_result* result, CUmemAllocationHandleType type,
    uint32_t object_kind)
{
  memset(result, 0, sizeof(*result));
  result->handle_type = type;
  result->object_kind = object_kind;
  result->fd = -1;
}

static int
clear_broker_result(struct broker_result* result, const char* close_error)
{
  int status = 0;

  if (result->fd >= 0 && close_owned_fd(&result->fd, close_error) != 0)
    status = -1;
  explicit_bzero(result->bytes, sizeof(result->bytes));
  explicit_bzero(
      &result->multicast_properties,
      sizeof(result->multicast_properties));
  result->bytes_size = 0;
  result->has_multicast_properties = false;
  return status;
}

static bool
valid_control_directory(const char* path)
{
  const char* component;
  size_t length;

  if (path == NULL || path[0] != '/')
    return false;
  length = strnlen(path, sizeof(control_directory));
  if (length == 0 || length == sizeof(control_directory) ||
      (length > 1 && path[length - 1] == '/'))
    return false;
  component = path + 1;
  while (*component != '\0') {
    const char* separator = strchr(component, '/');
    size_t component_length =
        separator == NULL ? strlen(component) : (size_t)(separator - component);

    if (component_length == 0 ||
        (component_length == 1 && component[0] == '.') ||
        (component_length == 2 && component[0] == '.' && component[1] == '.'))
      return false;
    if (separator == NULL)
      break;
    component = separator + 1;
  }
  return true;
}

static int
format_socket_path(char* output, size_t size, unsigned long pid)
{
  const char* separator = strcmp(control_directory, "/") == 0 ? "" : "/";
  int length = snprintf(
      output, size, "%s%s%s%lu.sock", control_directory, separator,
      DYN_VMM_SOCKET_PREFIX, pid);

  return length >= 0 && (size_t)length < size ? 0 : -1;
}

static bool
valid_socket_path(const char path[sizeof(((struct sockaddr_un*)0)->sun_path)])
{
  char canonical[sizeof(((struct sockaddr_un*)0)->sun_path)];
  const char* basename;
  const char* current;
  size_t basename_length;
  size_t control_length;
  size_t length = strnlen(path, sizeof(((struct sockaddr_un*)0)->sun_path));
  size_t prefix_length = strlen(DYN_VMM_SOCKET_PREFIX);
  unsigned long pid = 0;

  if (control_directory[0] == '\0' || length == 0 ||
      length == sizeof(((struct sockaddr_un*)0)->sun_path) ||
      !all_zero(
          path + length + 1,
          sizeof(((struct sockaddr_un*)0)->sun_path) - length - 1))
    return false;
  control_length = strlen(control_directory);
  if (strcmp(control_directory, "/") == 0) {
    if (path[0] != '/')
      return false;
    basename = path + 1;
  } else {
    if (length <= control_length + 1 ||
        memcmp(path, control_directory, control_length) != 0 ||
        path[control_length] != '/')
      return false;
    basename = path + control_length + 1;
  }
  basename_length = strlen(basename);
  if (basename_length <= prefix_length ||
      memcmp(basename, DYN_VMM_SOCKET_PREFIX, prefix_length) != 0)
    return false;
  current = basename + prefix_length;
  if (*current < '1' || *current > '9')
    return false;
  do {
    unsigned long digit = (unsigned long)(*current - '0');

    if (pid > ((unsigned long)INT_MAX - digit) / 10)
      return false;
    pid = pid * 10 + digit;
    current++;
  } while (*current >= '0' && *current <= '9');
  if (strcmp(current, ".sock") != 0 ||
      format_socket_path(canonical, sizeof(canonical), pid) != 0)
    return false;
  return strcmp(path, canonical) == 0;
}

static int
create_capability(
    const uint8_t uuid[DYN_VMM_ALLOCATION_UUID_SIZE], uint32_t object_kind,
    int* output)
{
  const int required_seals = F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL;
  struct dyn_vmm_capability capability;
  struct stat status;
  int fd = -1;
  int seals;

  memset(&capability, 0, sizeof(capability));
  capability.magic = DYN_VMM_CAPABILITY_MAGIC;
  capability.version = DYN_VMM_CAPABILITY_VERSION;
  capability.object_kind = (uint16_t)object_kind;
  memcpy(capability.allocation_uuid, uuid, sizeof(capability.allocation_uuid));
  snprintf(capability.owner_socket_path, sizeof(capability.owner_socket_path), "%s", socket_path);
  snprintf(capability.owner_participant_id, sizeof(capability.owner_participant_id), "%s", participant_id);
  fd = memfd_create("dynamo-cuda-vmm-capability", MFD_CLOEXEC | MFD_ALLOW_SEALING);
  if (fd < 0 || write_all(fd, &capability, sizeof(capability)) != 0 || fstat(fd, &status) != 0 ||
      status.st_size != (off_t)sizeof(capability) || fcntl(fd, F_ADD_SEALS, required_seals) != 0) {
    (void)close(fd);
    return -1;
  }
  seals = fcntl(fd, F_GET_SEALS);
  if (seals < 0 || (seals & required_seals) != required_seals) {
    (void)close(fd);
    return -1;
  }
  *output = fd;
  return 0;
}

static int
read_capability(int fd, struct dyn_vmm_capability* capability)
{
  const int required_seals = F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL;
  struct stat status;
  int seals;

  memset(capability, 0, sizeof(*capability));
  if (fd < 0)
    return -1;
  seals = fcntl(fd, F_GET_SEALS);
  if (seals < 0 || (seals & required_seals) != required_seals || fstat(fd, &status) != 0 ||
      status.st_size != (off_t)sizeof(*capability) || pread_all(fd, capability, sizeof(*capability)) != 0 ||
      capability->magic != DYN_VMM_CAPABILITY_MAGIC || capability->version != DYN_VMM_CAPABILITY_VERSION ||
      (capability->object_kind != DYN_VMM_ALLOCATION &&
       capability->object_kind != DYN_VMM_MULTICAST) ||
      all_zero(capability->allocation_uuid, sizeof(capability->allocation_uuid)) ||
      !valid_socket_path(capability->owner_socket_path) || !valid_participant_id(capability->owner_participant_id) ||
      !all_zero(capability->reserved_identity, sizeof(capability->reserved_identity)))
    return -1;
  return 0;
}

static int
create_fabric_token(
    const uint8_t uuid[DYN_VMM_ALLOCATION_UUID_SIZE], uint32_t object_kind,
    struct dyn_vmm_fabric_token* token)
{
  pid_t pid = getpid();

  if (pid <= 0 || (uint64_t)pid > UINT32_MAX)
    return -1;
  memset(token, 0, sizeof(*token));
  token->magic = DYN_VMM_FABRIC_TOKEN_MAGIC;
  token->version = DYN_VMM_FABRIC_TOKEN_VERSION;
  token->handle_type = CU_MEM_HANDLE_TYPE_FABRIC;
  memcpy(token->allocation_uuid, uuid, sizeof(token->allocation_uuid));
  memcpy(token->owner_participant_id, participant_id, sizeof(token->owner_participant_id));
  token->owner_namespace_pid = (uint32_t)pid;
  token->object_kind = object_kind;
  return 0;
}

static int
read_fabric_token(
    const void* encoded, struct dyn_vmm_fabric_token* token, char owner_participant[DYN_VMM_PARTICIPANT_ID_SIZE],
    char owner_socket[sizeof(((struct sockaddr_un*)0)->sun_path)])
{
  if (encoded == NULL)
    return -1;
  memcpy(token, encoded, sizeof(*token));
  if (token->magic != DYN_VMM_FABRIC_TOKEN_MAGIC || token->version != DYN_VMM_FABRIC_TOKEN_VERSION ||
      token->handle_type != CU_MEM_HANDLE_TYPE_FABRIC ||
      all_zero(token->allocation_uuid, sizeof(token->allocation_uuid)) ||
      !valid_fabric_participant_id(token->owner_participant_id) || token->owner_namespace_pid == 0 ||
      token->owner_namespace_pid > INT_MAX ||
      (token->object_kind != DYN_VMM_ALLOCATION &&
       token->object_kind != DYN_VMM_MULTICAST) ||
      format_socket_path(owner_socket, sizeof(((struct sockaddr_un*)0)->sun_path), token->owner_namespace_pid) != 0)
    return -1;
  memcpy(owner_participant, token->owner_participant_id, sizeof(token->owner_participant_id));
  owner_participant[DYN_VMM_PARTICIPANT_ID_SIZE - 1] = '\0';
  return 0;
}

#ifdef DYN_VMM_TESTING
int
dyn_vmm_test_capability_valid(int fd)
{
  struct dyn_vmm_capability capability;

  return read_capability(fd, &capability) == 0;
}

int
dyn_vmm_test_fabric_token_valid(const void* encoded)
{
  struct dyn_vmm_fabric_token token;
  char owner_participant[DYN_VMM_PARTICIPANT_ID_SIZE];
  char owner_socket[sizeof(((struct sockaddr_un*)0)->sun_path)];
  return read_fabric_token(encoded, &token, owner_participant, owner_socket) == 0;
}

int
dyn_vmm_test_broker_shape(CUmemAllocationHandleType type, int fd, uint64_t bytes_size)
{
  return valid_broker_shape(type, fd, bytes_size);
}

int
dyn_vmm_test_corrupt_detached_placement(
    const char* field, int value)
{
  struct allocation* allocation;
  int result = -1;

  pthread_mutex_lock(&lock);
  for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
    if (!allocation->detached)
      continue;
    if (strcmp(field, "device") == 0)
      allocation->device_ordinal = value;
    else if (strcmp(field, "source") == 0)
      allocation->source_device_ordinal = value;
    else if (strcmp(field, "properties") == 0)
      allocation->properties.location.id = value;
    else if (strcmp(field, "access") == 0 && allocation->access_count != 0)
      allocation->access[0].location.id = value;
    else
      break;
    result = 0;
    break;
  }
  pthread_mutex_unlock(&lock);
  return result;
}
#endif

CUresult CUDAAPI
cuMulticastBindAddr(
    CUmemGenericAllocationHandle multicast_handle, size_t multicast_offset,
    CUdeviceptr address, size_t size, unsigned long long flags)
{
  typedef CUresult(CUDAAPI * function_type)(
      CUmemGenericAllocationHandle, size_t, CUdeviceptr, size_t,
      unsigned long long);
  function_type function =
      (function_type)real_symbol("cuMulticastBindAddr");

  if (!enabled || !is_logical_handle(multicast_handle))
    return function != NULL
        ? function(multicast_handle, multicast_offset, address, size, flags)
        : unavailable();
  pthread_mutex_lock(&lock);
  cuda_seen = true;
  if (find_logical_handle(multicast_handle) == NULL)
    (void)unknown_logical_handle();
  else
    fail("managed CUDA multicast address binding is unsupported");
  pthread_mutex_unlock(&lock);
  return CUDA_ERROR_NOT_SUPPORTED;
}

#if CUDA_VERSION >= 13010
CUresult CUDAAPI
cuMulticastBindAddr_v2(
    CUmemGenericAllocationHandle multicast_handle, CUdevice device,
    size_t multicast_offset, CUdeviceptr address, size_t size,
    unsigned long long flags)
{
  typedef CUresult(CUDAAPI * function_type)(
      CUmemGenericAllocationHandle, CUdevice, size_t, CUdeviceptr, size_t,
      unsigned long long);
  function_type function =
      (function_type)real_symbol("cuMulticastBindAddr_v2");

  if (!enabled || !is_logical_handle(multicast_handle))
    return function != NULL
        ? function(
              multicast_handle, device, multicast_offset, address, size,
              flags)
        : unavailable();
  pthread_mutex_lock(&lock);
  cuda_seen = true;
  if (find_logical_handle(multicast_handle) == NULL)
    (void)unknown_logical_handle();
  else
    fail("managed CUDA multicast address binding is unsupported");
  pthread_mutex_unlock(&lock);
  return CUDA_ERROR_NOT_SUPPORTED;
}
#endif

CUresult CUDAAPI
cuMulticastUnbind(
    CUmemGenericAllocationHandle multicast_handle, CUdevice device,
    size_t multicast_offset, size_t size)
{
  typedef CUresult(CUDAAPI * function_type)(
      CUmemGenericAllocationHandle, CUdevice, size_t, size_t);
  function_type function =
      (function_type)real_symbol("cuMulticastUnbind");
  struct allocation* multicast;
  CUresult result;

  if (!enabled || !is_logical_handle(multicast_handle))
    return function != NULL
        ? function(multicast_handle, device, multicast_offset, size)
        : unavailable();
  pthread_mutex_lock(&lock);
  cuda_seen = true;
  multicast = find_logical_handle(multicast_handle);
  if (multicast == NULL) {
    result = unknown_logical_handle();
  } else if (multicast->object_kind != DYN_VMM_MULTICAST ||
             !multicast->application_handle_live ||
             multicast->real_handle == 0 || !multicast->bound ||
             device != multicast->member_device ||
             multicast_offset != multicast->multicast_offset ||
             size != multicast->bind_size) {
    fail("invalid managed CUDA multicast unbind");
    result = CUDA_ERROR_INVALID_VALUE;
  } else {
    result = function != NULL
                 ? function(
                       multicast->real_handle, device, multicast_offset, size)
                 : unavailable();
    if (result == CUDA_SUCCESS) {
      multicast->bound = false;
      memset(
          multicast->backing_allocation_uuid, 0,
          sizeof(multicast->backing_allocation_uuid));
      multicast->backing_role = 0;
      multicast->bind_api = 0;
      multicast->bind_size = 0;
    }
  }
  pthread_mutex_unlock(&lock);
  return result;
}

static const struct dyn_vmm_placement*
find_placement(const struct dyn_vmm_placement* placements, size_t count, const CUuuid* gpu_uuid)
{
  size_t index;

  for (index = 0; index < count; index++) {
    if (memcmp(placements[index].source_gpu_uuid, gpu_uuid->bytes, sizeof(gpu_uuid->bytes)) == 0)
      return &placements[index];
  }
  return NULL;
}

static int
set_placement(
    int client, const struct dyn_vmm_header* request, struct dyn_vmm_header* response,
    const struct dyn_vmm_placement* placements)
{
  const struct allocation* previous;
  struct allocation* allocation;
  struct device_identity_api identity_api;
  struct target_member {
    struct allocation* allocation;
    CUdevice device;
  } *target_members = NULL;
  size_t multicast_count = 0;
  size_t target_member_index = 0;
  size_t index;

  for (index = 0; index < request->count; index++) {
    size_t previous_index;

    if (placements[index].device_ordinal < 0 || placements[index].reserved != 0 ||
        all_zero(placements[index].source_gpu_uuid, sizeof(placements[index].source_gpu_uuid)) ||
        all_zero(placements[index].target_gpu_uuid, sizeof(placements[index].target_gpu_uuid))) {
      set_error(response, "invalid CUDA VMM placement entry");
      return send_header(client, response, -1);
    }
    for (previous_index = 0; previous_index < index; previous_index++) {
      if (memcmp(
              placements[previous_index].source_gpu_uuid, placements[index].source_gpu_uuid,
              sizeof(placements[index].source_gpu_uuid)) == 0) {
        set_error(response, "duplicate CUDA VMM source GPU UUID");
        return send_header(client, response, -1);
      }
      if (memcmp(
              placements[previous_index].target_gpu_uuid, placements[index].target_gpu_uuid,
              sizeof(placements[index].target_gpu_uuid)) == 0) {
        set_error(response, "duplicate CUDA VMM target GPU UUID");
        return send_header(client, response, -1);
      }
      if (placements[previous_index].device_ordinal == placements[index].device_ordinal) {
        set_error(response, "duplicate CUDA VMM target ordinal");
        return send_header(client, response, -1);
      }
    }
  }
  if (resolve_device_identity_api(&identity_api) != 0) {
    set_error(response, "cannot resolve CUDA device identity API");
    return send_header(client, response, -1);
  }
  for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
    const struct dyn_vmm_placement* placement;
    size_t access_index;

    if (!allocation->detached)
      continue;
    placement = find_placement(placements, request->count, &allocation->gpu_uuid);
    if (placement == NULL ||
        allocation->device_ordinal != allocation->source_device_ordinal ||
        (allocation->object_kind == DYN_VMM_ALLOCATION &&
         (allocation->properties.location.type !=
              CU_MEM_LOCATION_TYPE_DEVICE ||
          allocation->properties.location.id !=
              allocation->device_ordinal)) ||
        (allocation->object_kind == DYN_VMM_MULTICAST &&
         (!allocation->member_added || !allocation->bound)) ||
        (allocation->access_count != 0 && allocation->access == NULL)) {
      set_error(response, "detached CUDA placement metadata is inconsistent");
      return send_header(client, response, -1);
    }
    for (previous = allocations; previous != allocation; previous = previous->next) {
      if (previous->detached &&
          memcmp(previous->gpu_uuid.bytes, allocation->gpu_uuid.bytes, sizeof(allocation->gpu_uuid.bytes)) == 0 &&
          previous->source_device_ordinal != allocation->source_device_ordinal) {
        set_error(response, "source CUDA GPU UUID has inconsistent saved ordinals");
        return send_header(client, response, -1);
      }
    }
    for (access_index = 0; access_index < allocation->access_count; access_index++) {
      if (allocation->access[access_index].location.type != CU_MEM_LOCATION_TYPE_DEVICE ||
          allocation->access[access_index].location.id != allocation->device_ordinal) {
        set_error(response, "detached CUDA access placement is inconsistent");
        return send_header(client, response, -1);
      }
    }
    if (allocation->object_kind == DYN_VMM_MULTICAST)
      multicast_count++;
  }
  for (index = 0; index < request->count; index++) {
    bool found = false;

    for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
      if (allocation->detached &&
          memcmp(
              placements[index].source_gpu_uuid, allocation->gpu_uuid.bytes,
              sizeof(placements[index].source_gpu_uuid)) == 0) {
        found = true;
        break;
      }
    }
    if (!found) {
      set_error(response, "unknown CUDA VMM source GPU UUID");
      return send_header(client, response, -1);
    }
  }
  if (multicast_count != 0) {
    target_members =
        malloc(multicast_count * sizeof(*target_members));
    if (target_members == NULL) {
      set_error(response, "cannot resolve CUDA multicast placement devices");
      return send_header(client, response, -1);
    }
    for (allocation = allocations; allocation != NULL;
         allocation = allocation->next) {
      const struct dyn_vmm_placement* placement;

      if (!allocation->detached ||
          allocation->object_kind != DYN_VMM_MULTICAST)
        continue;
      placement = find_placement(
          placements, request->count, &allocation->gpu_uuid);
      target_members[target_member_index].allocation = allocation;
      if (identity_api.get_device(
              &target_members[target_member_index].device,
              placement->device_ordinal) != CUDA_SUCCESS) {
        free(target_members);
        set_error(response, "cannot resolve target CUDA multicast member");
        return send_header(client, response, -1);
      }
      target_member_index++;
    }
  }
  target_member_index = 0;
  for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
    const struct dyn_vmm_placement* placement;
    size_t access_index;

    if (!allocation->detached)
      continue;
    placement = find_placement(placements, request->count, &allocation->gpu_uuid);
    allocation->device_ordinal = placement->device_ordinal;
    if (allocation->object_kind == DYN_VMM_ALLOCATION) {
      allocation->properties.location.id = placement->device_ordinal;
    } else {
      if (target_member_index >= multicast_count ||
          target_members[target_member_index].allocation != allocation) {
        free(target_members);
        set_error(response, "inconsistent CUDA multicast placement plan");
        return send_header(client, response, -1);
      }
      allocation->member_device =
          target_members[target_member_index].device;
      target_member_index++;
    }
    for (access_index = 0; access_index < allocation->access_count; access_index++) {
      if (allocation->access[access_index].location.type == CU_MEM_LOCATION_TYPE_DEVICE)
        allocation->access[access_index].location.id = placement->device_ordinal;
    }
  }
  free(target_members);
  return send_header(client, response, -1);
}

static int
decode_multicast_record(
    const struct dyn_vmm_header* request, char* payload,
    struct dyn_vmm_record* record, CUmulticastObjectProp** properties,
    CUmemAccessDesc** access,
    struct dyn_vmm_multicast_record** multicast, size_t* metadata_size)
{
  uint64_t decoded_size;

  if (request->payload_size < sizeof(*record))
    return -1;
  memcpy(record, payload, sizeof(*record));
  decoded_size = sizeof(*record) + record->properties_size +
      (uint64_t)record->access_count * record->access_size +
      sizeof(**multicast);
  if (decoded_size > request->payload_size || decoded_size > SIZE_MAX ||
      record->object_kind != DYN_VMM_MULTICAST ||
      record->access_size != sizeof(**access) ||
      !supported_handle_type(
          (CUmemAllocationHandleType)record->requested_handle_type) ||
      (record->flags &
       ~(DYN_VMM_APPLICATION_HANDLE_LIVE |
         DYN_VMM_RETAIN_RESTORE_HANDLE)) != 0)
    return -1;
  *properties = record->properties_size == 0
      ? NULL
      : (CUmulticastObjectProp*)(payload + sizeof(*record));
  *access = (CUmemAccessDesc*)(
      payload + sizeof(*record) + record->properties_size);
  *multicast = (struct dyn_vmm_multicast_record*)(
      (char*)*access + record->access_count * record->access_size);
  *metadata_size = (size_t)decoded_size;
  return 0;
}

static int
reconcile_multicast_metadata(
    struct allocation* allocation, const struct dyn_vmm_record* record,
    const CUmulticastObjectProp* properties, CUmemAccessDesc* access,
    const struct dyn_vmm_multicast_record* multicast, bool owner)
{
  size_t index;

  if ((owner && (properties == NULL ||
                 record->properties_size != sizeof(*properties))) ||
      (!owner && properties != NULL) ||
      record->access_count != allocation->access_count ||
      record->offset != 0 || record->address != allocation->address ||
      record->size != allocation->size ||
      record->device_ordinal != allocation->device_ordinal ||
      memcmp(
          record->gpu_uuid, allocation->gpu_uuid.bytes,
          sizeof(record->gpu_uuid)) != 0 ||
      multicast->reserved != 0 ||
      !all_zero(
          multicast->backing_allocation_uuid,
          sizeof(multicast->backing_allocation_uuid)) ||
      multicast->backing_object_id == 0 ||
      multicast->multicast_offset != 0 ||
      multicast->memory_offset != 0 ||
      multicast->bind_size != allocation->size ||
      multicast->bind_flags != 0 ||
      multicast->object_flags != allocation->multicast_properties.flags ||
      multicast->object_handle_types !=
          allocation->multicast_properties.handleTypes ||
      multicast->object_size != allocation->multicast_properties.size ||
      multicast->num_devices !=
          allocation->multicast_properties.numDevices ||
      multicast->backing_role != allocation->backing_role ||
      (multicast->bind_api != DYN_VMM_MULTICAST_BIND_MEM &&
       multicast->bind_api != DYN_VMM_MULTICAST_BIND_MEM_V2) ||
      (owner &&
       memcmp(
           properties, &allocation->multicast_properties,
           sizeof(*properties)) != 0))
    return -1;
  for (index = 0; index < record->access_count; index++) {
    if (access[index].location.type == CU_MEM_LOCATION_TYPE_DEVICE) {
      if (access[index].location.id != allocation->source_device_ordinal)
        return -1;
      access[index].location.id = allocation->device_ordinal;
    }
  }
  return memcmp(
             access, allocation->access,
             record->access_count * sizeof(*access)) == 0
      ? 0
      : -1;
}

static int
restore_multicast_owner(
    int client, const struct dyn_vmm_header* request,
    struct dyn_vmm_header* response, char* payload, int passed_fd)
{
  typedef CUresult(CUDAAPI * create_type)(
      CUmemGenericAllocationHandle*, const CUmulticastObjectProp*);
  typedef CUresult(CUDAAPI * add_type)(
      CUmemGenericAllocationHandle, CUdevice);
  create_type create = (create_type)real_symbol("cuMulticastCreate");
  add_type add = (add_type)real_symbol("cuMulticastAddDevice");
  export_fn export_handle =
      (export_fn)real_symbol("cuMemExportToShareableHandle");
  release_fn release = (release_fn)real_symbol("cuMemRelease");
  struct dyn_vmm_record record;
  struct dyn_vmm_multicast_record* multicast;
  CUmulticastObjectProp* properties;
  CUmemAccessDesc* access;
  struct allocation* allocation;
  struct broker_result exported;
  struct context_scope scope;
  CUmemGenericAllocationHandle handle = 0;
  size_t metadata_size;
  bool context_entered = false;
  bool owns_handle = false;
  const char* primary_error =
      "multicast owner restore failed; process must not resume";

  initialize_broker_result(
      &exported, (CUmemAllocationHandleType)request->handle_type,
      DYN_VMM_MULTICAST);
  if (passed_fd >= 0 || request->object_kind != DYN_VMM_MULTICAST ||
      decode_multicast_record(
          request, payload, &record, &properties, &access, &multicast,
          &metadata_size) != 0 ||
      metadata_size != request->payload_size ||
      record.role != DYN_VMM_OWNER ||
      request->handle_type != record.requested_handle_type) {
    set_error(response, "invalid multicast owner restore payload");
    return send_header(client, response, -1);
  }
  allocation = find_logical_resource(
      record.object_id, DYN_VMM_OWNER, DYN_VMM_MULTICAST);
  if (allocation == NULL || !allocation->detached ||
      allocation->requested_handle_type !=
          (CUmemAllocationHandleType)record.requested_handle_type ||
      allocation->application_handle_live !=
          ((record.flags & DYN_VMM_APPLICATION_HANDLE_LIVE) != 0) ||
      reconcile_multicast_metadata(
          allocation, &record, properties, access, multicast, true) != 0 ||
      create == NULL || add == NULL || export_handle == NULL ||
      release == NULL || enter_context(allocation->context, &scope) != 0)
    goto failed;
  context_entered = true;
  if (create(&handle, properties) != CUDA_SUCCESS)
    goto failed;
  owns_handle = true;
  if (add(handle, allocation->member_device) != CUDA_SUCCESS)
    goto failed;
  if (exported.handle_type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    if (export_handle(
            &exported.fd, handle, exported.handle_type, 0) != CUDA_SUCCESS)
      goto failed;
  } else {
    if (export_handle(
            exported.bytes, handle, exported.handle_type, 0) != CUDA_SUCCESS)
      goto failed;
    exported.bytes_size = sizeof(exported.bytes);
  }
  allocation->real_handle = handle;
  allocation->backing_object_id = multicast->backing_object_id;
  allocation->bind_api = multicast->bind_api;
  allocation->member_added = true;
  owns_handle = false;
  if (leave_context(&scope) != 0)
    goto failed;
  context_entered = false;
  if (send_broker_result(client, response, &exported) != 0 ||
      clear_broker_result(
          &exported, "cannot close multicast owner restore broker") != 0)
    goto failed;
  return 0;

failed:
  fail(primary_error);
  phase = DYN_VMM_FAILED;
  if (allocation != NULL && allocation->real_handle == handle) {
    allocation->real_handle = 0;
    allocation->member_added = false;
  }
  if ((owns_handle || handle != 0) && release != NULL &&
      release(handle) != CUDA_SUCCESS)
    fail_cleanup("cannot release failed multicast owner handle");
  (void)clear_broker_result(
      &exported, "cannot close failed multicast owner broker");
  if (context_entered && leave_context(&scope) != 0)
    fail_cleanup("cannot restore context after multicast owner failure");
  (void)cleanup_multicast_restore();
  set_error(response, primary_error);
  return send_header(client, response, -1);
}

static int
restore_multicast_importer(
    int client, const struct dyn_vmm_header* request,
    struct dyn_vmm_header* response, char* payload, int* imported_fd)
{
  typedef CUresult(CUDAAPI * add_type)(
      CUmemGenericAllocationHandle, CUdevice);
  import_fn import_handle =
      (import_fn)real_symbol("cuMemImportFromShareableHandle");
  add_type add = (add_type)real_symbol("cuMulticastAddDevice");
  release_fn release = (release_fn)real_symbol("cuMemRelease");
  struct dyn_vmm_record record;
  struct dyn_vmm_multicast_record* multicast;
  CUmulticastObjectProp* properties;
  CUmemAccessDesc* access;
  struct allocation* allocation;
  struct context_scope scope;
  CUmemGenericAllocationHandle handle = 0;
  size_t metadata_size;
  size_t broker_size;
  void* import_value;
  bool context_entered = false;
  const char* primary_error =
      "multicast importer restore failed; process must not resume";

  if (request->object_kind != DYN_VMM_MULTICAST ||
      decode_multicast_record(
          request, payload, &record, &properties, &access, &multicast,
          &metadata_size) != 0 ||
      record.role != DYN_VMM_IMPORTER || properties != NULL ||
      request->handle_type != record.requested_handle_type) {
    set_error(response, "invalid multicast importer restore payload");
    return send_header(client, response, -1);
  }
  broker_size = (size_t)request->payload_size - metadata_size;
  if ((record.requested_handle_type ==
           CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
       (broker_size != 0 || *imported_fd < 0)) ||
      (record.requested_handle_type == CU_MEM_HANDLE_TYPE_FABRIC &&
       (broker_size != DYN_VMM_FABRIC_HANDLE_SIZE ||
        *imported_fd >= 0))) {
    set_error(response, "invalid multicast importer broker");
    return send_header(client, response, -1);
  }
  import_value =
      record.requested_handle_type ==
              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
          ? (void*)(uintptr_t)*imported_fd
          : payload + metadata_size;
  allocation = find_logical_resource(
      record.object_id, DYN_VMM_IMPORTER, DYN_VMM_MULTICAST);
  if (allocation == NULL || !allocation->detached ||
      allocation->requested_handle_type !=
          (CUmemAllocationHandleType)record.requested_handle_type ||
      allocation->application_handle_live !=
          ((record.flags & DYN_VMM_APPLICATION_HANDLE_LIVE) != 0) ||
      reconcile_multicast_metadata(
          allocation, &record, properties, access, multicast, false) != 0 ||
      import_handle == NULL || add == NULL || release == NULL ||
      enter_context(allocation->context, &scope) != 0)
    goto failed;
  context_entered = true;
  if (import_handle(
          &handle, import_value,
          (CUmemAllocationHandleType)record.requested_handle_type) !=
      CUDA_SUCCESS)
    goto failed;
  if (record.requested_handle_type ==
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
      close_owned_fd(
          imported_fd,
          "cannot close multicast importer broker FD") != 0)
    goto failed;
  if (record.requested_handle_type == CU_MEM_HANDLE_TYPE_FABRIC)
    explicit_bzero(payload + metadata_size, broker_size);
  if (add(handle, allocation->member_device) != CUDA_SUCCESS)
    goto failed;
  allocation->real_handle = handle;
  allocation->backing_object_id = multicast->backing_object_id;
  allocation->bind_api = multicast->bind_api;
  allocation->member_added = true;
  if (leave_context(&scope) != 0)
    goto failed;
  context_entered = false;
  if (send_header(client, response, -1) != 0)
    goto failed;
  return 0;

failed:
  fail(primary_error);
  phase = DYN_VMM_FAILED;
  if (allocation != NULL && allocation->real_handle == handle) {
    allocation->real_handle = 0;
    allocation->member_added = false;
  }
  if (handle != 0 && release != NULL &&
      release(handle) != CUDA_SUCCESS)
    fail_cleanup("cannot release failed multicast importer handle");
  if (close_owned_fd(
          imported_fd,
          "cannot close failed multicast importer broker FD") != 0)
    fail_cleanup("cannot close failed multicast importer broker FD");
  if (record.requested_handle_type == CU_MEM_HANDLE_TYPE_FABRIC)
    explicit_bzero(payload + metadata_size, broker_size);
  if (context_entered && leave_context(&scope) != 0)
    fail_cleanup("cannot restore context after multicast importer failure");
  (void)cleanup_multicast_restore();
  set_error(response, primary_error);
  return send_header(client, response, -1);
}

static struct allocation*
find_logical_object(uint64_t object_id, uint32_t role)
{
  struct allocation* allocation;

  for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
    if (allocation->object_id == object_id && allocation->role == role)
      return allocation;
  }
  return NULL;
}

static CUresult
unavailable(void)
{
  return CUDA_ERROR_NOT_SUPPORTED;
}

static void
fail(const char* message)
{
  if (!failed)
    snprintf(failure, sizeof(failure), "%s", message);
  failed = true;
}

static void
fail_cleanup(const char* message)
{
  size_t used;

  if (!failed) {
    fail(message);
    return;
  }
  used = strlen(failure);
  if (used + 2 < sizeof(failure))
    snprintf(failure + used, sizeof(failure) - used, "; %s", message);
}

static int
close_owned_fd(int* fd, const char* message)
{
  int result;

  if (*fd < 0)
    return 0;
  result = close(*fd);
  *fd = -1;
  if (result != 0)
    fail_cleanup(message);
  return result;
}

static bool
all_zero(const void* value, size_t size)
{
  const uint8_t* bytes = value;
  size_t index;

  for (index = 0; index < size; index++) {
    if (bytes[index] != 0)
      return false;
  }
  return true;
}

static int
random_bytes(void* output, size_t size)
{
  uint8_t* current = output;

  while (size != 0) {
    ssize_t received = getrandom(current, size, 0);

    if (received < 0 && errno == EINTR)
      continue;
    if (received <= 0)
      return -1;
    current += received;
    size -= (size_t)received;
  }
  return 0;
}

static int
random_uuid(uint8_t uuid[DYN_VMM_ALLOCATION_UUID_SIZE])
{
  do {
    if (random_bytes(uuid, DYN_VMM_ALLOCATION_UUID_SIZE) != 0)
      return -1;
  } while (all_zero(uuid, DYN_VMM_ALLOCATION_UUID_SIZE));
  return 0;
}

static struct allocation*
find_logical_handle(CUmemGenericAllocationHandle handle)
{
  struct allocation* allocation;

  for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
    if (allocation->logical_handle == handle)
      return allocation;
  }
  return NULL;
}

static struct allocation*
find_logical_resource(
    uint64_t object_id, uint32_t role, uint32_t object_kind)
{
  struct allocation* allocation = find_logical_object(object_id, role);

  return allocation != NULL && allocation->object_kind == object_kind
             ? allocation
             : NULL;
}

static struct unmanaged_multicast*
find_unmanaged_multicast(CUmemGenericAllocationHandle handle)
{
  struct unmanaged_multicast* object;

  for (object = unmanaged_multicasts; object != NULL;
       object = object->next) {
    if (object->handle == handle)
      return object;
  }
  return NULL;
}

static void
record_unmanaged_multicast(CUmemGenericAllocationHandle handle)
{
  struct unmanaged_multicast* object = calloc(1, sizeof(*object));

  if (object == NULL) {
    fail("cannot record unmanaged CUDA multicast object");
    return;
  }
  object->handle = handle;
  object->next = unmanaged_multicasts;
  unmanaged_multicasts = object;
}

static void
remove_unmanaged_multicast(CUmemGenericAllocationHandle handle)
{
  struct unmanaged_multicast** cursor = &unmanaged_multicasts;

  while (*cursor != NULL) {
    if ((*cursor)->handle == handle) {
      struct unmanaged_multicast* object = *cursor;

      *cursor = object->next;
      free(object);
      return;
    }
    cursor = &(*cursor)->next;
  }
}

static bool
is_logical_handle(CUmemGenericAllocationHandle handle)
{
  return ((uint64_t)handle & LOGICAL_HANDLE_TAG_MASK) == LOGICAL_HANDLE_TAG;
}

static CUresult
unknown_logical_handle(void)
{
  fail("unknown logical generic allocation handle");
  return CUDA_ERROR_INVALID_HANDLE;
}

static CUresult
allocate_logical_handle(CUmemGenericAllocationHandle* output)
{
  CUmemGenericAllocationHandle candidate;

  if (next_logical_handle == 0 ||
      next_logical_handle > LOGICAL_HANDLE_VALUE_MASK)
    return CUDA_ERROR_OUT_OF_MEMORY;
  candidate = (CUmemGenericAllocationHandle)(
      LOGICAL_HANDLE_TAG | next_logical_handle++);
  if (find_logical_handle(candidate) != NULL)
    return CUDA_ERROR_OUT_OF_MEMORY;
  *output = candidate;
  return CUDA_SUCCESS;
}

static int
record_gpu_identity(struct allocation* allocation)
{
  struct device_identity_api identity_api;

  if (allocation->properties.location.type != CU_MEM_LOCATION_TYPE_DEVICE)
    return -1;
  allocation->device_ordinal = allocation->properties.location.id;
  allocation->source_device_ordinal = allocation->device_ordinal;
  return resolve_device_identity_api(&identity_api) == 0 &&
                 device_uuid(&identity_api, allocation->device_ordinal, &allocation->gpu_uuid) == CUDA_SUCCESS
             ? 0
             : -1;
}

static int
record_multicast_member(
    struct allocation* allocation, CUdevice device)
{
  struct device_identity_api identity_api;
  int count;
  int ordinal;

  if (resolve_device_identity_api(&identity_api) != 0 ||
      identity_api.get_count(&count) != CUDA_SUCCESS || count <= 0 ||
      identity_api.get_uuid(&allocation->gpu_uuid, device) != CUDA_SUCCESS)
    return -1;
  for (ordinal = 0; ordinal < count; ordinal++) {
    CUdevice candidate;

    if (identity_api.get_device(&candidate, ordinal) == CUDA_SUCCESS &&
        candidate == device)
      break;
  }
  if (ordinal == count)
    return -1;
  allocation->device_ordinal = ordinal;
  allocation->source_device_ordinal = ordinal;
  allocation->member_device = device;
  return 0;
}

static struct allocation*
find_mapping(CUdeviceptr address, size_t size)
{
  struct allocation* allocation;

  for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
    if (allocation->mapped && allocation->address == address && allocation->size == size)
      return allocation;
  }
  return NULL;
}

static struct allocation*
find_object(const uint8_t uuid[DYN_VMM_ALLOCATION_UUID_SIZE], uint32_t role)
{
  struct allocation* allocation;

  for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
    if (allocation->role == role && allocation->exported &&
        memcmp(allocation->allocation_uuid, uuid, DYN_VMM_ALLOCATION_UUID_SIZE) == 0)
      return allocation;
  }
  return NULL;
}

static void
remove_allocation(struct allocation* allocation)
{
  struct allocation** cursor = &allocations;

  while (*cursor != NULL) {
    if (*cursor == allocation) {
      *cursor = allocation->next;
      free(allocation->access);
      free(allocation);
      return;
    }
    cursor = &(*cursor)->next;
  }
}

static int
current_context(CUcontext* context)
{
  typedef CUresult(CUDAAPI * function_type)(CUcontext*);
  function_type function = (function_type)real_symbol("cuCtxGetCurrent");

  return function != NULL && function(context) == CUDA_SUCCESS && *context != NULL ? 0 : -1;
}

static int
enter_context(CUcontext context, struct context_scope* scope)
{
  typedef CUresult(CUDAAPI * get_type)(CUcontext*);
  typedef CUresult(CUDAAPI * set_type)(CUcontext);
  get_type get = (get_type)real_symbol("cuCtxGetCurrent");
  set_type set = (set_type)real_symbol("cuCtxSetCurrent");

  memset(scope, 0, sizeof(*scope));
  if (get == NULL || set == NULL || get(&scope->previous) != CUDA_SUCCESS)
    return -1;
  if (scope->previous != context) {
    if (set(context) != CUDA_SUCCESS)
      return -1;
    scope->changed = true;
  }
  return 0;
}

static int
leave_context(const struct context_scope* scope)
{
  typedef CUresult(CUDAAPI * function_type)(CUcontext);
  function_type function = (function_type)real_symbol("cuCtxSetCurrent");

  return !scope->changed ||
          (function != NULL && function(scope->previous) == CUDA_SUCCESS)
      ? 0
      : -1;
}

static bool
overlaps(const struct allocation* allocation, CUdeviceptr address, size_t size)
{
  bool allocation_overflow;
  bool range_overflow;
  CUdeviceptr allocation_end;
  CUdeviceptr range_end;

  if (!allocation->mapped || allocation->size == 0 || size == 0)
    return false;
  allocation_overflow =
      allocation->address > UINT64_MAX - allocation->size;
  range_overflow = address > UINT64_MAX - size;
  allocation_end = allocation_overflow
      ? UINT64_MAX
      : allocation->address + allocation->size;
  range_end = range_overflow ? UINT64_MAX : address + size;
  return (allocation_overflow || address < allocation_end) &&
      (range_overflow || allocation->address < range_end);
}

static int
compare_access_updates(const void* left, const void* right)
{
  const struct access_update* first = left;
  const struct access_update* second = right;

  if (first->allocation->address < second->allocation->address)
    return -1;
  return first->allocation->address > second->allocation->address ? 1 : 0;
}

#ifdef DYN_VMM_TESTING
size_t
dyn_vmm_test_access(
    CUdeviceptr address, size_t size, CUmemAccessDesc* access,
    size_t capacity)
{
  struct allocation* allocation;
  size_t count = 0;

  pthread_mutex_lock(&lock);
  allocation = find_mapping(address, size);
  if (allocation != NULL) {
    count = allocation->access_count;
    if (access != NULL && capacity != 0) {
      size_t copied = count < capacity ? count : capacity;

      if (copied != 0)
        memcpy(access, allocation->access, copied * sizeof(*access));
    }
  }
  pthread_mutex_unlock(&lock);
  return count;
}
#endif

static int
write_all(int fd, const void* buffer, size_t size)
{
  const char* current = buffer;

  while (size != 0) {
    ssize_t written = write(fd, current, size);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0)
      return -1;
    current += written;
    size -= (size_t)written;
  }
  return 0;
}

static int
read_all(int fd, void* buffer, size_t size)
{
  char* current = buffer;

  while (size != 0) {
    ssize_t received = read(fd, current, size);
    if (received < 0 && errno == EINTR)
      continue;
    if (received <= 0)
      return -1;
    current += received;
    size -= (size_t)received;
  }
  return 0;
}

static int
pread_all(int fd, void* buffer, size_t size)
{
  char* current = buffer;
  off_t offset = 0;

  while (size != 0) {
    ssize_t received = pread(fd, current, size, offset);

    if (received < 0 && errno == EINTR)
      continue;
    if (received <= 0)
      return -1;
    current += received;
    offset += received;
    size -= (size_t)received;
  }
  return 0;
}

static int
receive_header(int fd, struct dyn_vmm_header* header, int* passed_fd)
{
  char control[CMSG_SPACE(2 * sizeof(int))] = {0};
  struct iovec vector = {.iov_base = header, .iov_len = sizeof(*header)};
  struct msghdr message = {
      .msg_iov = &vector,
      .msg_iovlen = 1,
      .msg_control = control,
      .msg_controllen = sizeof(control),
  };
  struct cmsghdr* item;
  ssize_t size;
  bool extra_fd = false;

  *passed_fd = -1;
  do {
    size = recvmsg(fd, &message, MSG_WAITALL | MSG_CMSG_CLOEXEC);
  } while (size < 0 && errno == EINTR);
  for (item = CMSG_FIRSTHDR(&message); item != NULL; item = CMSG_NXTHDR(&message, item)) {
    size_t count;
    size_t index;
    int* received;

    if (item->cmsg_level != SOL_SOCKET || item->cmsg_type != SCM_RIGHTS ||
        item->cmsg_len < CMSG_LEN(sizeof(int)))
      continue;
    count = (item->cmsg_len - CMSG_LEN(0)) / sizeof(int);
    received = (int*)CMSG_DATA(item);
    for (index = 0; index < count; index++) {
      if (*passed_fd < 0)
        *passed_fd = received[index];
      else {
        (void)close(received[index]);
        extra_fd = true;
      }
    }
  }
  return size == (ssize_t)sizeof(*header) && (message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) == 0 && !extra_fd ? 0 : -1;
}

static int
send_header(int fd, const struct dyn_vmm_header* header, int passed_fd)
{
  char control[CMSG_SPACE(sizeof(int))] = {0};
  struct iovec vector = {.iov_base = (void*)header, .iov_len = sizeof(*header)};
  struct msghdr message = {.msg_iov = &vector, .msg_iovlen = 1};
  ssize_t size;

  if (test_response_send_failure())
    return -1;
  if (passed_fd >= 0) {
    struct cmsghdr* item;
    message.msg_control = control;
    message.msg_controllen = sizeof(control);
    item = CMSG_FIRSTHDR(&message);
    item->cmsg_level = SOL_SOCKET;
    item->cmsg_type = SCM_RIGHTS;
    item->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(item), &passed_fd, sizeof(passed_fd));
  }
  do {
    size = sendmsg(fd, &message, MSG_NOSIGNAL);
  } while (size < 0 && errno == EINTR);
  return size == (ssize_t)sizeof(*header) ? 0 : -1;
}

static int
set_socket_timeouts(int fd)
{
  struct timeval timeout = {.tv_sec = BROKER_TIMEOUT_SECONDS};

  return setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0 &&
                 setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0
             ? 0
             : -1;
}

static int
initialize_identity(void)
{
  unsigned char random[16];
  size_t index;

  if (random_bytes(random, sizeof(random)) != 0)
    return -1;
  for (index = 0; index < sizeof(random); index++)
    snprintf(
        participant_id + index * 2, sizeof(participant_id) - index * 2,
        "%02x", random[index]);
  return 0;
}

static CUresult
export_raw_owner(struct allocation* allocation, struct broker_result* output)
{
  export_fn export_handle = (export_fn)real_symbol("cuMemExportToShareableHandle");
  struct context_scope scope;
  CUresult result;

  initialize_broker_result(
      output,
      allocation == NULL ? CU_MEM_HANDLE_TYPE_NONE
                         : allocation->requested_handle_type,
      allocation == NULL ? 0 : allocation->object_kind);
  if (allocation == NULL || allocation->role != DYN_VMM_OWNER || !allocation->exported ||
      all_zero(allocation->allocation_uuid, sizeof(allocation->allocation_uuid)) ||
      !allocation->application_handle_live || allocation->real_handle == 0 || allocation->detached ||
      !supported_handle_type(allocation->requested_handle_type) || phase != DYN_VMM_ACTIVE || failed ||
      forked_after_cuda || export_handle == NULL || enter_context(allocation->context, &scope) != 0)
    return CUDA_ERROR_INVALID_HANDLE;
  if (output->handle_type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    result = export_handle(&output->fd, allocation->real_handle, output->handle_type, 0);
  } else {
    result = export_handle(output->bytes, allocation->real_handle, output->handle_type, 0);
    if (result == CUDA_SUCCESS)
      output->bytes_size = sizeof(output->bytes);
  }
  if (leave_context(&scope) != 0) {
    (void)clear_broker_result(output, "cannot close owner raw export after context failure");
    fail("cannot restore context after owner raw export");
    return CUDA_ERROR_UNKNOWN;
  }
  if (result == CUDA_SUCCESS && !valid_broker_shape(output->handle_type, output->fd, output->bytes_size))
    return CUDA_ERROR_INVALID_VALUE;
  if (result == CUDA_SUCCESS && allocation->object_kind == DYN_VMM_MULTICAST) {
    output->multicast_properties = allocation->multicast_properties;
    output->has_multicast_properties = true;
  }
  return result;
}

static int
send_broker_result(int client, struct dyn_vmm_header* response, const struct broker_result* result)
{
  response->handle_type = (uint32_t)result->handle_type;
  response->object_kind = result->object_kind;
  if (response->operation == DYN_VMM_EXPORT_OWNER &&
      result->has_multicast_properties) {
    memcpy(
        response->reserved_identity, &result->multicast_properties,
        sizeof(result->multicast_properties));
  }
  response->payload_size = result->bytes_size;
  if (send_header(client, response, result->fd) != 0)
    return -1;
  return result->bytes_size == 0 || write_all(client, result->bytes, result->bytes_size) == 0 ? 0 : -1;
}

static int
export_owner(int client, const struct dyn_vmm_header* request, struct dyn_vmm_header* response, int passed_fd)
{
  struct allocation* allocation;
  struct broker_result exported;
  int result;

  initialize_broker_result(
      &exported, (CUmemAllocationHandleType)request->handle_type,
      request->object_kind);

  memcpy(response->allocation_uuid, request->allocation_uuid, sizeof(response->allocation_uuid));
  if (passed_fd >= 0 || request->payload_size != 0 || request->count != 0 || request->status != 0 ||
      request->object_id != 0 || !supported_handle_type((CUmemAllocationHandleType)request->handle_type) ||
      (request->object_kind != DYN_VMM_ALLOCATION &&
       request->object_kind != DYN_VMM_MULTICAST) ||
      !all_zero(request->message, sizeof(request->message)) ||
      !all_zero(request->reserved_identity, sizeof(request->reserved_identity)) ||
      memcmp(request->participant_id, participant_id, sizeof(request->participant_id)) != 0 ||
      all_zero(request->allocation_uuid, sizeof(request->allocation_uuid)) || failed || forked_after_cuda ||
      phase != DYN_VMM_ACTIVE) {
    set_error(response, "invalid owner export request");
    return send_header(client, response, -1);
  }
  allocation = find_object(request->allocation_uuid, DYN_VMM_OWNER);
  if (allocation == NULL ||
      memcmp(allocation->allocation_uuid, request->allocation_uuid, sizeof(allocation->allocation_uuid)) != 0 ||
      allocation->object_kind != request->object_kind ||
      allocation->requested_handle_type != (CUmemAllocationHandleType)request->handle_type ||
      export_raw_owner(allocation, &exported) != CUDA_SUCCESS) {
    (void)clear_broker_result(&exported, "cannot close failed owner raw export FD");
    set_error(response, "owner allocation is unavailable for export");
    return send_header(client, response, -1);
  }
  result = send_broker_result(client, response, &exported);
  if (clear_broker_result(&exported, "cannot close owner raw export FD") != 0)
    phase = DYN_VMM_FAILED;
  return result;
}

static int
request_owner_export(
    const uint8_t allocation_uuid[DYN_VMM_ALLOCATION_UUID_SIZE],
    const char owner_participant[DYN_VMM_PARTICIPANT_ID_SIZE], const char* owner_socket,
    CUmemAllocationHandleType handle_type, uint32_t object_kind,
    struct broker_result* exported)
{
  struct sockaddr_un address = {.sun_family = AF_UNIX};
  struct dyn_vmm_header request;
  struct dyn_vmm_header response;
  int client = -1;
  int result = -1;

  initialize_broker_result(exported, handle_type, object_kind);
  memset(&request, 0, sizeof(request));
  request.magic = DYN_VMM_MAGIC;
  request.version = DYN_VMM_VERSION;
  request.operation = DYN_VMM_EXPORT_OWNER;
  request.handle_type = (uint32_t)handle_type;
  request.object_kind = object_kind;
  memcpy(request.allocation_uuid, allocation_uuid, sizeof(request.allocation_uuid));
  snprintf(request.participant_id, sizeof(request.participant_id), "%s", owner_participant);
  snprintf(address.sun_path, sizeof(address.sun_path), "%s", owner_socket);
  client = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (client < 0 || set_socket_timeouts(client) != 0 ||
      connect(client, (const struct sockaddr*)&address, sizeof(address)) != 0 ||
      send_header(client, &request, -1) != 0 || receive_header(client, &response, &exported->fd) != 0)
    goto done;
  if (response.magic != DYN_VMM_MAGIC || response.version != DYN_VMM_VERSION ||
      response.operation != DYN_VMM_EXPORT_OWNER || response.status != 0 || response.count != 0 ||
      response.object_id != 0 ||
      response.handle_type != (uint32_t)handle_type ||
      response.object_kind != object_kind ||
      !all_zero(response.message, sizeof(response.message)) ||
      memcmp(response.allocation_uuid, allocation_uuid, sizeof(response.allocation_uuid)) != 0 ||
      !valid_participant_id(response.participant_id) ||
      memcmp(response.participant_id, owner_participant, sizeof(response.participant_id)) != 0 ||
      !valid_broker_shape(handle_type, exported->fd, response.payload_size))
    goto done;
  if (object_kind == DYN_VMM_MULTICAST) {
    if (!all_zero(
            response.reserved_identity +
                sizeof(exported->multicast_properties),
            sizeof(response.reserved_identity) -
                sizeof(exported->multicast_properties)))
      goto done;
    memcpy(
        &exported->multicast_properties, response.reserved_identity,
        sizeof(exported->multicast_properties));
    if (!valid_multicast_properties(
            &exported->multicast_properties))
      goto done;
    exported->has_multicast_properties = true;
  } else if (!all_zero(
                 response.reserved_identity,
                 sizeof(response.reserved_identity))) {
    goto done;
  }
  if (handle_type == CU_MEM_HANDLE_TYPE_FABRIC) {
    if (read_all(client, exported->bytes, sizeof(exported->bytes)) != 0)
      goto done;
    exported->bytes_size = sizeof(exported->bytes);
  }
  result = 0;
done:
  if (client >= 0 && close(client) != 0)
    result = -1;
  if (result != 0)
    (void)clear_broker_result(exported, "cannot close invalid owner broker response FD");
  return result;
}

static void
set_error(struct dyn_vmm_header* header, const char* message)
{
  header->status = -1;
  snprintf(header->message, sizeof(header->message), "%s", message);
}

static int
validate_admission(struct dyn_vmm_header* response)
{
  struct allocation* allocation;

  if (failed || forked_after_cuda) {
    set_error(
        response,
        forked_after_cuda ? "fork after CUDA initialization is unsupported" : failure);
    return -1;
  }
  if (phase != DYN_VMM_ACTIVE) {
    set_error(response, "process is not in active VMM phase");
    return -1;
  }
  for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
    if (!allocation->exported && allocation->role == DYN_VMM_OWNER)
      continue;
    if (!allocation->mapped || allocation->offset != 0 || allocation->address == 0 ||
        allocation->access_count == 0) {
      set_error(response, "managed object is not one complete offset-zero mapping with access");
      return -1;
    }
    if (allocation->object_kind == DYN_VMM_MULTICAST &&
        (!allocation->member_added || !allocation->bound ||
         allocation->multicast_properties.flags != 0 ||
         allocation->multicast_properties.size != allocation->size ||
         allocation->multicast_properties.numDevices == 0 ||
         allocation->multicast_properties.handleTypes !=
             (unsigned long long)allocation->requested_handle_type ||
         all_zero(
             allocation->backing_allocation_uuid,
             sizeof(allocation->backing_allocation_uuid)) ||
         allocation->multicast_offset != 0 ||
         allocation->memory_offset != 0 ||
         allocation->bind_size != allocation->size ||
         allocation->bind_flags != 0)) {
      set_error(response, "managed multicast object is incomplete");
      return -1;
    }
    if (allocation->access_count != 1 ||
        allocation->access[0].location.type != CU_MEM_LOCATION_TYPE_DEVICE ||
        allocation->access[0].location.id != allocation->device_ordinal ||
        allocation->access[0].flags != CU_MEM_ACCESS_FLAGS_PROT_READWRITE) {
      set_error(
          response,
          "unsupported managed access: requires one read/write DEVICE "
          "descriptor on the allocation GPU");
      return -1;
    }
  }
  return 0;
}

static int
inspect(int client, struct dyn_vmm_header* response)
{
  struct allocation* allocation;
  size_t payload_size = 0;
  char* payload;
  char* cursor;

  if (validate_admission(response) != 0)
    return send_header(client, response, -1);
  for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
    if (!allocation->exported && allocation->role == DYN_VMM_OWNER)
      continue;
    response->count++;
    payload_size += sizeof(struct dyn_vmm_record) +
        (allocation->role == DYN_VMM_OWNER
             ? (allocation->object_kind == DYN_VMM_MULTICAST
                    ? sizeof(allocation->multicast_properties)
                    : sizeof(allocation->properties))
             : 0) +
        allocation->access_count * sizeof(*allocation->access) +
        (allocation->object_kind == DYN_VMM_MULTICAST
             ? sizeof(struct dyn_vmm_multicast_record)
             : 0);
  }
  payload = malloc(payload_size == 0 ? 1 : payload_size);
  if (payload == NULL) {
    set_error(response, "out of memory encoding VMM metadata");
    return send_header(client, response, -1);
  }
  cursor = payload;
  for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
    struct dyn_vmm_record record;
    size_t properties_size;
    size_t access_size;

    if (!allocation->exported && allocation->role == DYN_VMM_OWNER)
      continue;
    properties_size =
        allocation->role == DYN_VMM_OWNER
        ? (allocation->object_kind == DYN_VMM_MULTICAST
               ? sizeof(allocation->multicast_properties)
               : sizeof(allocation->properties))
        : 0;
    access_size = allocation->access_count * sizeof(*allocation->access);
    memset(&record, 0, sizeof(record));
    memcpy(record.allocation_uuid, allocation->allocation_uuid, sizeof(record.allocation_uuid));
    record.address = allocation->address;
    record.size = allocation->size;
    record.offset = allocation->offset;
    record.role = allocation->role;
    record.object_kind = allocation->object_kind;
    record.requested_handle_type =
        (uint32_t)allocation->requested_handle_type;
    record.flags = allocation->application_handle_live
        ? DYN_VMM_APPLICATION_HANDLE_LIVE
        : 0;
    record.device_ordinal = allocation->device_ordinal;
    record.properties_size = (uint32_t)properties_size;
    record.access_count = (uint32_t)allocation->access_count;
    record.access_size = sizeof(*allocation->access);
    memcpy(record.gpu_uuid, allocation->gpu_uuid.bytes, sizeof(record.gpu_uuid));
    memcpy(cursor, &record, sizeof(record));
    cursor += sizeof(record);
    if (properties_size != 0) {
      memcpy(
          cursor,
          allocation->object_kind == DYN_VMM_MULTICAST
              ? (void*)&allocation->multicast_properties
              : (void*)&allocation->properties,
          properties_size);
      cursor += properties_size;
    }
    memcpy(cursor, allocation->access, access_size);
    cursor += access_size;
    if (allocation->object_kind == DYN_VMM_MULTICAST) {
      struct dyn_vmm_multicast_record multicast;

      memset(&multicast, 0, sizeof(multicast));
      memcpy(
          multicast.backing_allocation_uuid,
          allocation->backing_allocation_uuid,
          sizeof(multicast.backing_allocation_uuid));
      multicast.multicast_offset = allocation->multicast_offset;
      multicast.memory_offset = allocation->memory_offset;
      multicast.bind_size = allocation->bind_size;
      multicast.bind_flags = allocation->bind_flags;
      multicast.object_flags = allocation->multicast_properties.flags;
      multicast.object_handle_types =
          allocation->multicast_properties.handleTypes;
      multicast.object_size = allocation->multicast_properties.size;
      multicast.num_devices =
          allocation->multicast_properties.numDevices;
      multicast.backing_role = allocation->backing_role;
      multicast.bind_api = allocation->bind_api;
      memcpy(cursor, &multicast, sizeof(multicast));
      cursor += sizeof(multicast);
    }
  }
  response->payload_size = payload_size;
  if (send_header(client, response, -1) != 0 ||
      write_all(client, payload, payload_size) != 0) {
    free(payload);
    return -1;
  }
  free(payload);
  return 0;
}

static int
read_owner(int client, const struct dyn_vmm_header* request, struct dyn_vmm_header* response)
{
  typedef CUresult(CUDAAPI * copy_type)(void*, CUdeviceptr, size_t);
  copy_type copy = (copy_type)real_symbol("cuMemcpyDtoH_v2");
  struct allocation* allocation = find_object(request->allocation_uuid, DYN_VMM_OWNER);
  struct context_scope scope;
  void* buffer;
  size_t offset;
  int result = 0;

  if (allocation == NULL ||
      allocation->object_kind != DYN_VMM_ALLOCATION ||
      !allocation->exported || !allocation->mapped ||
      allocation->detached) {
    set_error(response, "owner allocation is unavailable");
    return send_header(client, response, -1);
  }
  buffer = malloc(COPY_CHUNK);
  if (copy == NULL || buffer == NULL || enter_context(allocation->context, &scope) != 0) {
    free(buffer);
    set_error(response, "cannot enter owner context for byte capture");
    return send_header(client, response, -1);
  }
  response->payload_size = allocation->size;
  if (send_header(client, response, -1) != 0)
    result = -1;
  for (offset = 0; result == 0 && offset < allocation->size;) {
    size_t chunk = allocation->size - offset;
    if (chunk > COPY_CHUNK)
      chunk = COPY_CHUNK;
    if (copy(buffer, allocation->address + offset, chunk) != CUDA_SUCCESS ||
        write_all(client, buffer, chunk) != 0)
      result = -1;
    offset += chunk;
  }
  if (leave_context(&scope) != 0)
    result = -1;
  free(buffer);
  return result;
}

static int
detach(
    int client, const struct dyn_vmm_header* request, struct dyn_vmm_header* response,
    uint32_t role)
{
  unmap_fn unmap = (unmap_fn)real_symbol("cuMemUnmap");
  release_fn release = (release_fn)real_symbol("cuMemRelease");
  typedef CUresult(CUDAAPI * unbind_type)(
      CUmemGenericAllocationHandle, CUdevice, size_t, size_t);
  unbind_type unbind =
      (unbind_type)real_symbol("cuMulticastUnbind");
  struct allocation* allocation = find_object(request->allocation_uuid, role);
  struct context_scope scope;

  if (allocation == NULL || !allocation->mapped || allocation->detached) {
    set_error(response, "allocation is unavailable for detach");
    return send_header(client, response, -1);
  }
  if (allocation->object_id != 0 && allocation->object_id != request->object_id) {
    set_error(response, "allocation already belongs to another logical object");
    return send_header(client, response, -1);
  }
  if (unmap == NULL || release == NULL ||
      (allocation->object_kind == DYN_VMM_MULTICAST &&
       (unbind == NULL || !allocation->bound)) ||
      enter_context(allocation->context, &scope) != 0 ||
      (allocation->object_kind == DYN_VMM_MULTICAST &&
       unbind(
           allocation->real_handle, allocation->member_device,
           allocation->multicast_offset, allocation->bind_size) !=
           CUDA_SUCCESS) ||
      unmap(allocation->address, allocation->size) != CUDA_SUCCESS ||
      (allocation->real_handle != 0 &&
       release(allocation->real_handle) != CUDA_SUCCESS) ||
      leave_context(&scope) != 0) {
    fail("CUDA VMM detach failed; process must not resume");
    phase = DYN_VMM_FAILED;
    set_error(response, failure);
    return send_header(client, response, -1);
  }
  allocation->mapped = false;
  if (allocation->object_kind == DYN_VMM_MULTICAST)
    allocation->bound = false;
  allocation->detached = true;
  allocation->real_handle = 0;
  allocation->object_id = request->object_id;
  phase = DYN_VMM_DETACHED;
  return send_header(client, response, -1);
}

static int
decode_record(
    const struct dyn_vmm_header* request, char* payload, struct dyn_vmm_record* record,
    CUmemAllocationProp** properties, CUmemAccessDesc** access,
    size_t* metadata_size)
{
  uint64_t decoded_size;

  if (request->payload_size < sizeof(*record))
    return -1;
  memcpy(record, payload, sizeof(*record));
  if (record->access_size != sizeof(**access))
    return -1;
  decoded_size = sizeof(*record) + record->properties_size +
      (uint64_t)record->access_count * record->access_size;
  if (decoded_size > request->payload_size || decoded_size > SIZE_MAX)
    return -1;
  *properties = record->properties_size == 0
      ? NULL
      : (CUmemAllocationProp*)(payload + sizeof(*record));
  *access = (CUmemAccessDesc*)(
      payload + sizeof(*record) + record->properties_size);
  *metadata_size = (size_t)decoded_size;
  if ((record->object_kind != DYN_VMM_ALLOCATION &&
       record->object_kind != DYN_VMM_MULTICAST) ||
      !supported_handle_type((CUmemAllocationHandleType)record->requested_handle_type) ||
      (record->flags &
       ~(DYN_VMM_APPLICATION_HANDLE_LIVE |
         DYN_VMM_RETAIN_RESTORE_HANDLE)) != 0)
    return -1;
  return 0;
}

static int
reconcile_restore_metadata(
    struct allocation* allocation, const struct dyn_vmm_record* record, CUmemAllocationProp* properties,
    CUmemAccessDesc* access)
{
  size_t index;

  if (record->access_count != allocation->access_count)
    return -1;
  if (properties != NULL) {
    if (properties->location.type != CU_MEM_LOCATION_TYPE_DEVICE ||
        properties->location.id != allocation->source_device_ordinal)
      return -1;
    properties->location.id = allocation->device_ordinal;
    if (memcmp(properties, &allocation->properties, sizeof(*properties)) != 0)
      return -1;
  }
  for (index = 0; index < record->access_count; index++) {
    if (access[index].location.type == CU_MEM_LOCATION_TYPE_DEVICE) {
      if (access[index].location.id != allocation->source_device_ordinal)
        return -1;
      access[index].location.id = allocation->device_ordinal;
    }
  }
  return memcmp(access, allocation->access, record->access_count * sizeof(*access)) == 0 ? 0 : -1;
}

#ifdef DYN_VMM_TESTING
static atomic_bool fail_next_response_send;

int
dyn_vmm_test_fail_next_response_send(void)
{
  atomic_store(&fail_next_response_send, true);
  return 0;
}

static bool
test_response_send_failure(void)
{
  return atomic_exchange(&fail_next_response_send, false);
}

static bool
test_failure(const char* stage)
{
  const char* configured = getenv("DYN_SNAPSHOT_CUDA_VMM_FAIL_STAGE");

  return configured != NULL && strcmp(configured, stage) == 0;
}
#else
static bool
test_response_send_failure(void)
{
  return false;
}

static bool
test_failure(const char* stage)
{
  (void)stage;
  return false;
}
#endif

static int
restore_owner(
    int client, const struct dyn_vmm_header* request, struct dyn_vmm_header* response, char* payload,
    int passed_fd)
{
  typedef CUresult(CUDAAPI * copy_type)(CUdeviceptr, const void*, size_t);
  copy_type copy = (copy_type)real_symbol("cuMemcpyHtoD_v2");
  create_fn create = (create_fn)real_symbol("cuMemCreate");
  map_fn map = (map_fn)real_symbol("cuMemMap");
  unmap_fn unmap = (unmap_fn)real_symbol("cuMemUnmap");
  access_fn set_access = (access_fn)real_symbol("cuMemSetAccess");
  export_fn export_handle = (export_fn)real_symbol("cuMemExportToShareableHandle");
  release_fn release = (release_fn)real_symbol("cuMemRelease");
  struct dyn_vmm_record record;
  CUmemAllocationProp* properties;
  CUmemAccessDesc* access;
  const char* contents;
  size_t metadata_size;
  struct allocation* allocation;
  struct context_scope scope;
  CUmemAccessDesc temporary_access;
  CUmemGenericAllocationHandle handle = 0;
  bool context_entered = false;
  bool owns_handle = false;
  bool mapping_installed = false;
  bool temporary_access_installed = false;
  bool logical_rebound = false;
  bool handle_released = false;
  struct context_scope cleanup_scope;
  bool cleanup_context_entered = false;
  struct broker_result exported;
  const char* primary_error = "owner VMM restore failed; process must not resume";

  initialize_broker_result(
      &exported, (CUmemAllocationHandleType)request->handle_type,
      request->object_kind);

  if (passed_fd >= 0 || request->object_kind != DYN_VMM_ALLOCATION ||
      decode_record(request, payload, &record, &properties, &access, &metadata_size) != 0 || properties == NULL ||
      record.properties_size != sizeof(*properties) || record.role != DYN_VMM_OWNER || record.offset != 0 ||
      request->handle_type != record.requested_handle_type ||
      properties->requestedHandleTypes != (CUmemAllocationHandleType)record.requested_handle_type ||
      record.size != request->payload_size - metadata_size) {
    set_error(response, "invalid owner restore payload");
    return send_header(client, response, -1);
  }
  contents = payload + metadata_size;
  allocation = find_logical_object(record.object_id, DYN_VMM_OWNER);
  if (allocation == NULL || !allocation->detached ||
      allocation->address != (CUdeviceptr)record.address ||
      allocation->size != (size_t)record.size ||
      allocation->object_kind != record.object_kind ||
      allocation->requested_handle_type !=
          (CUmemAllocationHandleType)record.requested_handle_type ||
      allocation->application_handle_live !=
          ((record.flags & DYN_VMM_APPLICATION_HANDLE_LIVE) != 0) ||
      allocation->device_ordinal != record.device_ordinal ||
      memcmp(
          allocation->gpu_uuid.bytes, record.gpu_uuid,
          sizeof(record.gpu_uuid)) != 0 ||
      reconcile_restore_metadata(allocation, &record, properties, access) != 0) {
    set_error(response, "owner restore metadata does not match detached process state");
    return send_header(client, response, -1);
  }
  memset(&temporary_access, 0, sizeof(temporary_access));
  temporary_access.location = properties->location;
  temporary_access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (create == NULL || map == NULL || unmap == NULL || set_access == NULL ||
      export_handle == NULL || release == NULL || copy == NULL ||
      enter_context(allocation->context, &scope) != 0)
    goto failed;
  context_entered = true;
  if (create(&handle, (size_t)record.size, properties, 0) != CUDA_SUCCESS)
    goto failed;
  owns_handle = true;
  if (test_failure("owner-create"))
    goto failed;
  if (map((CUdeviceptr)record.address, (size_t)record.size, 0, handle, 0) !=
      CUDA_SUCCESS)
    goto failed;
  mapping_installed = true;
  allocation->mapped = true;
  allocation->detached = false;
  if (test_failure("owner-map"))
    goto failed;
  if (set_access(
          (CUdeviceptr)record.address, (size_t)record.size,
          &temporary_access, 1) != CUDA_SUCCESS)
    goto failed;
  temporary_access_installed = true;
  if (test_failure("owner-access"))
    goto failed;
  if (copy((CUdeviceptr)record.address, contents, (size_t)record.size) !=
      CUDA_SUCCESS)
    goto failed;
  if (test_failure("owner-copy"))
    goto failed;
  temporary_access.flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
  if (set_access(
          (CUdeviceptr)record.address, (size_t)record.size,
          &temporary_access, 1) != CUDA_SUCCESS ||
      set_access(
          (CUdeviceptr)record.address, (size_t)record.size, access,
          record.access_count) != CUDA_SUCCESS)
    goto failed;
  temporary_access_installed = false;
  if (record.requested_handle_type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    if (export_handle(&exported.fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) != CUDA_SUCCESS ||
        exported.fd < 0)
      goto failed;
  } else {
    if (export_handle(exported.bytes, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0) != CUDA_SUCCESS)
      goto failed;
    exported.bytes_size = sizeof(exported.bytes);
  }
  if (test_failure("owner-export"))
    goto failed;
  if (allocation->application_handle_live ||
      (record.flags & DYN_VMM_RETAIN_RESTORE_HANDLE) != 0) {
    allocation->real_handle = handle;
    logical_rebound = true;
    allocation->temporary_restore_handle =
        !allocation->application_handle_live;
    owns_handle = false;
  } else {
    if (release(handle) != CUDA_SUCCESS)
      goto failed;
    owns_handle = false;
    handle_released = true;
  }
  if (test_failure("owner-rebind") || leave_context(&scope) != 0)
    goto failed;
  context_entered = false;
  if (send_broker_result(client, response, &exported) != 0)
    goto failed;
  if (clear_broker_result(&exported, "cannot close owner restore export FD") != 0) {
    phase = DYN_VMM_FAILED;
    return -1;
  }
  phase = DYN_VMM_RESTORED;
  return 0;

failed:
  fail(primary_error);
  phase = DYN_VMM_FAILED;
  if (!context_entered && mapping_installed) {
    if (enter_context(allocation->context, &cleanup_scope) != 0)
      fail_cleanup("owner restore cleanup could not enter allocation context");
    else
      cleanup_context_entered = true;
  }
  if (logical_rebound) {
    allocation->real_handle = 0;
    logical_rebound = false;
    owns_handle = true;
  }
  if (mapping_installed && temporary_access_installed) {
    temporary_access.flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
    if (set_access == NULL ||
        set_access(
            (CUdeviceptr)record.address, (size_t)record.size,
            &temporary_access, 1) != CUDA_SUCCESS)
      fail_cleanup("owner restore cleanup could not remove temporary access");
    else
      temporary_access_installed = false;
  }
  if (mapping_installed) {
    if (unmap == NULL ||
        unmap((CUdeviceptr)record.address, (size_t)record.size) != CUDA_SUCCESS)
      fail_cleanup("owner restore cleanup could not unmap fresh mapping");
    else {
      allocation->mapped = false;
      allocation->detached = true;
      mapping_installed = false;
      temporary_access_installed = false;
    }
  }
  if (owns_handle && !handle_released) {
    if (release == NULL || release(handle) != CUDA_SUCCESS)
      fail_cleanup("owner restore cleanup could not release fresh handle");
    owns_handle = false;
    handle_released = true;
  }
  if (clear_broker_result(&exported, "owner restore cleanup could not close export FD") != 0)
    phase = DYN_VMM_FAILED;
  if (context_entered && leave_context(&scope) != 0)
    fail_cleanup("owner restore cleanup could not restore prior CUDA context");
  if (cleanup_context_entered && leave_context(&cleanup_scope) != 0)
    fail_cleanup("owner restore cleanup could not restore prior CUDA context");
  set_error(response, primary_error);
  return send_header(client, response, -1);
}

static int
restore_importer(
    int client, const struct dyn_vmm_header* request, struct dyn_vmm_header* response,
    char* payload, int* imported_fd)
{
  import_fn import_handle = (import_fn)real_symbol("cuMemImportFromShareableHandle");
  map_fn map = (map_fn)real_symbol("cuMemMap");
  unmap_fn unmap = (unmap_fn)real_symbol("cuMemUnmap");
  access_fn set_access = (access_fn)real_symbol("cuMemSetAccess");
  release_fn release = (release_fn)real_symbol("cuMemRelease");
  struct dyn_vmm_record record;
  CUmemAllocationProp* properties;
  CUmemAccessDesc* access;
  char* contents;
  size_t metadata_size;
  struct allocation* allocation;
  struct context_scope scope;
  CUmemGenericAllocationHandle handle = 0;
  bool context_entered = false;
  bool owns_handle = false;
  bool mapping_installed = false;
  bool logical_rebound = false;
  bool handle_released = false;
  struct context_scope cleanup_scope;
  bool cleanup_context_entered = false;
  void* import_value;
  size_t broker_size;
  const char* primary_error = "importer VMM restore failed; process must not resume";

  if (request->object_kind != DYN_VMM_ALLOCATION ||
      decode_record(request, payload, &record, &properties, &access, &metadata_size) != 0) {
    set_error(response, "invalid importer restore payload");
    return send_header(client, response, -1);
  }
  contents = payload + metadata_size;
  broker_size = (size_t)request->payload_size - metadata_size;
  if (properties != NULL || record.role != DYN_VMM_IMPORTER || record.offset != 0 ||
      request->handle_type != record.requested_handle_type ||
      (record.requested_handle_type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
       (broker_size != 0 || *imported_fd < 0)) ||
      (record.requested_handle_type == CU_MEM_HANDLE_TYPE_FABRIC &&
       (broker_size != DYN_VMM_FABRIC_HANDLE_SIZE || *imported_fd >= 0))) {
    set_error(response, "invalid importer restore metadata");
    return send_header(client, response, -1);
  }
  import_value = record.requested_handle_type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
                     ? (void*)(uintptr_t)*imported_fd
                     : contents;
  allocation = find_logical_object(record.object_id, DYN_VMM_IMPORTER);
  if (allocation == NULL || !allocation->detached ||
      allocation->address != (CUdeviceptr)record.address ||
      allocation->size != (size_t)record.size ||
      allocation->object_kind != record.object_kind ||
      allocation->requested_handle_type !=
          (CUmemAllocationHandleType)record.requested_handle_type ||
      allocation->application_handle_live !=
          ((record.flags & DYN_VMM_APPLICATION_HANDLE_LIVE) != 0) ||
      allocation->device_ordinal != record.device_ordinal ||
      memcmp(
          allocation->gpu_uuid.bytes, record.gpu_uuid,
          sizeof(record.gpu_uuid)) != 0 ||
      reconcile_restore_metadata(allocation, &record, properties, access) != 0)
    goto failed;
  if (import_handle == NULL || map == NULL || unmap == NULL ||
      set_access == NULL || release == NULL ||
      enter_context(allocation->context, &scope) != 0)
    goto failed;
  context_entered = true;
  if (import_handle(&handle, import_value, (CUmemAllocationHandleType)record.requested_handle_type) !=
      CUDA_SUCCESS)
    goto failed;
  owns_handle = true;
  if (record.requested_handle_type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR &&
      close_owned_fd(imported_fd, "importer restore could not close received broker FD") != 0)
    goto failed;
  if (record.requested_handle_type == CU_MEM_HANDLE_TYPE_FABRIC)
    explicit_bzero(contents, DYN_VMM_FABRIC_HANDLE_SIZE);
  if (test_failure("importer-import"))
    goto failed;
  if (map((CUdeviceptr)record.address, (size_t)record.size, 0, handle, 0) !=
      CUDA_SUCCESS)
    goto failed;
  mapping_installed = true;
  allocation->mapped = true;
  allocation->detached = false;
  if (test_failure("importer-map"))
    goto failed;
  if (set_access(
          (CUdeviceptr)record.address, (size_t)record.size, access,
          record.access_count) != CUDA_SUCCESS)
    goto failed;
  if (test_failure("importer-access"))
    goto failed;
  if (allocation->application_handle_live ||
      (record.flags & DYN_VMM_RETAIN_RESTORE_HANDLE) != 0) {
    allocation->real_handle = handle;
    logical_rebound = true;
    allocation->temporary_restore_handle =
        !allocation->application_handle_live;
    owns_handle = false;
  } else {
    if (release(handle) != CUDA_SUCCESS)
      goto failed;
    owns_handle = false;
    handle_released = true;
  }
  if (test_failure("importer-rebind") || leave_context(&scope) != 0)
    goto failed;
  context_entered = false;
  if (send_header(client, response, -1) != 0)
    goto failed;
  phase = DYN_VMM_RESTORED;
  return 0;

failed:
  fail(primary_error);
  phase = DYN_VMM_FAILED;
  if (!context_entered && mapping_installed) {
    if (enter_context(allocation->context, &cleanup_scope) != 0)
      fail_cleanup("importer restore cleanup could not enter allocation context");
    else
      cleanup_context_entered = true;
  }
  if (logical_rebound) {
    allocation->real_handle = 0;
    logical_rebound = false;
    owns_handle = true;
  }
  if (mapping_installed) {
    if (unmap == NULL ||
        unmap((CUdeviceptr)record.address, (size_t)record.size) != CUDA_SUCCESS)
      fail_cleanup("importer restore cleanup could not unmap fresh mapping");
    else {
      allocation->mapped = false;
      allocation->detached = true;
      mapping_installed = false;
    }
  }
  if (owns_handle && !handle_released) {
    if (release == NULL || release(handle) != CUDA_SUCCESS)
      fail_cleanup("importer restore cleanup could not release fresh handle");
    owns_handle = false;
    handle_released = true;
  }
  if (close_owned_fd(
          imported_fd, "importer restore cleanup could not close broker FD") != 0)
    phase = DYN_VMM_FAILED;
  if (record.requested_handle_type == CU_MEM_HANDLE_TYPE_FABRIC)
    explicit_bzero(contents, broker_size);
  if (context_entered && leave_context(&scope) != 0)
    fail_cleanup("importer restore cleanup could not restore prior CUDA context");
  if (cleanup_context_entered && leave_context(&cleanup_scope) != 0)
    fail_cleanup("importer restore cleanup could not restore prior CUDA context");
  set_error(response, primary_error);
  return send_header(client, response, -1);
}

static int
cleanup_multicast_restore(void)
{
  typedef CUresult(CUDAAPI * unbind_type)(
      CUmemGenericAllocationHandle, CUdevice, size_t, size_t);
  unbind_type unbind =
      (unbind_type)real_symbol("cuMulticastUnbind");
  unmap_fn unmap = (unmap_fn)real_symbol("cuMemUnmap");
  release_fn release = (release_fn)real_symbol("cuMemRelease");
  struct allocation* allocation;
  int result = 0;

  for (allocation = allocations; allocation != NULL;
       allocation = allocation->next) {
    struct context_scope scope;
    bool entered = false;

    if (allocation->object_kind == DYN_VMM_MULTICAST &&
        allocation->real_handle != 0) {
      if (enter_context(allocation->context, &scope) == 0)
        entered = true;
      else {
        fail_cleanup("cannot enter multicast cleanup context");
        result = -1;
      }
      if (entered && allocation->bound &&
          (unbind == NULL ||
           unbind(
               allocation->real_handle, allocation->member_device,
               allocation->multicast_offset, allocation->bind_size) !=
               CUDA_SUCCESS)) {
        fail_cleanup("cannot unbind failed multicast restore");
        result = -1;
      } else if (entered) {
        allocation->bound = false;
      }
      if (entered && allocation->mapped &&
          (unmap == NULL ||
           unmap(allocation->address, allocation->size) != CUDA_SUCCESS)) {
        fail_cleanup("cannot unmap failed multicast restore");
        result = -1;
      } else if (entered) {
        allocation->mapped = false;
      }
      if (entered &&
          (release == NULL ||
           release(allocation->real_handle) != CUDA_SUCCESS)) {
        fail_cleanup("cannot release failed multicast restore");
        result = -1;
      } else if (entered) {
        allocation->real_handle = 0;
        allocation->member_added = false;
        allocation->bound = false;
        allocation->mapped = false;
        allocation->backing_object_id = 0;
        allocation->temporary_restore_handle = false;
      }
      allocation->detached = true;
      if (entered && leave_context(&scope) != 0) {
        fail_cleanup("cannot leave multicast cleanup context");
        result = -1;
      }
    }
    if (allocation->object_kind == DYN_VMM_ALLOCATION &&
        allocation->temporary_restore_handle &&
        allocation->real_handle != 0) {
      if (release == NULL ||
          release(allocation->real_handle) != CUDA_SUCCESS) {
        fail_cleanup("cannot release temporary backing handle");
        result = -1;
      } else {
        allocation->real_handle = 0;
        allocation->temporary_restore_handle = false;
      }
    }
  }
  return result;
}

static int
restore_multicast_binding(
    int client, const struct dyn_vmm_header* request,
    struct dyn_vmm_header* response, char* payload)
{
  typedef CUresult(CUDAAPI * legacy_bind_type)(
      CUmemGenericAllocationHandle, size_t,
      CUmemGenericAllocationHandle, size_t, size_t,
      unsigned long long);
#if CUDA_VERSION >= 13010
  typedef CUresult(CUDAAPI * v2_bind_type)(
      CUmemGenericAllocationHandle, CUdevice, size_t,
      CUmemGenericAllocationHandle, size_t, size_t,
      unsigned long long);
#endif
  legacy_bind_type bind =
      (legacy_bind_type)real_symbol("cuMulticastBindMem");
#if CUDA_VERSION >= 13010
  v2_bind_type bind_v2 =
      (v2_bind_type)real_symbol("cuMulticastBindMem_v2");
#endif
  map_fn map = (map_fn)real_symbol("cuMemMap");
  unmap_fn unmap = (unmap_fn)real_symbol("cuMemUnmap");
  access_fn set_access = (access_fn)real_symbol("cuMemSetAccess");
  typedef CUresult(CUDAAPI * unbind_type)(
      CUmemGenericAllocationHandle, CUdevice, size_t, size_t);
  unbind_type unbind =
      (unbind_type)real_symbol("cuMulticastUnbind");
  struct dyn_vmm_record record;
  struct dyn_vmm_multicast_record* multicast;
  CUmulticastObjectProp* properties;
  CUmemAccessDesc* access;
  struct allocation* allocation;
  struct allocation* backing;
  struct context_scope scope;
  size_t metadata_size;
  bool entered = false;
  bool bound = false;
  bool mapped = false;
  CUresult result;
  const char* primary_error =
      "multicast binding restore failed; process must not resume";

  if (request->object_kind != DYN_VMM_MULTICAST ||
      decode_multicast_record(
          request, payload, &record, &properties, &access, &multicast,
          &metadata_size) != 0 ||
      metadata_size != request->payload_size ||
      request->object_id != record.object_id ||
      request->handle_type != record.requested_handle_type) {
    set_error(response, "invalid multicast binding restore payload");
    return send_header(client, response, -1);
  }
  allocation = find_logical_resource(
      record.object_id, record.role, DYN_VMM_MULTICAST);
  backing = find_logical_resource(
      multicast->backing_object_id, multicast->backing_role,
      DYN_VMM_ALLOCATION);
  if (allocation == NULL || backing == NULL ||
      allocation->real_handle == 0 || backing->real_handle == 0 ||
      !allocation->member_added || allocation->bound ||
      allocation->mapped || allocation->detached == false ||
      backing->detached || !backing->mapped ||
      backing->requested_handle_type !=
          allocation->requested_handle_type ||
      backing->size != allocation->size ||
      backing->device_ordinal != allocation->device_ordinal ||
      reconcile_multicast_metadata(
          allocation, &record, properties, access, multicast,
          record.role == DYN_VMM_OWNER) != 0 ||
      map == NULL || unmap == NULL || set_access == NULL ||
      unbind == NULL ||
      enter_context(allocation->context, &scope) != 0)
    goto failed;
  entered = true;
  if (multicast->bind_api == DYN_VMM_MULTICAST_BIND_MEM) {
    result = bind != NULL
        ? bind(
              allocation->real_handle, multicast->multicast_offset,
              backing->real_handle, multicast->memory_offset,
              multicast->bind_size, multicast->bind_flags)
        : CUDA_ERROR_NOT_SUPPORTED;
  } else {
#if CUDA_VERSION >= 13010
    result = bind_v2 != NULL
        ? bind_v2(
              allocation->real_handle, allocation->member_device,
              multicast->multicast_offset, backing->real_handle,
              multicast->memory_offset, multicast->bind_size,
              multicast->bind_flags)
        : CUDA_ERROR_NOT_SUPPORTED;
#else
    result = CUDA_ERROR_NOT_SUPPORTED;
#endif
  }
  if (result != CUDA_SUCCESS)
    goto failed;
  bound = true;
  if (test_failure("multicast-bind"))
    goto failed;
  if (map(
          (CUdeviceptr)record.address, (size_t)record.size, 0,
          allocation->real_handle, 0) != CUDA_SUCCESS)
    goto failed;
  mapped = true;
  if (set_access(
          (CUdeviceptr)record.address, (size_t)record.size, access,
          record.access_count) != CUDA_SUCCESS)
    goto failed;
  allocation->backing_object_id = multicast->backing_object_id;
  allocation->bound = true;
  allocation->mapped = true;
  allocation->detached = false;
  allocation->temporary_restore_handle =
      !allocation->application_handle_live;
  if (leave_context(&scope) != 0)
    goto failed;
  entered = false;
  if (send_header(client, response, -1) != 0)
    goto failed;
  return 0;

failed:
  fail(primary_error);
  phase = DYN_VMM_FAILED;
  if (entered && bound) {
    if (unbind(
            allocation->real_handle, allocation->member_device,
            multicast->multicast_offset, multicast->bind_size) !=
        CUDA_SUCCESS)
      fail_cleanup("cannot unbind partial multicast replay");
    else {
      bound = false;
      allocation->bound = false;
    }
  }
  if (entered && mapped) {
    if (unmap((CUdeviceptr)record.address, (size_t)record.size) !=
        CUDA_SUCCESS)
      fail_cleanup("cannot unmap partial multicast replay");
    else {
      mapped = false;
      allocation->mapped = false;
    }
  }
  if (entered && leave_context(&scope) != 0)
    fail_cleanup("cannot restore context after multicast bind failure");
  (void)cleanup_multicast_restore();
  set_error(response, primary_error);
  return send_header(client, response, -1);
}

static int
abort_restore(int client, struct dyn_vmm_header* response)
{
  fail("CUDA VMM multicast restore aborted; process must not resume");
  phase = DYN_VMM_FAILED;
  if (cleanup_multicast_restore() != 0)
    set_error(response, failure);
  return send_header(client, response, -1);
}

static int
finalize_restore(int client, struct dyn_vmm_header* response)
{
  release_fn release = (release_fn)real_symbol("cuMemRelease");
  struct allocation* allocation;

  if (failed || phase == DYN_VMM_FAILED || release == NULL) {
    set_error(
        response,
        failure[0] != '\0' ? failure : "CUDA VMM shim is unhealthy");
    return send_header(client, response, -1);
  }
  for (allocation = allocations; allocation != NULL;
       allocation = allocation->next) {
    if (!allocation->temporary_restore_handle)
      continue;
    if ((allocation->object_kind != DYN_VMM_ALLOCATION &&
         allocation->object_kind != DYN_VMM_MULTICAST) ||
        allocation->real_handle == 0 ||
        release(allocation->real_handle) != CUDA_SUCCESS) {
      fail("cannot finalize temporary CUDA VMM backing handle");
      phase = DYN_VMM_FAILED;
      (void)cleanup_multicast_restore();
      set_error(response, failure);
      return send_header(client, response, -1);
    }
    allocation->real_handle = 0;
    allocation->temporary_restore_handle = false;
  }
  phase = DYN_VMM_RESTORED;
  return send_header(client, response, -1);
}

static bool
valid_request_shape(const struct dyn_vmm_header* request, int passed_fd)
{
  const uint64_t owner_metadata_size =
      sizeof(struct dyn_vmm_record) + sizeof(CUmemAllocationProp) + sizeof(CUmemAccessDesc);
  const uint64_t importer_metadata_size =
      sizeof(struct dyn_vmm_record) + sizeof(CUmemAccessDesc);
  const uint64_t multicast_owner_metadata_size =
      sizeof(struct dyn_vmm_record) + sizeof(CUmulticastObjectProp) +
      sizeof(CUmemAccessDesc) + sizeof(struct dyn_vmm_multicast_record);
  const uint64_t multicast_importer_metadata_size =
      sizeof(struct dyn_vmm_record) + sizeof(CUmemAccessDesc) +
      sizeof(struct dyn_vmm_multicast_record);
  bool bootstrap_identify =
      request->operation == DYN_VMM_IDENTIFY &&
      all_zero(request->participant_id, sizeof(request->participant_id));

  if (request->magic != DYN_VMM_MAGIC || request->version != DYN_VMM_VERSION ||
      request->status != 0 ||
      (request->operation != DYN_VMM_SET_PLACEMENT && request->count != 0) ||
      !all_zero(request->message, sizeof(request->message)) ||
      !all_zero(request->reserved_identity, sizeof(request->reserved_identity)) ||
      (!bootstrap_identify &&
       memcmp(request->participant_id, participant_id, sizeof(request->participant_id)) != 0))
    return false;

  switch (request->operation) {
    case DYN_VMM_IDENTIFY:
      return passed_fd < 0 && request->payload_size == 0 && request->handle_type == 0 &&
          request->object_kind == 0 &&
          all_zero(request->allocation_uuid, sizeof(request->allocation_uuid)) && request->object_id == 0;
    case DYN_VMM_INSPECT:
      return !bootstrap_identify && passed_fd < 0 && request->payload_size == 0 &&
          request->handle_type == 0 && request->object_kind == 0 &&
          all_zero(request->allocation_uuid, sizeof(request->allocation_uuid)) &&
          request->object_id == 0;
    case DYN_VMM_SET_PLACEMENT:
      return !bootstrap_identify && passed_fd < 0 && request->handle_type == 0 && request->object_kind == 0 &&
          all_zero(request->allocation_uuid, sizeof(request->allocation_uuid)) && request->object_id == 0 &&
          request->payload_size <= SIZE_MAX &&
          request->payload_size <= DYN_VMM_MAXIMUM_ALLOCATION_SIZE &&
          request->payload_size % sizeof(struct dyn_vmm_placement) == 0 &&
          request->payload_size / sizeof(struct dyn_vmm_placement) == request->count;
    case DYN_VMM_READ_OWNER:
    case DYN_VMM_DETACH_IMPORTS:
    case DYN_VMM_DETACH_OWNERS:
      return !bootstrap_identify && passed_fd < 0 && request->payload_size == 0 &&
          request->handle_type == 0 && request->object_kind == 0 &&
          !all_zero(request->allocation_uuid, sizeof(request->allocation_uuid)) &&
          request->object_id != 0;
    case DYN_VMM_RESTORE_OWNER:
      return !bootstrap_identify && passed_fd < 0 &&
          supported_handle_type((CUmemAllocationHandleType)request->handle_type) &&
          all_zero(request->allocation_uuid, sizeof(request->allocation_uuid)) && request->object_id != 0 &&
          ((request->object_kind == DYN_VMM_ALLOCATION &&
            request->payload_size >= owner_metadata_size &&
            request->payload_size <=
                owner_metadata_size + DYN_VMM_MAXIMUM_ALLOCATION_SIZE) ||
           (request->object_kind == DYN_VMM_MULTICAST &&
            request->payload_size == multicast_owner_metadata_size));
    case DYN_VMM_RESTORE_IMPORT:
      return !bootstrap_identify &&
          ((request->handle_type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR && passed_fd >= 0) ||
           (request->handle_type == CU_MEM_HANDLE_TYPE_FABRIC && passed_fd < 0)) &&
          all_zero(request->allocation_uuid, sizeof(request->allocation_uuid)) && request->object_id != 0 &&
          ((request->object_kind == DYN_VMM_ALLOCATION &&
            request->payload_size ==
                importer_metadata_size +
                    (request->handle_type == CU_MEM_HANDLE_TYPE_FABRIC
                         ? DYN_VMM_FABRIC_HANDLE_SIZE
                         : 0)) ||
           (request->object_kind == DYN_VMM_MULTICAST &&
            request->payload_size ==
                multicast_importer_metadata_size +
                    (request->handle_type == CU_MEM_HANDLE_TYPE_FABRIC
                         ? DYN_VMM_FABRIC_HANDLE_SIZE
                         : 0)));
    case DYN_VMM_RESTORE_MULTICAST:
      return !bootstrap_identify && passed_fd < 0 &&
          request->object_kind == DYN_VMM_MULTICAST &&
          supported_handle_type(
              (CUmemAllocationHandleType)request->handle_type) &&
          all_zero(
              request->allocation_uuid,
              sizeof(request->allocation_uuid)) &&
          request->object_id != 0 &&
          (request->payload_size == multicast_owner_metadata_size ||
           request->payload_size == multicast_importer_metadata_size);
    case DYN_VMM_FINALIZE_RESTORE:
    case DYN_VMM_ABORT_RESTORE:
      return !bootstrap_identify && passed_fd < 0 &&
          request->payload_size == 0 && request->handle_type == 0 &&
          request->object_kind == 0 &&
          all_zero(
              request->allocation_uuid,
              sizeof(request->allocation_uuid)) &&
          request->object_id == 0;
    case DYN_VMM_EXPORT_OWNER:
      return !bootstrap_identify && passed_fd < 0 && request->payload_size == 0 &&
          supported_handle_type((CUmemAllocationHandleType)request->handle_type) &&
          (request->object_kind == DYN_VMM_ALLOCATION ||
           request->object_kind == DYN_VMM_MULTICAST) &&
          !all_zero(request->allocation_uuid, sizeof(request->allocation_uuid)) && request->object_id == 0;
    default:
      return false;
  }
}

static void
serve(int client)
{
  struct dyn_vmm_header request;
  struct dyn_vmm_header response;
  char* payload = NULL;
  int passed_fd = -1;

  if (receive_header(client, &request, &passed_fd) != 0)
    goto done;
  memset(&response, 0, sizeof(response));
  response.magic = DYN_VMM_MAGIC;
  response.version = DYN_VMM_VERSION;
  response.operation = request.operation;
  snprintf(
      response.participant_id, sizeof(response.participant_id), "%s",
      participant_id);
  if (!valid_request_shape(&request, passed_fd)) {
    set_error(&response, "invalid VMM control protocol");
    (void)send_header(client, &response, -1);
    goto done;
  }
  if (request.payload_size != 0) {
    payload = malloc((size_t)request.payload_size);
    if (payload == NULL || read_all(client, payload, (size_t)request.payload_size) != 0)
      goto done;
  }
  pthread_mutex_lock(&lock);
  switch (request.operation) {
    case DYN_VMM_INSPECT:
      (void)inspect(client, &response);
      break;
    case DYN_VMM_READ_OWNER:
      (void)read_owner(client, &request, &response);
      break;
    case DYN_VMM_DETACH_IMPORTS:
      (void)detach(client, &request, &response, DYN_VMM_IMPORTER);
      break;
    case DYN_VMM_DETACH_OWNERS:
      (void)detach(client, &request, &response, DYN_VMM_OWNER);
      break;
    case DYN_VMM_RESTORE_OWNER:
      if (request.object_kind == DYN_VMM_MULTICAST)
        (void)restore_multicast_owner(
            client, &request, &response, payload, passed_fd);
      else
        (void)restore_owner(
            client, &request, &response, payload, passed_fd);
      break;
    case DYN_VMM_RESTORE_IMPORT:
      if (request.object_kind == DYN_VMM_MULTICAST)
        (void)restore_multicast_importer(
            client, &request, &response, payload, &passed_fd);
      else
        (void)restore_importer(
            client, &request, &response, payload, &passed_fd);
      break;
    case DYN_VMM_RESTORE_MULTICAST:
      (void)restore_multicast_binding(
          client, &request, &response, payload);
      break;
    case DYN_VMM_FINALIZE_RESTORE:
      (void)finalize_restore(client, &response);
      break;
    case DYN_VMM_ABORT_RESTORE:
      (void)abort_restore(client, &response);
      break;
    case DYN_VMM_IDENTIFY:
      if (failed || phase == DYN_VMM_FAILED)
        set_error(&response, failure[0] != '\0' ? failure : "CUDA VMM shim is unhealthy");
      (void)send_header(client, &response, -1);
      break;
    case DYN_VMM_SET_PLACEMENT:
      (void)set_placement(
          client, &request, &response,
          (const struct dyn_vmm_placement*)payload);
      break;
    case DYN_VMM_EXPORT_OWNER:
      (void)export_owner(client, &request, &response, passed_fd);
      break;
    default:
      set_error(&response, "unknown VMM control operation");
      (void)send_header(client, &response, -1);
      break;
  }
  pthread_mutex_unlock(&lock);
done:
  if (passed_fd >= 0) {
    pthread_mutex_lock(&lock);
    if (close_owned_fd(&passed_fd, "cannot close received VMM broker FD") != 0)
      phase = DYN_VMM_FAILED;
    pthread_mutex_unlock(&lock);
  }
  if (payload != NULL)
    explicit_bzero(payload, (size_t)request.payload_size);
  free(payload);
}

static void*
agent(void* unused)
{
  (void)unused;
  for (;;) {
    int client = accept4(listener, NULL, NULL, SOCK_CLOEXEC);
    if (client < 0) {
      if (errno == EINTR)
        continue;
      return NULL;
    }
    if (set_socket_timeouts(client) != 0) {
      pthread_mutex_lock(&lock);
      fail("cannot bound VMM control client socket");
      phase = DYN_VMM_FAILED;
      pthread_mutex_unlock(&lock);
      (void)close(client);
      continue;
    }
    serve(client);
    if (close(client) != 0) {
      pthread_mutex_lock(&lock);
      fail_cleanup("cannot close VMM control client socket");
      phase = DYN_VMM_FAILED;
      pthread_mutex_unlock(&lock);
    }
  }
}

static void
atfork_prepare(void)
{
  pthread_mutex_lock(&lock);
  if (cuda_seen)
    fail("fork after CUDA initialization is unsupported");
}

static void
atfork_parent(void)
{
  pthread_mutex_unlock(&lock);
}

static void
atfork_child(void)
{
  if (cuda_seen) {
    forked_after_cuda = true;
    fail("fork after CUDA initialization is unsupported");
  }
  pthread_mutex_unlock(&lock);
}

__attribute__((constructor)) static void
initialize(void)
{
  const char* control = getenv("DYN_SNAPSHOT_CONTROL_DIR");
  struct sockaddr_un address = {.sun_family = AF_UNIX};
  pthread_t thread;

  enabled = getenv("DYN_SNAPSHOT_CUDA_VMM_INTERPOSE") != NULL;
  if (!enabled)
    return;
  if (initialize_identity() != 0) {
    fail("cannot initialize CUDA VMM participant identity");
    return;
  }
  if (pthread_atfork(atfork_prepare, atfork_parent, atfork_child) != 0) {
    fail("cannot install fork-after-CUDA rejection");
    return;
  }
  if (control == NULL || control[0] == '\0')
    control = CONTROL_DIR;
  if (!valid_control_directory(control) ||
      snprintf(
          control_directory, sizeof(control_directory), "%s", control) >=
          (int)sizeof(control_directory)) {
    fail("VMM control directory must be a canonical absolute path");
    return;
  }
  if (getpid() <= 0 ||
      format_socket_path(
          socket_path, sizeof(socket_path), (unsigned long)getpid()) != 0) {
    fail("VMM control socket path is too long");
    return;
  }
  snprintf(address.sun_path, sizeof(address.sun_path), "%s", socket_path);
  listener = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  unlink(socket_path);
  if (listener < 0 ||
      bind(listener, (const struct sockaddr*)&address, sizeof(address)) != 0 ||
      listen(listener, 4) != 0 || pthread_create(&thread, NULL, agent, NULL) != 0) {
    fail("cannot start VMM control socket");
    return;
  }
  pthread_detach(thread);
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
cuMulticastCreate(
    CUmemGenericAllocationHandle* output,
    const CUmulticastObjectProp* properties)
{
  typedef CUresult(CUDAAPI * function_type)(
      CUmemGenericAllocationHandle*, const CUmulticastObjectProp*);
  function_type function =
      (function_type)real_symbol("cuMulticastCreate");
  struct allocation* allocation;
  CUmemGenericAllocationHandle real_handle = 0;
  CUmemAllocationHandleType handle_type;
  CUresult result;

  if (!enabled)
    return function != NULL ? function(output, properties) : unavailable();
  if (output == NULL || properties == NULL)
    return function != NULL ? function(output, properties) : unavailable();
  if (!valid_multicast_properties(properties)) {
    pthread_mutex_lock(&lock);
    cuda_seen = true;
    result = function != NULL ? function(output, properties) : unavailable();
    if (result == CUDA_SUCCESS)
      record_unmanaged_multicast(*output);
    pthread_mutex_unlock(&lock);
    return result;
  }
  handle_type = (CUmemAllocationHandleType)properties->handleTypes;
  pthread_mutex_lock(&lock);
  cuda_seen = true;
  result =
      function != NULL ? function(&real_handle, properties) : unavailable();
  if (result == CUDA_SUCCESS) {
    allocation = calloc(1, sizeof(*allocation));
    if (is_logical_handle(real_handle) || allocation == NULL ||
        current_context(&allocation->context) != 0 ||
        allocate_logical_handle(&allocation->logical_handle) !=
            CUDA_SUCCESS) {
      release_fn release = (release_fn)real_symbol("cuMemRelease");

      if (release != NULL)
        (void)release(real_handle);
      free(allocation);
      fail("cannot record cuMulticastCreate metadata");
      result = CUDA_ERROR_OUT_OF_MEMORY;
    } else {
      allocation->real_handle = real_handle;
      allocation->application_handle_live = true;
      allocation->size = properties->size;
      allocation->multicast_properties = *properties;
      allocation->role = DYN_VMM_OWNER;
      allocation->object_kind = DYN_VMM_MULTICAST;
      allocation->requested_handle_type = handle_type;
      allocation->next = allocations;
      allocations = allocation;
      *output = allocation->logical_handle;
    }
  }
  pthread_mutex_unlock(&lock);
  return result;
}

CUresult CUDAAPI
cuMulticastAddDevice(
    CUmemGenericAllocationHandle handle, CUdevice device)
{
  typedef CUresult(CUDAAPI * function_type)(
      CUmemGenericAllocationHandle, CUdevice);
  function_type function =
      (function_type)real_symbol("cuMulticastAddDevice");
  struct allocation* allocation;
  CUresult result;

  if (!enabled || !is_logical_handle(handle))
    return function != NULL ? function(handle, device) : unavailable();
  pthread_mutex_lock(&lock);
  cuda_seen = true;
  allocation = find_logical_handle(handle);
  if (allocation == NULL) {
    result = unknown_logical_handle();
  } else if (allocation->object_kind != DYN_VMM_MULTICAST ||
             !allocation->application_handle_live ||
             allocation->real_handle == 0 || allocation->member_added ||
             allocation->bound) {
    fail("invalid managed CUDA multicast device addition");
    result = CUDA_ERROR_INVALID_HANDLE;
  } else {
    result = function != NULL
                 ? function(allocation->real_handle, device)
                 : unavailable();
    if (result == CUDA_SUCCESS) {
      if (record_multicast_member(allocation, device) != 0) {
        fail("cannot record CUDA multicast member identity");
        result = CUDA_ERROR_INVALID_DEVICE;
      } else {
        allocation->member_added = true;
      }
    }
  }
  pthread_mutex_unlock(&lock);
  return result;
}

static CUresult
bind_multicast_memory(
    CUmemGenericAllocationHandle multicast_handle, CUdevice device,
    bool explicit_device, size_t multicast_offset,
    CUmemGenericAllocationHandle memory_handle, size_t memory_offset,
    size_t size, unsigned long long flags)
{
  typedef CUresult(CUDAAPI * legacy_type)(
      CUmemGenericAllocationHandle, size_t, CUmemGenericAllocationHandle,
      size_t, size_t, unsigned long long);
#if CUDA_VERSION >= 13010
  typedef CUresult(CUDAAPI * v2_type)(
      CUmemGenericAllocationHandle, CUdevice, size_t,
      CUmemGenericAllocationHandle, size_t, size_t, unsigned long long);
#endif
  struct allocation* multicast = find_logical_handle(multicast_handle);
  struct allocation* memory = find_logical_handle(memory_handle);
  CUresult result;

  if (multicast == NULL || memory == NULL) {
    result = unknown_logical_handle();
  } else if (multicast->object_kind != DYN_VMM_MULTICAST ||
             memory->object_kind != DYN_VMM_ALLOCATION ||
             !multicast->application_handle_live ||
             !memory->application_handle_live ||
             multicast->real_handle == 0 || memory->real_handle == 0 ||
             !multicast->member_added || multicast->bound ||
             !memory->exported ||
             multicast->requested_handle_type !=
                 memory->requested_handle_type ||
             multicast_offset != 0 || memory_offset != 0 || flags != 0 ||
             size == 0 || size != multicast->multicast_properties.size ||
             size != memory->size ||
             memcmp(
                 multicast->gpu_uuid.bytes, memory->gpu_uuid.bytes,
                 sizeof(multicast->gpu_uuid.bytes)) != 0 ||
             (explicit_device && device != multicast->member_device)) {
    fail("invalid managed CUDA multicast memory binding");
    result = CUDA_ERROR_INVALID_VALUE;
  } else if (explicit_device) {
#if CUDA_VERSION >= 13010
    v2_type function =
        (v2_type)real_symbol("cuMulticastBindMem_v2");
    result = function != NULL
                 ? function(
                       multicast->real_handle, device, multicast_offset,
                       memory->real_handle, memory_offset, size, flags)
                 : unavailable();
#else
    result = unavailable();
#endif
  } else {
    legacy_type function =
        (legacy_type)real_symbol("cuMulticastBindMem");
    result = function != NULL
                 ? function(
                       multicast->real_handle, multicast_offset,
                       memory->real_handle, memory_offset, size, flags)
                 : unavailable();
  }
  if (result == CUDA_SUCCESS) {
    memcpy(
        multicast->backing_allocation_uuid, memory->allocation_uuid,
        sizeof(multicast->backing_allocation_uuid));
    multicast->backing_role = memory->role;
    multicast->bind_api = explicit_device
        ? DYN_VMM_MULTICAST_BIND_MEM_V2
        : DYN_VMM_MULTICAST_BIND_MEM;
    multicast->multicast_offset = multicast_offset;
    multicast->memory_offset = memory_offset;
    multicast->bind_size = size;
    multicast->bind_flags = flags;
    multicast->bound = true;
  }
  return result;
}

CUresult CUDAAPI
cuMemCreate(
    CUmemGenericAllocationHandle* output, size_t size,
    const CUmemAllocationProp* properties, unsigned long long flags)
{
  create_fn function = (create_fn)real_symbol("cuMemCreate");
  struct allocation* allocation;
  CUmemGenericAllocationHandle real_handle = 0;
  CUresult result;

  if (!enabled)
    return function != NULL
        ? function(output, size, properties, flags)
        : unavailable();
  if (properties == NULL || output == NULL)
    return function != NULL
        ? function(output, size, properties, flags)
        : unavailable();
  if (!supported_handle_type(properties->requestedHandleTypes))
    return unavailable();
  pthread_mutex_lock(&lock);
  cuda_seen = true;
  result = function != NULL
      ? function(&real_handle, size, properties, flags)
      : unavailable();
  if (result == CUDA_SUCCESS) {
    if (is_logical_handle(real_handle)) {
      release_fn release = (release_fn)real_symbol("cuMemRelease");
      if (release != NULL)
        (void)release(real_handle);
      fail("real cuMemCreate handle collides with logical tag");
      result = CUDA_ERROR_INVALID_HANDLE;
      pthread_mutex_unlock(&lock);
      return result;
    }
    allocation = calloc(1, sizeof(*allocation));
    if (is_logical_handle(real_handle) || allocation == NULL ||
        current_context(&allocation->context) != 0 ||
        allocate_logical_handle(&allocation->logical_handle) != CUDA_SUCCESS) {
      release_fn release = (release_fn)real_symbol("cuMemRelease");
      if (release != NULL)
        (void)release(real_handle);
      free(allocation);
      fail("cannot record cuMemCreate metadata");
      result = CUDA_ERROR_OUT_OF_MEMORY;
    } else {
      allocation->real_handle = real_handle;
      allocation->application_handle_live = true;
      allocation->size = size;
      allocation->properties = *properties;
      allocation->role = DYN_VMM_OWNER;
      allocation->object_kind = DYN_VMM_ALLOCATION;
      allocation->requested_handle_type = properties->requestedHandleTypes;
      allocation->next = allocations;
      allocations = allocation;
      if (record_gpu_identity(allocation) != 0)
        fail("cannot record CUDA allocation GPU identity");
      *output = allocation->logical_handle;
    }
  }
  pthread_mutex_unlock(&lock);
  return result;
}

CUresult CUDAAPI
cuMemRelease(CUmemGenericAllocationHandle handle)
{
  release_fn function = (release_fn)real_symbol("cuMemRelease");
  CUresult result;

  if (!enabled)
    return function != NULL ? function(handle) : unavailable();
  pthread_mutex_lock(&lock);
  cuda_seen = true;
  if (!is_logical_handle(handle)) {
    result = function != NULL ? function(handle) : unavailable();
    if (result == CUDA_SUCCESS)
      remove_unmanaged_multicast(handle);
    pthread_mutex_unlock(&lock);
    return result;
  }
  struct allocation* allocation = find_logical_handle(handle);
  if (allocation == NULL || !allocation->application_handle_live) {
    result = unknown_logical_handle();
  } else if (allocation->real_handle == 0) {
    fail("logical generic allocation handle has no real backing");
    result = CUDA_ERROR_INVALID_HANDLE;
  } else if (
      allocation->object_kind == DYN_VMM_MULTICAST &&
      allocation->mapped) {
    fail("managed CUDA multicast release requires unmap");
    result = CUDA_ERROR_INVALID_VALUE;
  } else {
    result = function != NULL
        ? function(allocation->real_handle)
        : unavailable();
    if (result == CUDA_SUCCESS) {
      if (allocation->object_kind == DYN_VMM_MULTICAST) {
        allocation->bound = false;
        allocation->member_added = false;
        memset(
            allocation->backing_allocation_uuid, 0,
            sizeof(allocation->backing_allocation_uuid));
        allocation->backing_object_id = 0;
        allocation->backing_role = 0;
        allocation->bind_api = 0;
        allocation->multicast_offset = 0;
        allocation->memory_offset = 0;
        allocation->bind_size = 0;
        allocation->bind_flags = 0;
      }
      allocation->real_handle = 0;
      allocation->application_handle_live = false;
      if (!allocation->mapped)
        remove_allocation(allocation);
    }
  }
  pthread_mutex_unlock(&lock);
  return result;
}

CUresult CUDAAPI
cuMemMap(
    CUdeviceptr address, size_t size, size_t offset,
    CUmemGenericAllocationHandle handle, unsigned long long flags)
{
  map_fn function = (map_fn)real_symbol("cuMemMap");
  CUresult result;

  if (!enabled)
    return function != NULL ? function(address, size, offset, handle, flags) : unavailable();
  pthread_mutex_lock(&lock);
  cuda_seen = true;
  if (!is_logical_handle(handle)) {
    if (find_unmanaged_multicast(handle) == NULL)
      fail("cuMemMap used an untracked real generic handle");
    result = function != NULL
        ? function(address, size, offset, handle, flags)
        : unavailable();
    pthread_mutex_unlock(&lock);
    return result;
  }
  struct allocation* allocation = find_logical_handle(handle);
  if (allocation == NULL || !allocation->application_handle_live ||
      allocation->real_handle == 0) {
    result = unknown_logical_handle();
  } else {
    result = function != NULL
        ? function(
              address, size, offset, allocation->real_handle, flags)
        : unavailable();
    if (result == CUDA_SUCCESS) {
      if (allocation->mapped ||
          (allocation->role == DYN_VMM_OWNER && size != allocation->size) ||
          offset != 0)
        fail("managed allocation has multiple, partial, or nonzero-offset mappings");
      else {
        allocation->address = address;
        allocation->size = size;
        allocation->offset = offset;
        allocation->mapped = true;
        if (current_context(&allocation->context) != 0)
          fail("cannot record cuMemMap context");
      }
    }
  }
  pthread_mutex_unlock(&lock);
  return result;
}

CUresult CUDAAPI
cuMemUnmap(CUdeviceptr address, size_t size)
{
  unmap_fn function = (unmap_fn)real_symbol("cuMemUnmap");
  CUresult result;

  if (!enabled)
    return function != NULL ? function(address, size) : unavailable();
  pthread_mutex_lock(&lock);
  cuda_seen = true;
  result = function != NULL ? function(address, size) : unavailable();
  if (result == CUDA_SUCCESS) {
    struct allocation* allocation = find_mapping(address, size);
    if (allocation != NULL) {
      allocation->mapped = false;
      if (!allocation->application_handle_live)
        remove_allocation(allocation);
    }
  }
  pthread_mutex_unlock(&lock);
  return result;
}

CUresult CUDAAPI
cuMemSetAccess(
    CUdeviceptr address, size_t size, const CUmemAccessDesc* descriptors,
    size_t count)
{
  access_fn function = (access_fn)real_symbol("cuMemSetAccess");
  CUresult result;

  if (!enabled)
    return function != NULL
        ? function(address, size, descriptors, count)
        : unavailable();
  pthread_mutex_lock(&lock);
  cuda_seen = true;
  result = function != NULL
      ? function(address, size, descriptors, count)
      : unavailable();
  if (result == CUDA_SUCCESS) {
    struct allocation* allocation;
    struct access_update* updates = NULL;
    size_t mapping_count = 0;
    size_t index = 0;
    CUdeviceptr cursor;
    CUdeviceptr end;

    for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
      if (overlaps(allocation, address, size))
        mapping_count++;
    }
    if (mapping_count == 0)
      goto access_done;
    if (count == 0 || descriptors == NULL ||
        address > UINT64_MAX - size ||
        count > SIZE_MAX / sizeof(*descriptors) ||
        mapping_count > SIZE_MAX / sizeof(*updates)) {
      fail("invalid cuMemSetAccess range or descriptors for managed mappings");
      goto access_done;
    }
    updates = calloc(mapping_count, sizeof(*updates));
    if (updates == NULL) {
      fail("cannot record cuMemSetAccess descriptors");
      goto access_done;
    }
    for (allocation = allocations; allocation != NULL; allocation = allocation->next) {
      if (overlaps(allocation, address, size))
        updates[index++].allocation = allocation;
    }
    qsort(
        updates, mapping_count, sizeof(*updates),
        compare_access_updates);
    cursor = address;
    end = address + size;
    for (index = 0; index < mapping_count; index++) {
      allocation = updates[index].allocation;
      if (allocation->address != cursor || allocation->size == 0 ||
          allocation->address > UINT64_MAX - allocation->size) {
        fail("cuMemSetAccess range is not an exact contiguous union of managed mappings");
        goto access_done;
      }
      if (allocation->access_count != 0) {
        fail("multiple access updates observed for managed mapping");
        goto access_done;
      }
      cursor += allocation->size;
    }
    if (cursor != end) {
      fail("cuMemSetAccess range is not an exact contiguous union of managed mappings");
      goto access_done;
    }
    for (index = 0; index < mapping_count; index++) {
      if (test_failure("access-copy-second") && index == 1)
        updates[index].access = NULL;
      else
        updates[index].access = malloc(count * sizeof(*descriptors));
      if (updates[index].access == NULL) {
        fail("cannot record cuMemSetAccess descriptors");
        goto access_done;
      }
      memcpy(
          updates[index].access, descriptors,
          count * sizeof(*descriptors));
    }
    for (index = 0; index < mapping_count; index++) {
      updates[index].allocation->access = updates[index].access;
      updates[index].allocation->access_count = count;
      updates[index].access = NULL;
    }
access_done:
    if (updates != NULL) {
      for (index = 0; index < mapping_count; index++)
        free(updates[index].access);
    }
    free(updates);
  }
  pthread_mutex_unlock(&lock);
  return result;
}

CUresult CUDAAPI
cuMemExportToShareableHandle(
    void* shareable, CUmemGenericAllocationHandle handle,
    CUmemAllocationHandleType type, unsigned long long flags)
{
  export_fn function = (export_fn)real_symbol("cuMemExportToShareableHandle");
  struct allocation* allocation;
  struct dyn_vmm_fabric_token fabric_token;
  uint8_t raw_fabric[DYN_VMM_FABRIC_HANDLE_SIZE];
  uint8_t uuid[DYN_VMM_ALLOCATION_UUID_SIZE];
  int capability_fd = -1;
  int raw_fd = -1;
  CUresult result;

  if (!enabled)
    return function != NULL ? function(shareable, handle, type, flags) : unavailable();
  if (!supported_handle_type(type) || shareable == NULL)
    return unavailable();
  memset(raw_fabric, 0, sizeof(raw_fabric));
  if (!is_logical_handle(handle)) {
    pthread_mutex_lock(&lock);
    cuda_seen = true;
    fail("shareable export requires a shim logical handle");
    pthread_mutex_unlock(&lock);
    return CUDA_ERROR_INVALID_HANDLE;
  }
  pthread_mutex_lock(&lock);
  cuda_seen = true;
  allocation = find_logical_handle(handle);
  if (allocation == NULL || !allocation->application_handle_live || allocation->real_handle == 0 ||
      allocation->role != DYN_VMM_OWNER || allocation->requested_handle_type != type || failed || forked_after_cuda ||
      phase != DYN_VMM_ACTIVE) {
    result = unknown_logical_handle();
  } else {
    result = function != NULL
                 ? function(
                       type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR ? (void*)&raw_fd : (void*)raw_fabric,
                       allocation->real_handle, type, flags)
                 : unavailable();
  }
  if (result == CUDA_SUCCESS) {
    if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR && raw_fd < 0) {
      fail("real CUDA export returned an invalid POSIX FD");
      result = CUDA_ERROR_INVALID_VALUE;
    } else {
      memcpy(uuid, allocation->allocation_uuid, sizeof(uuid));
      if ((!allocation->exported && random_uuid(uuid) != 0) ||
          (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
               ? create_capability(
                     uuid, allocation->object_kind, &capability_fd)
               : create_fabric_token(
                     uuid, allocation->object_kind, &fabric_token)) != 0) {
        fail("cannot create CUDA VMM allocation capability");
        result = CUDA_ERROR_OUT_OF_MEMORY;
      }
    }
  }
  if (close_owned_fd(&raw_fd, "cannot close raw CUDA export FD") != 0) {
    result = CUDA_ERROR_UNKNOWN;
    phase = DYN_VMM_FAILED;
  }
  if (result == CUDA_SUCCESS) {
    memcpy(allocation->allocation_uuid, uuid, sizeof(allocation->allocation_uuid));
    if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      *(int*)shareable = capability_fd;
      capability_fd = -1;
    } else {
      memcpy(shareable, &fabric_token, sizeof(fabric_token));
    }
    allocation->exported = true;
  }
  if (capability_fd >= 0) {
    if (close_owned_fd(&capability_fd, "cannot close failed CUDA VMM capability FD") != 0) {
      result = CUDA_ERROR_UNKNOWN;
      phase = DYN_VMM_FAILED;
    }
  }
  explicit_bzero(raw_fabric, sizeof(raw_fabric));
  explicit_bzero(&fabric_token, sizeof(fabric_token));
  pthread_mutex_unlock(&lock);
  return result;
}

CUresult CUDAAPI
cuMemImportFromShareableHandle(
    CUmemGenericAllocationHandle* output, void* os_handle,
    CUmemAllocationHandleType type)
{
  import_fn function = (import_fn)real_symbol("cuMemImportFromShareableHandle");
  properties_fn get_properties = (properties_fn)real_symbol("cuMemGetAllocationPropertiesFromHandle");
  release_fn release = (release_fn)real_symbol("cuMemRelease");
  struct dyn_vmm_capability capability;
  struct dyn_vmm_fabric_token fabric_token;
  struct broker_result broker;
  struct allocation* allocation = NULL;
  struct allocation* owner;
  CUmemGenericAllocationHandle real_handle = 0;
  int capability_fd = (int)(uintptr_t)os_handle;
  uint8_t allocation_uuid[DYN_VMM_ALLOCATION_UUID_SIZE];
  char owner_participant[DYN_VMM_PARTICIPANT_ID_SIZE];
  char owner_socket[sizeof(((struct sockaddr_un*)0)->sun_path)];
  CUmulticastObjectProp multicast_properties;
  uint32_t object_kind = 0;
  bool local_owner;
  CUresult result;

  if (!enabled)
    return function != NULL ? function(output, os_handle, type) : unavailable();
  initialize_broker_result(&broker, type, 0);
  memset(&fabric_token, 0, sizeof(fabric_token));
  memset(&capability, 0, sizeof(capability));
  memset(allocation_uuid, 0, sizeof(allocation_uuid));
  memset(owner_participant, 0, sizeof(owner_participant));
  memset(owner_socket, 0, sizeof(owner_socket));
  memset(&multicast_properties, 0, sizeof(multicast_properties));
  if (!supported_handle_type(type) || output == NULL) {
    pthread_mutex_lock(&lock);
    cuda_seen = true;
    fail("shareable import requires a VMM capability");
    pthread_mutex_unlock(&lock);
    return CUDA_ERROR_INVALID_VALUE;
  }
  pthread_mutex_lock(&lock);
  cuda_seen = true;
  if (type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    if (read_capability(capability_fd, &capability) != 0) {
      fail("shareable import requires a valid sealed VMM capability");
      result = CUDA_ERROR_INVALID_VALUE;
      goto done;
    }
    memcpy(allocation_uuid, capability.allocation_uuid, sizeof(allocation_uuid));
    object_kind = capability.object_kind;
    snprintf(owner_participant, sizeof(owner_participant), "%s", capability.owner_participant_id);
    snprintf(owner_socket, sizeof(owner_socket), "%s", capability.owner_socket_path);
  } else if (read_fabric_token(os_handle, &fabric_token, owner_participant, owner_socket) != 0) {
    fail("shareable import requires a valid sealed VMM capability");
    result = CUDA_ERROR_INVALID_VALUE;
    goto done;
  } else {
    memcpy(allocation_uuid, fabric_token.allocation_uuid, sizeof(allocation_uuid));
    object_kind = fabric_token.object_kind;
  }
  broker.object_kind = object_kind;
  if (failed || forked_after_cuda || phase != DYN_VMM_ACTIVE) {
    result = CUDA_ERROR_INVALID_VALUE;
    goto done;
  }
  local_owner = strcmp(owner_socket, socket_path) == 0 && strcmp(owner_participant, participant_id) == 0;
  if (local_owner) {
    owner = find_object(allocation_uuid, DYN_VMM_OWNER);
    if (owner == NULL || owner->requested_handle_type != type ||
        owner->object_kind != object_kind)
      result = CUDA_ERROR_INVALID_HANDLE;
    else
      result = export_raw_owner(owner, &broker);
    if (result != CUDA_SUCCESS) {
      fail("local CUDA VMM capability owner is unavailable");
      result = CUDA_ERROR_INVALID_VALUE;
      goto done;
    }
  } else {
    pthread_mutex_unlock(&lock);
    result = request_owner_export(
                 allocation_uuid, owner_participant, owner_socket, type,
                 object_kind, &broker) == 0
                 ? CUDA_SUCCESS
                 : CUDA_ERROR_INVALID_VALUE;
    pthread_mutex_lock(&lock);
    if (result != CUDA_SUCCESS || failed || forked_after_cuda || phase != DYN_VMM_ACTIVE) {
      fail("remote CUDA VMM capability owner is unavailable");
      result = CUDA_ERROR_INVALID_VALUE;
      goto done;
    }
  }
  result = function != NULL ? function(
                                  &real_handle,
                                  type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR ? (void*)(uintptr_t)broker.fd
                                                                                   : (void*)broker.bytes,
                                  type)
                            : unavailable();
  if (result == CUDA_SUCCESS && object_kind == DYN_VMM_MULTICAST &&
      broker.has_multicast_properties)
    multicast_properties = broker.multicast_properties;
  if (clear_broker_result(&broker, "cannot close imported raw CUDA broker FD") != 0) {
    if (result == CUDA_SUCCESS && (release == NULL || release(real_handle) != CUDA_SUCCESS))
      fail_cleanup("cannot release failed imported CUDA handle");
    fail("cannot close imported raw CUDA broker FD");
    result = CUDA_ERROR_UNKNOWN;
    goto done;
  }
  if (result == CUDA_SUCCESS) {
    allocation = calloc(1, sizeof(*allocation));
    if (allocation == NULL ||
        (object_kind == DYN_VMM_ALLOCATION &&
         (get_properties == NULL ||
          get_properties(&allocation->properties, real_handle) !=
              CUDA_SUCCESS)) ||
        (object_kind == DYN_VMM_MULTICAST &&
         multicast_properties.size == 0) ||
        current_context(&allocation->context) != 0 ||
        allocate_logical_handle(&allocation->logical_handle) != CUDA_SUCCESS ||
        (object_kind == DYN_VMM_ALLOCATION &&
         record_gpu_identity(allocation) != 0)) {
      if (release == NULL || release(real_handle) != CUDA_SUCCESS)
        fail_cleanup("cannot release untracked imported CUDA handle");
      free(allocation);
      fail("cannot record VMM capability import");
      result = CUDA_ERROR_INVALID_VALUE;
    } else {
      memcpy(allocation->allocation_uuid, allocation_uuid, sizeof(allocation->allocation_uuid));
      allocation->exported = true;
      allocation->real_handle = real_handle;
      allocation->application_handle_live = true;
      allocation->role = DYN_VMM_IMPORTER;
      allocation->object_kind = object_kind;
      if (object_kind == DYN_VMM_MULTICAST) {
        allocation->size = multicast_properties.size;
        allocation->multicast_properties = multicast_properties;
      }
      /* CUDA reports re-exportable types, not the transport used to import. */
      allocation->requested_handle_type = type;
      allocation->next = allocations;
      allocations = allocation;
      *output = allocation->logical_handle;
    }
  } else {
    fail("real CUDA capability import failed");
  }
done:
  if (clear_broker_result(&broker, "cannot close failed raw CUDA broker FD") != 0) {
    result = CUDA_ERROR_UNKNOWN;
    phase = DYN_VMM_FAILED;
  }
  explicit_bzero(&fabric_token, sizeof(fabric_token));
  explicit_bzero(&multicast_properties, sizeof(multicast_properties));
  pthread_mutex_unlock(&lock);
  return result;
}

CUresult CUDAAPI
cuMemRetainAllocationHandle(
    CUmemGenericAllocationHandle* output, void* address)
{
  retain_fn function = (retain_fn)real_symbol("cuMemRetainAllocationHandle");
  CUresult result;

  if (!enabled)
    return function != NULL ? function(output, address) : unavailable();
  pthread_mutex_lock(&lock);
  cuda_seen = true;
  result = function != NULL ? function(output, address) : unavailable();
  if (result == CUDA_SUCCESS)
    fail("cuMemRetainAllocationHandle succeeded; retained handles are unsupported");
  pthread_mutex_unlock(&lock);
  return result;
}

CUresult CUDAAPI
cuMemGetAllocationPropertiesFromHandle(
    CUmemAllocationProp* properties, CUmemGenericAllocationHandle handle)
{
  properties_fn function =
      (properties_fn)real_symbol("cuMemGetAllocationPropertiesFromHandle");
  CUresult result;

  if (!enabled || !is_logical_handle(handle))
    return function != NULL ? function(properties, handle) : unavailable();
  pthread_mutex_lock(&lock);
  cuda_seen = true;
  struct allocation* allocation = find_logical_handle(handle);
  if (allocation == NULL || !allocation->application_handle_live ||
      allocation->real_handle == 0)
    result = unknown_logical_handle();
  else
    result = function != NULL
        ? function(properties, allocation->real_handle)
        : unavailable();
  pthread_mutex_unlock(&lock);
  return result;
}

CUresult CUDAAPI
cuMulticastBindMem(
    CUmemGenericAllocationHandle multicast_handle, size_t multicast_offset,
    CUmemGenericAllocationHandle memory_handle, size_t memory_offset,
    size_t size, unsigned long long flags)
{
  typedef CUresult(CUDAAPI * function_type)(
      CUmemGenericAllocationHandle, size_t, CUmemGenericAllocationHandle,
      size_t, size_t, unsigned long long);
  CUresult result;

  if (!enabled ||
      (!is_logical_handle(multicast_handle) &&
       !is_logical_handle(memory_handle))) {
    function_type function =
        (function_type)real_symbol("cuMulticastBindMem");
    return function != NULL
        ? function(
              multicast_handle, multicast_offset, memory_handle, memory_offset,
              size, flags)
        : unavailable();
  }
  pthread_mutex_lock(&lock);
  cuda_seen = true;
  if (!is_logical_handle(multicast_handle) ||
      !is_logical_handle(memory_handle)) {
    fail("mixed managed and unmanaged CUDA multicast binding");
    result = CUDA_ERROR_INVALID_HANDLE;
  } else if (find_logical_handle(multicast_handle) == NULL ||
             find_logical_handle(memory_handle) == NULL) {
    result = unknown_logical_handle();
  } else {
    result = bind_multicast_memory(
        multicast_handle, 0, false, multicast_offset, memory_handle,
        memory_offset, size, flags);
  }
  pthread_mutex_unlock(&lock);
  return result;
}

#if CUDA_VERSION >= 13010
CUresult CUDAAPI
cuMulticastBindMem_v2(
    CUmemGenericAllocationHandle multicast_handle, CUdevice device,
    size_t multicast_offset, CUmemGenericAllocationHandle memory_handle,
    size_t memory_offset, size_t size, unsigned long long flags)
{
  typedef CUresult(CUDAAPI * function_type)(
      CUmemGenericAllocationHandle, CUdevice, size_t,
      CUmemGenericAllocationHandle, size_t, size_t, unsigned long long);
  CUresult result;

  if (!enabled ||
      (!is_logical_handle(multicast_handle) &&
       !is_logical_handle(memory_handle))) {
    function_type function =
        (function_type)real_symbol("cuMulticastBindMem_v2");
    return function != NULL
        ? function(
              multicast_handle, device, multicast_offset, memory_handle,
              memory_offset, size, flags)
        : unavailable();
  }
  pthread_mutex_lock(&lock);
  cuda_seen = true;
  if (!is_logical_handle(multicast_handle) ||
      !is_logical_handle(memory_handle)) {
    fail("mixed managed and unmanaged CUDA multicast binding");
    result = CUDA_ERROR_INVALID_HANDLE;
  } else if (find_logical_handle(multicast_handle) == NULL ||
             find_logical_handle(memory_handle) == NULL) {
    result = unknown_logical_handle();
  } else {
    result = bind_multicast_memory(
        multicast_handle, device, true, multicast_offset, memory_handle,
        memory_offset, size, flags);
  }
  pthread_mutex_unlock(&lock);
  return result;
}
#endif

static void*
replacement(const char* symbol, int version)
{
#define ENTRY(name)       \
  if (strcmp(symbol, #name) == 0) \
    return (void*)&name
  if (symbol == NULL)
    return NULL;
  ENTRY(cuMemCreate);
  ENTRY(cuMemRelease);
  ENTRY(cuMemMap);
  ENTRY(cuMemUnmap);
  ENTRY(cuMemSetAccess);
  ENTRY(cuMemExportToShareableHandle);
  ENTRY(cuMemImportFromShareableHandle);
  ENTRY(cuMemRetainAllocationHandle);
  ENTRY(cuMemGetAllocationPropertiesFromHandle);
  ENTRY(cuMulticastCreate);
  ENTRY(cuMulticastAddDevice);
  ENTRY(cuMulticastBindAddr);
  ENTRY(cuMulticastUnbind);
#if CUDA_VERSION >= 13010
  ENTRY(cuMulticastBindMem_v2);
  ENTRY(cuMulticastBindAddr_v2);
#endif
  ENTRY(cuGetProcAddress_v2);
  ENTRY(cuGetProcAddress_v2_ptsz);
#undef ENTRY
  if (strcmp(symbol, "cuMulticastBindMem") == 0) {
#if CUDA_VERSION >= 13010
    if (version >= 13010)
      return (void*)&cuMulticastBindMem_v2;
#endif
    return (void*)&cuMulticastBindMem;
  }
  if (strcmp(symbol, "cuMulticastBindAddr") == 0) {
#if CUDA_VERSION >= 13010
    if (version >= 13010)
      return (void*)&cuMulticastBindAddr_v2;
#endif
    return (void*)&cuMulticastBindAddr;
  }
  if (strcmp(symbol, "cuGetProcAddress") == 0)
    return version >= 12000 ? (void*)&cuGetProcAddress_v2 : (void*)&cuGetProcAddress;
  return NULL;
}

CUresult CUDAAPI
cuGetProcAddress(
    const char* symbol, void** output, int version, cuuint64_t flags)
{
  typedef CUresult(CUDAAPI * function_type)(const char*, void**, int, cuuint64_t);
  function_type function = (function_type)atomic_load(&explicit_cu_get_proc_address);
  if (function == NULL)
    function = (function_type)real_symbol("cuGetProcAddress");
  CUresult result =
      function != NULL ? function(symbol, output, version, flags) : unavailable();
  void* entry;

  if (enabled && result == CUDA_SUCCESS && output != NULL && *output != NULL &&
      (entry = replacement(symbol, version)) != NULL)
    *output = entry;
  return result;
}

CUresult CUDAAPI
cuGetProcAddress_v2(
    const char* symbol, void** output, int version, cuuint64_t flags,
    CUdriverProcAddressQueryResult* status)
{
  typedef CUresult(CUDAAPI * function_type)(
      const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
  function_type function =
      (function_type)atomic_load(&explicit_cu_get_proc_address_v2);
  if (function == NULL)
    function = (function_type)real_symbol("cuGetProcAddress_v2");
  CUresult result = function != NULL
      ? function(symbol, output, version, flags, status)
      : unavailable();
  void* entry;

  if (enabled && result == CUDA_SUCCESS && output != NULL && *output != NULL &&
      (status == NULL || *status == CU_GET_PROC_ADDRESS_SUCCESS) &&
      (entry = replacement(symbol, version)) != NULL)
    *output = entry;
  return result;
}

CUresult CUDAAPI
cuGetProcAddress_v2_ptsz(
    const char* symbol, void** output, int version, cuuint64_t flags,
    CUdriverProcAddressQueryResult* status)
{
  typedef CUresult(CUDAAPI * function_type)(
      const char*, void**, int, cuuint64_t, CUdriverProcAddressQueryResult*);
  function_type function =
      (function_type)atomic_load(&explicit_cu_get_proc_address_v2);
  if (function == NULL)
    function = (function_type)real_symbol("cuGetProcAddress_v2");
  cuuint64_t stream_flags =
      CU_GET_PROC_ADDRESS_LEGACY_STREAM |
      CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM;
  CUresult result = function != NULL
      ? function(
            symbol, output, version,
            (flags & stream_flags) == 0
                ? flags | CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM
                : flags,
            status)
      : unavailable();
  void* entry;

  if (enabled && result == CUDA_SUCCESS && output != NULL && *output != NULL &&
      (status == NULL || *status == CU_GET_PROC_ADDRESS_SUCCESS) &&
      (entry = replacement(symbol, version)) != NULL)
    *output = entry;
  return result;
}

static cudaError_t
runtime_resolve(
    const char* real_name, const char* symbol, void** output,
    unsigned int version, unsigned long long flags,
    enum cudaDriverEntryPointQueryResult* status, bool versioned,
    bool per_thread_default_stream)
{
  typedef cudaError_t(CUDARTAPI * old_type)(
      const char*, void**, unsigned long long,
      enum cudaDriverEntryPointQueryResult*);
  typedef cudaError_t(CUDARTAPI * version_type)(
      const char*, void**, unsigned int, unsigned long long,
      enum cudaDriverEntryPointQueryResult*);
  unsigned long long stream_flags =
      cudaEnableLegacyStream | cudaEnablePerThreadDefaultStream;
  void* raw = real_symbol(real_name);
  if (per_thread_default_stream && (flags & stream_flags) == 0)
    flags |= cudaEnablePerThreadDefaultStream;
  cudaError_t result = raw == NULL
      ? cudaErrorNotSupported
      : versioned
      ? ((version_type)raw)(symbol, output, version, flags, status)
      : ((old_type)raw)(symbol, output, flags, status);
  void* entry;

  if (enabled && result == cudaSuccess && output != NULL && *output != NULL &&
      (status == NULL || *status == cudaDriverEntryPointSuccess) &&
      (entry = replacement(symbol, versioned ? (int)version : CUDA_VERSION)) != NULL)
    *output = entry;
  return result;
}

cudaError_t CUDARTAPI
cudaGetDriverEntryPoint(
    const char* symbol, void** output, unsigned long long flags,
    enum cudaDriverEntryPointQueryResult* status)
{
  return runtime_resolve(
      "cudaGetDriverEntryPoint", symbol, output, 0, flags, status, false,
      false);
}

cudaError_t CUDARTAPI
cudaGetDriverEntryPointByVersion(
    const char* symbol, void** output, unsigned int version,
    unsigned long long flags, enum cudaDriverEntryPointQueryResult* status)
{
  return runtime_resolve(
      "cudaGetDriverEntryPointByVersion", symbol, output, version, flags,
      status, true, false);
}

cudaError_t CUDARTAPI
cudaGetDriverEntryPoint_ptsz(
    const char* symbol, void** output, unsigned long long flags,
    enum cudaDriverEntryPointQueryResult* status)
{
  return runtime_resolve(
      "cudaGetDriverEntryPoint", symbol, output, 0, flags, status, false,
      true);
}

cudaError_t CUDARTAPI
cudaGetDriverEntryPointByVersion_ptsz(
    const char* symbol, void** output, unsigned int version,
    unsigned long long flags, enum cudaDriverEntryPointQueryResult* status)
{
  return runtime_resolve(
      "cudaGetDriverEntryPointByVersion", symbol, output, version, flags,
      status, true, true);
}
