/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE

#include <cuda.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include "../protocol.h"

unsigned int fake_cuda_create_count(void);
unsigned int fake_cuda_import_count(void);
unsigned int fake_cuda_release_count(void);
unsigned int fake_cuda_map_count(void);
unsigned int fake_cuda_export_count(void);
unsigned int fake_cuda_logical_forward_count(void);
unsigned int fake_cuda_active_handle_count(void);
unsigned int fake_cuda_active_mapping_count(void);
CUmemGenericAllocationHandle fake_cuda_last_consumed_handle(void);
CUcontext fake_cuda_last_context(void);
int fake_cuda_last_exported_fd(void);
int fake_cuda_last_internal_export_alias(void);
int fake_cuda_internal_export_alias(unsigned int);
int fake_cuda_last_imported_fd(void);
CUdeviceptr fake_cuda_map_address(unsigned int);
CUdeviceptr fake_cuda_access_address(unsigned int);
CUmemAccessDesc fake_cuda_access_descriptor(unsigned int);
size_t fake_cuda_access_count(unsigned int);
unsigned int fake_cuda_set_access_count(void);
CUdeviceptr fake_cuda_copy_destination(void);
size_t fake_cuda_copy_size(void);
unsigned char fake_cuda_copy_byte(void);
int fake_cuda_copy_uniform(void);
CUmemGenericAllocationHandle fake_cuda_released_handle(unsigned int);

static void
require(int condition, const char* message)
{
  if (!condition) {
    fprintf(stderr, "%s\n", message);
    exit(1);
  }
}

static struct dyn_vmm_capability
read_capability(int fd)
{
  struct dyn_vmm_capability capability;
  struct stat status;
  const int required_seals = F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL;

  memset(&capability, 0, sizeof(capability));
  require(
      fstat(fd, &status) == 0 && status.st_size == (off_t)sizeof(capability) &&
          pread(fd, &capability, sizeof(capability), 0) == (ssize_t)sizeof(capability) &&
          capability.magic == DYN_VMM_CAPABILITY_MAGIC && capability.version == DYN_VMM_CAPABILITY_VERSION &&
          (fcntl(fd, F_GET_SEALS) & required_seals) == required_seals,
      "invalid VMM application capability");
  return capability;
}

static int
make_capability(const struct dyn_vmm_capability* capability, size_t size, int sealed)
{
  const int required_seals = F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL;
  int fd = memfd_create("dynamo-vmm-test-capability", MFD_CLOEXEC | MFD_ALLOW_SEALING);

  require(
      fd >= 0 && size <= sizeof(*capability) && write(fd, capability, size) == (ssize_t)size &&
          (!sealed || fcntl(fd, F_ADD_SEALS, required_seals) == 0),
      "cannot create test VMM capability");
  return fd;
}

static int
capability_is_valid(int fd)
{
  typedef int (*function_type)(int);
  function_type function =
      (function_type)dlsym(RTLD_DEFAULT, "dyn_vmm_test_capability_valid");

  require(function != NULL, "cannot resolve capability validation test seam");
  return function(fd);
}

static void
set_capability_path(struct dyn_vmm_capability* capability, const char* path)
{
  memset(
      capability->owner_socket_path, 0,
      sizeof(capability->owner_socket_path));
  require(
      snprintf(
          capability->owner_socket_path,
          sizeof(capability->owner_socket_path), "%s", path) <
          (int)sizeof(capability->owner_socket_path),
      "test capability path is too long");
}

static void
set_capability_control_path(
    struct dyn_vmm_capability* capability, const char* basename)
{
  const char* control = getenv("DYN_SNAPSHOT_CONTROL_DIR");
  const char* separator = strcmp(control, "/") == 0 ? "" : "/";

  memset(
      capability->owner_socket_path, 0,
      sizeof(capability->owner_socket_path));
  require(
      snprintf(
          capability->owner_socket_path,
          sizeof(capability->owner_socket_path), "%s%s%s", control, separator,
          basename) < (int)sizeof(capability->owner_socket_path),
      "test capability control path is too long");
}

static void
send_application_fd(int socket_fd, int fd)
{
  char control[CMSG_SPACE(sizeof(int))] = {0};
  char byte = 0;
  struct iovec vector = {.iov_base = &byte, .iov_len = sizeof(byte)};
  struct msghdr message = {
      .msg_iov = &vector,
      .msg_iovlen = 1,
      .msg_control = control,
      .msg_controllen = sizeof(control),
  };
  struct cmsghdr* item = CMSG_FIRSTHDR(&message);

  item->cmsg_level = SOL_SOCKET;
  item->cmsg_type = SCM_RIGHTS;
  item->cmsg_len = CMSG_LEN(sizeof(fd));
  memcpy(CMSG_DATA(item), &fd, sizeof(fd));
  require(sendmsg(socket_fd, &message, 0) == 1, "cannot send application capability");
}

static int
receive_application_fd(int socket_fd)
{
  char control[CMSG_SPACE(sizeof(int))] = {0};
  char byte;
  struct iovec vector = {.iov_base = &byte, .iov_len = sizeof(byte)};
  struct msghdr message = {
      .msg_iov = &vector,
      .msg_iovlen = 1,
      .msg_control = control,
      .msg_controllen = sizeof(control),
  };
  struct cmsghdr* item;
  int fd = -1;

  require(recvmsg(socket_fd, &message, MSG_CMSG_CLOEXEC) == 1, "cannot receive application capability");
  for (item = CMSG_FIRSTHDR(&message); item != NULL; item = CMSG_NXTHDR(&message, item)) {
    if (item->cmsg_level == SOL_SOCKET && item->cmsg_type == SCM_RIGHTS)
      memcpy(&fd, CMSG_DATA(item), sizeof(fd));
  }
  require(fd >= 0, "application capability transport omitted FD");
  return fd;
}

static int
matching_fd_count(int reference_fd)
{
  struct stat reference;
  struct stat candidate;
  char path[64];
  int count = 0;

  require(fstat(reference_fd, &reference) == 0,
          "cannot inspect reference broker FD");
  for (int fd = 0; fd < 4096; fd++) {
    if (fd == reference_fd) {
      count++;
      continue;
    }
    snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
    if (stat(path, &candidate) == 0 &&
        candidate.st_dev == reference.st_dev &&
        candidate.st_ino == reference.st_ino)
      count++;
  }
  return count;
}

static int
connect_control(void)
{
  const char* control = getenv("DYN_SNAPSHOT_CONTROL_DIR");
  struct sockaddr_un address = {.sun_family = AF_UNIX};
  int client = socket(AF_UNIX, SOCK_STREAM, 0);

  require(client >= 0, "cannot create VMM control client");
  require(
      snprintf(
          address.sun_path, sizeof(address.sun_path), "%s/%s%d.sock", control,
          DYN_VMM_SOCKET_PREFIX, getpid()) < (int)sizeof(address.sun_path),
      "VMM control socket path is too long");
  require(
      connect(client, (const struct sockaddr*)&address, sizeof(address)) == 0,
      "cannot connect to VMM control socket");
  return client;
}

static struct dyn_vmm_header
exchange_fd(
    const struct dyn_vmm_header* request, const void* payload,
    void** response_payload, int passed_fd, int* received_fd,
    int require_success)
{
  char request_control[CMSG_SPACE(sizeof(int))] = {0};
  char response_control[CMSG_SPACE(sizeof(int))] = {0};
  struct iovec request_vector = {
      .iov_base = (void*)request,
      .iov_len = sizeof(*request),
  };
  struct msghdr request_message = {
      .msg_iov = &request_vector,
      .msg_iovlen = 1,
  };
  struct dyn_vmm_header response;
  struct iovec vector = {.iov_base = &response, .iov_len = sizeof(response)};
  struct msghdr message = {
      .msg_iov = &vector,
      .msg_iovlen = 1,
      .msg_control = response_control,
      .msg_controllen = sizeof(response_control),
  };
  struct cmsghdr* item;
  int client = connect_control();

  if (passed_fd >= 0) {
    struct cmsghdr* passed;
    request_message.msg_control = request_control;
    request_message.msg_controllen = sizeof(request_control);
    passed = CMSG_FIRSTHDR(&request_message);
    passed->cmsg_level = SOL_SOCKET;
    passed->cmsg_type = SCM_RIGHTS;
    passed->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(passed), &passed_fd, sizeof(passed_fd));
  }
  require(
      sendmsg(client, &request_message, 0) == (ssize_t)sizeof(*request),
      "cannot send VMM request header");
  if (request->payload_size != 0)
    require(
        write(client, payload, (size_t)request->payload_size) ==
            (ssize_t)request->payload_size,
        "cannot write VMM request payload");
  require(
      recvmsg(client, &message, MSG_WAITALL) == (ssize_t)sizeof(response),
      "cannot read VMM response header");
  *received_fd = -1;
  for (item = CMSG_FIRSTHDR(&message); item != NULL;
       item = CMSG_NXTHDR(&message, item)) {
    if (item->cmsg_level == SOL_SOCKET && item->cmsg_type == SCM_RIGHTS)
      memcpy(received_fd, CMSG_DATA(item), sizeof(*received_fd));
  }
  *response_payload = NULL;
  if (response.payload_size != 0) {
    *response_payload = malloc((size_t)response.payload_size);
    require(*response_payload != NULL, "cannot allocate VMM response payload");
    require(
        read(client, *response_payload, (size_t)response.payload_size) ==
            (ssize_t)response.payload_size,
        "cannot read VMM response payload");
  }
  close(client);
  if (require_success)
    require(response.status == 0, response.message);
  return response;
}

static struct dyn_vmm_header
exchange(
    const struct dyn_vmm_header* request, const void* payload,
    void** response_payload, int* received_fd)
{
  return exchange_fd(
      request, payload, response_payload, -1, received_fd, 1);
}

static void
establish_mapping(
    CUmemGenericAllocationHandle* logical, int* export_fd,
    CUdeviceptr address)
{
  CUmemAllocationProp properties;
  CUmemAccessDesc access;

  memset(&properties, 0, sizeof(properties));
  properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  properties.location.id = 0;
  properties.requestedHandleTypes =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  memset(&access, 0, sizeof(access));
  access.location = properties.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  require(
      cuMemCreate(logical, 4096, &properties, 0) == CUDA_SUCCESS &&
          cuMemMap(address, 4096, 0, *logical, 0) == CUDA_SUCCESS &&
          cuMemSetAccess(address, 4096, &access, 1) == CUDA_SUCCESS &&
          cuMemExportToShareableHandle(
              export_fd, *logical,
              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) == CUDA_SUCCESS,
      "cannot establish managed mapping");
}

static void
test_access_shape(const char* shape)
{
  CUmemAllocationProp properties;
  CUmemAccessDesc access[2];
  CUmemGenericAllocationHandle logical;
  struct dyn_vmm_header request = {
      .magic = DYN_VMM_MAGIC,
      .version = DYN_VMM_VERSION,
      .operation = DYN_VMM_INSPECT,
  };
  struct dyn_vmm_header response;
  void* payload;
  int received_fd;
  int export_fd;
  size_t count = strcmp(shape, "multi-device") == 0 ? 2 : 1;
  int supported = strcmp(shape, "supported") == 0;

  memset(&properties, 0, sizeof(properties));
  properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  properties.location.id = 0;
  properties.requestedHandleTypes =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  memset(access, 0, sizeof(access));
  access[0].location = properties.location;
  access[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  access[1] = access[0];
  if (strcmp(shape, "multi-device") == 0)
    access[1].location.id = 1;
  else if (strcmp(shape, "no-access") == 0)
    access[0].flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
  if (strcmp(shape, "host") == 0)
    access[0].location.type = CU_MEM_LOCATION_TYPE_HOST;
  else if (strcmp(shape, "other-device") == 0)
    access[0].location.id = 1;
  else
    require(
        supported || strcmp(shape, "multi-device") == 0 ||
            strcmp(shape, "no-access") == 0,
        "unknown access-shape test");

  require(
      cuMemCreate(&logical, 4096, &properties, 0) == CUDA_SUCCESS &&
          cuMemMap(0x100000, 4096, 0, logical, 0) == CUDA_SUCCESS &&
          cuMemSetAccess(0x100000, 4096, access, count) == CUDA_SUCCESS &&
          cuMemExportToShareableHandle(
              &export_fd, logical,
              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) == CUDA_SUCCESS,
      "cannot establish access-shape mapping");
  close(export_fd);
  response = exchange_fd(
      &request, NULL, &payload, -1, &received_fd, 0);
  require(
      received_fd < 0 && ((response.status == 0) == supported),
      supported
          ? "supported same-GPU access descriptor was rejected"
          : "unsupported access descriptor shape was admitted");
  if (supported) {
    const struct dyn_vmm_record* record = payload;
    const CUmemAccessDesc* recorded_access =
        (const CUmemAccessDesc*)((const char*)payload + sizeof(*record) +
                                record->properties_size);

    require(
        response.count == 1 && record->access_count == 1 &&
            record->access_size == sizeof(*recorded_access) &&
            recorded_access->location.type == CU_MEM_LOCATION_TYPE_DEVICE &&
            recorded_access->location.id == 0 &&
            recorded_access->flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        "single-mapping inspect omitted the recorded access");
  }
  if (!supported) {
    struct dyn_vmm_header identify = {
        .magic = DYN_VMM_MAGIC,
        .version = DYN_VMM_VERSION,
        .operation = DYN_VMM_IDENTIFY,
    };
    void* ignored;

    require(
        strstr(response.message, "unsupported managed access") != NULL &&
            fake_cuda_release_count() == 0 &&
            fake_cuda_active_handle_count() == 1 &&
            fake_cuda_active_mapping_count() == 1,
        "unsupported access shape was not rejected before detach");
    (void)exchange(&identify, NULL, &ignored, &received_fd);
  }
  free(payload);
}

static size_t
recorded_access(
    CUdeviceptr address, CUmemAccessDesc* access, size_t capacity)
{
  typedef size_t (*function_type)(
      CUdeviceptr, size_t, CUmemAccessDesc*, size_t);
  function_type function =
      (function_type)dlsym(RTLD_DEFAULT, "dyn_vmm_test_access");

  require(function != NULL, "cannot resolve access metadata test seam");
  return function(address, 4096, access, capacity);
}

static void
require_identify_failure(const char* message)
{
  struct dyn_vmm_header request = {
      .magic = DYN_VMM_MAGIC,
      .version = DYN_VMM_VERSION,
      .operation = DYN_VMM_IDENTIFY,
  };
  struct dyn_vmm_header response;
  void* payload;
  int received_fd;

  response = exchange_fd(
      &request, NULL, &payload, -1, &received_fd, 0);
  require(
      response.status != 0 && received_fd < 0 &&
          strstr(response.message, message) != NULL,
      "unsupported combined access shape did not poison the shim");
  free(payload);
}

static void
test_access_range(const char* shape)
{
  const CUdeviceptr first_address = 0x100000;
  CUdeviceptr second_address =
      strcmp(shape, "gap") == 0 ? 0x102000 : 0x101000;
  CUmemAllocationProp properties;
  CUmemAccessDesc access;
  CUmemGenericAllocationHandle first;
  CUmemGenericAllocationHandle second;
  size_t access_size =
      strcmp(shape, "partial") == 0 ? 6144 :
      strcmp(shape, "gap") == 0 ? 12288 : 8192;

  memset(&properties, 0, sizeof(properties));
  properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  properties.location.id = 0;
  properties.requestedHandleTypes =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  memset(&access, 0, sizeof(access));
  access.location = properties.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  require(
      strcmp(shape, "combined") == 0 ||
          strcmp(shape, "partial") == 0 ||
          strcmp(shape, "gap") == 0 ||
          strcmp(shape, "repeated") == 0 ||
          strcmp(shape, "allocation-failure") == 0,
      "unknown combined access-range test");
  require(
      cuMemCreate(&first, 4096, &properties, 0) == CUDA_SUCCESS &&
          cuMemCreate(&second, 4096, &properties, 0) == CUDA_SUCCESS &&
          cuMemMap(first_address, 4096, 0, first, 0) == CUDA_SUCCESS &&
          cuMemMap(second_address, 4096, 0, second, 0) == CUDA_SUCCESS,
      "cannot establish combined access mappings");
  if (strcmp(shape, "allocation-failure") == 0)
    require(
        setenv(
            "DYN_SNAPSHOT_CUDA_VMM_FAIL_STAGE",
            "access-copy-second", 1) == 0,
        "cannot configure access metadata allocation failure");
  require(
      cuMemSetAccess(
          first_address, access_size, &access, 1) == CUDA_SUCCESS,
      "fake CUDA rejected combined access update");
  if (strcmp(shape, "repeated") == 0) {
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
    require(
        cuMemSetAccess(
            first_address, access_size, &access, 1) == CUDA_SUCCESS,
        "fake CUDA rejected repeated combined access update");
  }

  if (strcmp(shape, "combined") == 0) {
    struct dyn_vmm_header request = {
        .magic = DYN_VMM_MAGIC,
        .version = DYN_VMM_VERSION,
        .operation = DYN_VMM_INSPECT,
    };
    struct dyn_vmm_header response;
    char* cursor;
    void* payload;
    int received_fd;
    int first_fd;
    int second_fd;
    bool saw_first = false;
    bool saw_second = false;

    require(
        cuMemExportToShareableHandle(
            &first_fd, first, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            0) == CUDA_SUCCESS &&
            cuMemExportToShareableHandle(
                &second_fd, second,
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) ==
                CUDA_SUCCESS,
        "cannot export combined access mappings");
    close(first_fd);
    close(second_fd);
    response = exchange(&request, NULL, &payload, &received_fd);
    require(
        response.count == 2 && received_fd < 0,
        "combined access inspect omitted a mapping");
    cursor = payload;
    for (uint32_t index = 0; index < response.count; index++) {
      const struct dyn_vmm_record* record =
          (const struct dyn_vmm_record*)cursor;
      const CUmemAccessDesc* recorded_access =
          (const CUmemAccessDesc*)(
              cursor + sizeof(*record) + record->properties_size);

      require(
          record->access_count == 1 &&
              record->access_size == sizeof(*recorded_access) &&
              recorded_access->location.type ==
                  CU_MEM_LOCATION_TYPE_DEVICE &&
              recorded_access->location.id == 0 &&
              recorded_access->flags ==
                  CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
          "combined access inspect omitted recorded access");
      saw_first |= record->address == first_address;
      saw_second |= record->address == second_address;
      cursor += sizeof(*record) + record->properties_size +
          (size_t)record->access_count * record->access_size;
    }
    require(
        saw_first && saw_second,
        "combined access inspect returned unexpected mappings");
    free(payload);
    return;
  }

  if (strcmp(shape, "repeated") == 0) {
    CUmemAccessDesc first_access;
    CUmemAccessDesc second_access;
    size_t first_count =
        recorded_access(first_address, &first_access, 1);
    size_t second_count =
        recorded_access(second_address, &second_access, 1);

    require(
        first_count == 1 && second_count == 1 &&
            first_access.location.type == CU_MEM_LOCATION_TYPE_DEVICE &&
            first_access.location.id == 0 &&
            first_access.flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE &&
            second_access.location.type == CU_MEM_LOCATION_TYPE_DEVICE &&
            second_access.location.id == 0 &&
            second_access.flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        "repeated combined access update changed recorded metadata");
    require_identify_failure("multiple access updates");
  } else {
    require(
        recorded_access(first_address, NULL, 0) == 0 &&
            recorded_access(second_address, NULL, 0) == 0,
        "failed combined access update partially recorded metadata");
    require_identify_failure(
        strcmp(shape, "allocation-failure") == 0
            ? "cannot record cuMemSetAccess descriptors"
            : "not an exact contiguous union");
  }
}

static void
detach_record(
    const struct dyn_vmm_record* record, uint16_t operation,
    uint64_t object_id)
{
  struct dyn_vmm_header request = {
      .magic = DYN_VMM_MAGIC,
      .version = DYN_VMM_VERSION,
      .operation = operation,
      .object_id = object_id,
  };
  void* ignored;
  int received_fd;

  memcpy(request.allocation_uuid, record->allocation_uuid, sizeof(request.allocation_uuid));
  (void)exchange(&request, NULL, &ignored, &received_fd);
}

static void*
restore_payload(
    const struct dyn_vmm_record* record, const void* inspect_payload,
    size_t* payload_size, uint64_t object_id, int with_contents)
{
  size_t metadata_size = sizeof(*record) + record->properties_size +
      (size_t)record->access_count * record->access_size;
  void* payload;

  *payload_size = metadata_size + (with_contents ? (size_t)record->size : 0);
  payload = calloc(1, *payload_size);
  require(payload != NULL, "cannot allocate restore payload");
  memcpy(payload, inspect_payload, metadata_size);
  ((struct dyn_vmm_record*)payload)->object_id = object_id;
  if (with_contents)
    memset((char*)payload + metadata_size, 0x5a, (size_t)record->size);
  return payload;
}

static void
test_owner_restore_failure(const char* stage)
{
  CUmemGenericAllocationHandle logical;
  struct dyn_vmm_header request = {
      .magic = DYN_VMM_MAGIC,
      .version = DYN_VMM_VERSION,
      .operation = DYN_VMM_INSPECT,
  };
  struct dyn_vmm_header response;
  struct dyn_vmm_record* record;
  void* inspect_payload;
  void* ignored;
  void* payload;
  size_t payload_size;
  int received_fd;
  int export_fd;

  establish_mapping(&logical, &export_fd, 0x100000);
  close(export_fd);
  response = exchange(&request, NULL, &inspect_payload, &received_fd);
  require(response.count == 1, "owner failure inspect did not return one record");
  record = inspect_payload;
  detach_record(record, DYN_VMM_DETACH_OWNERS, 1);
  payload = restore_payload(record, inspect_payload, &payload_size, 1, 1);
  require(setenv("DYN_SNAPSHOT_CUDA_VMM_FAIL_STAGE", stage, 1) == 0,
          "cannot configure owner restore failure");
  memset(&request, 0, sizeof(request));
  request.magic = DYN_VMM_MAGIC;
  request.version = DYN_VMM_VERSION;
  request.operation = DYN_VMM_RESTORE_OWNER;
  request.object_id = 1;
  request.payload_size = payload_size;
  response = exchange_fd(
      &request, payload, &ignored, -1, &received_fd, 0);
  require(
      response.status != 0 && received_fd < 0 &&
          fake_cuda_active_handle_count() == 0 &&
          fake_cuda_active_mapping_count() == 0 &&
          fake_cuda_last_context() == (CUcontext)(uintptr_t)0x77,
      "owner restore failure leaked resources or prior context");
  if (strcmp(stage, "owner-export") == 0 ||
      strcmp(stage, "owner-rebind") == 0) {
    errno = 0;
    require(
        fcntl(fake_cuda_last_exported_fd(), F_GETFD) == -1 &&
            errno == EBADF,
        "owner restore failure leaked exported FD");
  }
  free(payload);
  free(inspect_payload);
}

static void
test_importer_restore_failure(const char* stage)
{
  CUmemGenericAllocationHandle owner;
  CUmemGenericAllocationHandle importer;
  struct dyn_vmm_header request = {
      .magic = DYN_VMM_MAGIC,
      .version = DYN_VMM_VERSION,
      .operation = DYN_VMM_INSPECT,
  };
  struct dyn_vmm_header response;
  struct dyn_vmm_record* owner_record = NULL;
  struct dyn_vmm_record* importer_record = NULL;
  char* cursor;
  void* inspect_payload;
  void* ignored;
  void* owner_payload;
  void* importer_payload;
  size_t owner_payload_size;
  size_t importer_payload_size;
  size_t record_size;
  int owner_restore_fd;
  int received_fd;
  int export_fd;
  int import_fd;
  int broker_fd_count;

  establish_mapping(&owner, &export_fd, 0x100000);
  import_fd = dup(export_fd);
  require(
      import_fd >= 0 &&
          cuMemImportFromShareableHandle(
              &importer, (void*)(uintptr_t)import_fd,
              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == CUDA_SUCCESS,
      "cannot import managed mapping");
  close(import_fd);
  close(export_fd);
  {
    CUmemAccessDesc access;
    memset(&access, 0, sizeof(access));
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = 0;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    require(
        cuMemMap(0x200000, 4096, 0, importer, 0) == CUDA_SUCCESS &&
            cuMemSetAccess(0x200000, 4096, &access, 1) == CUDA_SUCCESS,
        "cannot map importer");
  }
  response = exchange(&request, NULL, &inspect_payload, &received_fd);
  require(response.count == 2, "importer failure inspect did not return two records");
  cursor = inspect_payload;
  for (uint32_t index = 0; index < response.count; index++) {
    struct dyn_vmm_record* record = (struct dyn_vmm_record*)cursor;
    record_size = sizeof(*record) + record->properties_size +
        (size_t)record->access_count * record->access_size;
    if (record->role == DYN_VMM_OWNER)
      owner_record = record;
    else if (record->role == DYN_VMM_IMPORTER)
      importer_record = record;
    cursor += record_size;
  }
  require(owner_record != NULL && importer_record != NULL,
          "inspect omitted owner or importer");
  detach_record(importer_record, DYN_VMM_DETACH_IMPORTS, 1);
  detach_record(owner_record, DYN_VMM_DETACH_OWNERS, 1);
  owner_payload = restore_payload(
      owner_record, owner_record, &owner_payload_size, 1, 1);
  memset(&request, 0, sizeof(request));
  request.magic = DYN_VMM_MAGIC;
  request.version = DYN_VMM_VERSION;
  request.operation = DYN_VMM_RESTORE_OWNER;
  request.object_id = 1;
  request.payload_size = owner_payload_size;
  (void)exchange(
      &request, owner_payload, &ignored, &owner_restore_fd);
  require(
      owner_restore_fd >= 0 && fake_cuda_active_handle_count() == 1 &&
          fake_cuda_active_mapping_count() == 1,
      "owner restore did not establish importer failure baseline");
  broker_fd_count = matching_fd_count(owner_restore_fd);

  importer_payload = restore_payload(
      importer_record, importer_record, &importer_payload_size, 1, 0);
  require(setenv("DYN_SNAPSHOT_CUDA_VMM_FAIL_STAGE", stage, 1) == 0,
          "cannot configure importer restore failure");
  request.operation = DYN_VMM_RESTORE_IMPORT;
  request.payload_size = importer_payload_size;
  response = exchange_fd(
      &request, importer_payload, &ignored, owner_restore_fd, &received_fd, 0);
  require(
      response.status != 0 && received_fd < 0 &&
          fake_cuda_active_handle_count() == 1 &&
          fake_cuda_active_mapping_count() == 1 &&
          fake_cuda_last_context() == (CUcontext)(uintptr_t)0x77 &&
          matching_fd_count(owner_restore_fd) == broker_fd_count,
      "importer restore failure leaked resources or prior context");
  close(owner_restore_fd);
  free(importer_payload);
  free(owner_payload);
  free(inspect_payload);
}

static void
test_owner_importer_success(void)
{
  CUmemGenericAllocationHandle owner;
  CUmemGenericAllocationHandle importer;
  CUmemGenericAllocationHandle owner_token;
  CUmemGenericAllocationHandle importer_token;
  struct dyn_vmm_header request = {
      .magic = DYN_VMM_MAGIC,
      .version = DYN_VMM_VERSION,
      .operation = DYN_VMM_INSPECT,
  };
  struct dyn_vmm_header response;
  struct dyn_vmm_record* owner_record = NULL;
  struct dyn_vmm_record* importer_record = NULL;
  CUmemAccessDesc owner_final_access;
  CUmemAccessDesc importer_final_access;
  char participant[DYN_VMM_PARTICIPANT_ID_SIZE];
  char* cursor;
  unsigned char* owner_bytes;
  void* inspect_payload;
  void* owner_payload;
  void* importer_payload;
  void* ignored;
  size_t owner_payload_size;
  size_t importer_payload_size;
  size_t record_size;
  int owner_restore_fd;
  int received_fd;
  int export_fd;
  int import_fd;
  int internal_alias;
  int broker_fd_count;

  establish_mapping(&owner, &export_fd, 0x100000);
  internal_alias = fake_cuda_last_internal_export_alias();
  import_fd = dup(export_fd);
  require(
      import_fd >= 0 &&
          cuMemImportFromShareableHandle(
              &importer, (void*)(uintptr_t)import_fd,
              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == CUDA_SUCCESS,
      "cannot establish successful importer");
  close(import_fd);
  close(export_fd);
  require(
      internal_alias >= 0 && matching_fd_count(internal_alias) == 1,
      "application sharing FDs did not close around the fake CUDA internal alias");
  {
    CUmemAccessDesc access;
    memset(&access, 0, sizeof(access));
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = 0;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    require(
        cuMemMap(0x200000, 4096, 0, importer, 0) == CUDA_SUCCESS &&
            cuMemSetAccess(0x200000, 4096, &access, 1) == CUDA_SUCCESS,
        "cannot establish successful importer mapping");
  }
  owner_token = owner;
  importer_token = importer;
  require(
      owner_token != importer_token &&
          ((uint64_t)owner_token & UINT64_C(0xffff000000000000)) ==
              UINT64_C(0xd94d000000000000) &&
          ((uint64_t)importer_token & UINT64_C(0xffff000000000000)) ==
              UINT64_C(0xd94d000000000000),
      "owner/importer did not receive distinct logical tokens");

  response = exchange(&request, NULL, &inspect_payload, &received_fd);
  require(
      response.count == 2 && received_fd < 0 &&
          strlen(response.participant_id) == 32,
      "successful lifecycle inspect did not return both mappings");
  snprintf(participant, sizeof(participant), "%s", response.participant_id);
  cursor = inspect_payload;
  for (uint32_t index = 0; index < response.count; index++) {
    struct dyn_vmm_record* record = (struct dyn_vmm_record*)cursor;
    record_size = sizeof(*record) + record->properties_size +
        (size_t)record->access_count * record->access_size;
    if (record->role == DYN_VMM_OWNER)
      owner_record = record;
    else if (record->role == DYN_VMM_IMPORTER)
      importer_record = record;
    cursor += record_size;
  }
  require(
      owner_record != NULL && importer_record != NULL &&
          owner_record->address == 0x100000 &&
          importer_record->address == 0x200000 &&
          owner_record->flags == DYN_VMM_APPLICATION_HANDLE_LIVE &&
          importer_record->flags == DYN_VMM_APPLICATION_HANDLE_LIVE,
      "successful lifecycle inspect metadata is incomplete");
  request.operation = DYN_VMM_READ_OWNER;
  memcpy(request.allocation_uuid, owner_record->allocation_uuid, sizeof(request.allocation_uuid));
  response = exchange(&request, NULL, (void**)&owner_bytes, &received_fd);
  require(
      response.payload_size == 4096 && received_fd < 0,
      "owner byte capture returned the wrong size");
  for (size_t index = 0; index < response.payload_size; index++)
    require(owner_bytes[index] == 0x5a, "owner byte capture changed contents");

  detach_record(importer_record, DYN_VMM_DETACH_IMPORTS, 1);
  detach_record(owner_record, DYN_VMM_DETACH_OWNERS, 1);
  require(
      fake_cuda_release_count() == 2 &&
          fake_cuda_released_handle(0) == 0x1235 &&
          fake_cuda_released_handle(1) == 0x1234 &&
          fake_cuda_active_handle_count() == 0 &&
          fake_cuda_active_mapping_count() == 0,
      "prepare did not detach importer then owner exactly once");

  owner_payload = restore_payload(
      owner_record, owner_record, &owner_payload_size, 1, 1);
  memcpy(
      (char*)owner_payload + owner_payload_size - owner_record->size,
      owner_bytes,
      (size_t)owner_record->size);
  memset(&request, 0, sizeof(request));
  request.magic = DYN_VMM_MAGIC;
  request.version = DYN_VMM_VERSION;
  request.operation = DYN_VMM_RESTORE_OWNER;
  request.object_id = 1;
  request.payload_size = owner_payload_size;
  (void)exchange(
      &request, owner_payload, &ignored, &owner_restore_fd);
  require(
      owner_restore_fd >= 0 && matching_fd_count(owner_restore_fd) == 2 && fake_cuda_active_handle_count() == 1 &&
          fake_cuda_active_mapping_count() == 1,
      "owner restore did not retain one internal alias and broker FD");
  broker_fd_count = matching_fd_count(owner_restore_fd);

  importer_payload = restore_payload(
      importer_record, importer_record, &importer_payload_size, 1, 0);
  request.operation = DYN_VMM_RESTORE_IMPORT;
  request.payload_size = importer_payload_size;
  (void)exchange_fd(
      &request, importer_payload, &ignored, owner_restore_fd, &received_fd, 1);
  require(
      received_fd < 0 && matching_fd_count(owner_restore_fd) == broker_fd_count &&
          fake_cuda_active_handle_count() == 2 && fake_cuda_active_mapping_count() == 2 &&
          fake_cuda_create_count() == 2 && fake_cuda_import_count() == 2 && fake_cuda_export_count() == 3 &&
          fake_cuda_map_count() == 4 && fake_cuda_map_address(2) == 0x100000 && fake_cuda_map_address(3) == 0x200000,
      "successful owner/importer restore violated VA or ownership invariants");
  require(
      fake_cuda_copy_destination() == 0x100000 &&
          fake_cuda_copy_size() == 4096 &&
          fake_cuda_copy_byte() == 0x5a && fake_cuda_copy_uniform(),
      "owner bytes were not restored exactly at the owner VA");
  owner_final_access = fake_cuda_access_descriptor(4);
  importer_final_access = fake_cuda_access_descriptor(5);
  require(
      fake_cuda_set_access_count() == 6 &&
          fake_cuda_access_address(4) == 0x100000 &&
          fake_cuda_access_count(4) == 1 &&
          owner_final_access.location.type == CU_MEM_LOCATION_TYPE_DEVICE &&
          owner_final_access.location.id == 0 &&
          owner_final_access.flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE &&
          fake_cuda_access_address(5) == 0x200000 &&
          fake_cuda_access_count(5) == 1 &&
          importer_final_access.location.type == CU_MEM_LOCATION_TYPE_DEVICE &&
          importer_final_access.location.id == 0 &&
          importer_final_access.flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
      "successful restore did not replay exact final access");
  close(owner_restore_fd);
  errno = 0;
  require(
      fcntl(owner_restore_fd, F_GETFD) == -1 && errno == EBADF &&
          fake_cuda_logical_forward_count() == 0 &&
          owner == owner_token && importer == importer_token,
      "broker FD or logical-token ownership changed during restore");

  memset(&request, 0, sizeof(request));
  request.magic = DYN_VMM_MAGIC;
  request.version = DYN_VMM_VERSION;
  request.operation = DYN_VMM_IDENTIFY;
  response = exchange(&request, NULL, &ignored, &received_fd);
  require(
      strcmp(response.participant_id, participant) == 0,
      "final shim health changed participant identity");

  require(
      cuMemRelease(owner) == CUDA_SUCCESS &&
          cuMemRelease(importer) == CUDA_SUCCESS &&
          fake_cuda_release_count() == 4 &&
          fake_cuda_released_handle(2) == 0x1236 &&
          fake_cuda_released_handle(3) == 0x1237 &&
          fake_cuda_active_handle_count() == 0 &&
          cuMemUnmap(0x100000, 4096) == CUDA_SUCCESS &&
          cuMemUnmap(0x200000, 4096) == CUDA_SUCCESS &&
          fake_cuda_active_mapping_count() == 0,
      "application release did not clean rebound handles exactly once");
  (void)exchange(&request, NULL, &ignored, &received_fd);

  free(importer_payload);
  free(owner_payload);
  free(owner_bytes);
  free(inspect_payload);
}

static void
test_repeated_export_and_self_import(void)
{
  CUmemGenericAllocationHandle owner;
  CUmemGenericAllocationHandle importer;
  struct dyn_vmm_capability first_capability;
  struct dyn_vmm_capability second_capability;
  struct stat first_status;
  struct stat second_status;
  int first_fd;
  int second_fd;
  int first_raw_alias;
  int imported_raw_fd;

  establish_mapping(&owner, &first_fd, 0x100000);
  first_raw_alias = fake_cuda_internal_export_alias(0);
  require(
      cuMemExportToShareableHandle(&second_fd, owner, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) == CUDA_SUCCESS,
      "repeated owner export failed");
  first_capability = read_capability(first_fd);
  second_capability = read_capability(second_fd);
  require(
      memcmp(
          first_capability.allocation_uuid, second_capability.allocation_uuid,
          sizeof(first_capability.allocation_uuid)) == 0 &&
          fstat(first_fd, &first_status) == 0 && fstat(second_fd, &second_status) == 0 &&
          (first_status.st_dev != second_status.st_dev || first_status.st_ino != second_status.st_ino) &&
          fake_cuda_export_count() == 2 && matching_fd_count(first_raw_alias) == 1,
      "repeated export did not reuse UUID with a fresh sealed token");
  require(
      cuMemImportFromShareableHandle(&importer, (void*)(uintptr_t)first_fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) ==
          CUDA_SUCCESS,
      "same-process capability import failed");
  imported_raw_fd = fake_cuda_last_imported_fd();
  errno = 0;
  require(
      fcntl(first_fd, F_GETFD) >= 0 && fake_cuda_export_count() == 3 && fake_cuda_import_count() == 1 &&
          imported_raw_fd >= 0 && fcntl(imported_raw_fd, F_GETFD) == -1 && errno == EBADF,
      "self-import changed token ownership or retained its raw broker FD");
  close(first_fd);
  close(second_fd);
}

static void
test_colliding_raw_export_identity(void)
{
  CUmemGenericAllocationHandle first;
  CUmemGenericAllocationHandle second;
  struct dyn_vmm_header request = {
      .magic = DYN_VMM_MAGIC,
      .version = DYN_VMM_VERSION,
      .operation = DYN_VMM_INSPECT,
  };
  struct dyn_vmm_header response;
  struct dyn_vmm_record* first_record;
  struct dyn_vmm_record* second_record;
  struct stat first_status;
  struct stat second_status;
  void* payload;
  int first_fd;
  int second_fd;
  int received_fd;

  require(
      setenv("FAKE_CUDA_COLLIDE_EXPORT_IDENTITY", "1", 1) == 0, "cannot configure colliding fake raw export identity");
  establish_mapping(&first, &first_fd, 0x100000);
  establish_mapping(&second, &second_fd, 0x200000);
  require(
      fstat(fake_cuda_internal_export_alias(0), &first_status) == 0 &&
          fstat(fake_cuda_internal_export_alias(1), &second_status) == 0 &&
          first_status.st_dev == second_status.st_dev && first_status.st_ino == second_status.st_ino,
      "fake CUDA raw exports did not collide");
  close(first_fd);
  close(second_fd);
  response = exchange(&request, NULL, &payload, &received_fd);
  require(response.count == 2 && received_fd < 0, "independent owners with colliding raw FDs were merged");
  first_record = payload;
  second_record = (struct dyn_vmm_record*)((char*)payload + sizeof(*first_record) + first_record->properties_size +
                                           (size_t)first_record->access_count * first_record->access_size);
  require(
      memcmp(first_record->allocation_uuid, second_record->allocation_uuid, sizeof(first_record->allocation_uuid)) != 0,
      "independent owners received one allocation UUID");
  free(payload);
}

static void
test_canonical_capability_path(void)
{
  CUmemGenericAllocationHandle owner;
  CUmemGenericAllocationHandle importer;
  struct dyn_vmm_capability capability;
  int valid_fd;
  int copied_fd;

  establish_mapping(&owner, &valid_fd, 0x100000);
  capability = read_capability(valid_fd);
  copied_fd = make_capability(&capability, sizeof(capability), 1);
  require(
      capability_is_valid(copied_fd) &&
          cuMemImportFromShareableHandle(
              &importer, (void*)(uintptr_t)copied_fd,
              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == CUDA_SUCCESS &&
          fcntl(copied_fd, F_GETFD) >= 0 && fake_cuda_import_count() == 1,
      "canonical capability path was rejected or changed caller ownership");
  close(copied_fd);
  close(valid_fd);
}

static void
test_invalid_capability(const char* shape)
{
  CUmemGenericAllocationHandle owner;
  CUmemGenericAllocationHandle importer;
  struct dyn_vmm_capability capability;
  int valid_fd;
  int invalid_fd;
  size_t size = sizeof(capability);
  int sealed = 1;
  int expect_invalid_path = 0;

  establish_mapping(&owner, &valid_fd, 0x100000);
  capability = read_capability(valid_fd);
  if (strcmp(shape, "malformed") == 0) {
    capability.magic ^= 1;
  } else if (strcmp(shape, "unsealed") == 0) {
    sealed = 0;
  } else if (strcmp(shape, "truncated") == 0) {
    size--;
  } else if (strcmp(shape, "zero-uuid") == 0) {
    memset(capability.allocation_uuid, 0, sizeof(capability.allocation_uuid));
  } else if (strcmp(shape, "wrong-participant") == 0) {
    capability.owner_participant_id[0] = capability.owner_participant_id[0] == 'f' ? 'e' : 'f';
  } else if (strcmp(shape, "unavailable-owner") == 0) {
    set_capability_control_path(
        &capability, DYN_VMM_SOCKET_PREFIX "2147483647.sock");
  } else if (strcmp(shape, "relative-path") == 0) {
    expect_invalid_path = 1;
    set_capability_path(
        &capability, DYN_VMM_SOCKET_PREFIX "1.sock");
  } else if (strcmp(shape, "other-directory") == 0) {
    expect_invalid_path = 1;
    set_capability_path(
        &capability, "/alternate-snapshot-control/" DYN_VMM_SOCKET_PREFIX
                     "1.sock");
  } else if (strcmp(shape, "traversal-path") == 0) {
    expect_invalid_path = 1;
    set_capability_control_path(
        &capability, "../control/" DYN_VMM_SOCKET_PREFIX "1.sock");
  } else if (strcmp(shape, "alias-path") == 0) {
    expect_invalid_path = 1;
    set_capability_control_path(
        &capability, "./" DYN_VMM_SOCKET_PREFIX "1.sock");
  } else if (strcmp(shape, "duplicate-separator") == 0) {
    expect_invalid_path = 1;
    set_capability_control_path(
        &capability, "/" DYN_VMM_SOCKET_PREFIX "1.sock");
  } else if (strcmp(shape, "invalid-basename") == 0) {
    expect_invalid_path = 1;
    set_capability_control_path(&capability, "owner-1.sock");
  } else if (strcmp(shape, "zero-pid") == 0) {
    expect_invalid_path = 1;
    set_capability_control_path(
        &capability, DYN_VMM_SOCKET_PREFIX "0.sock");
  } else if (strcmp(shape, "leading-zero-pid") == 0) {
    expect_invalid_path = 1;
    set_capability_control_path(
        &capability, DYN_VMM_SOCKET_PREFIX "01.sock");
  } else if (strcmp(shape, "nondecimal-pid") == 0) {
    expect_invalid_path = 1;
    set_capability_control_path(
        &capability, DYN_VMM_SOCKET_PREFIX "one.sock");
  } else if (strcmp(shape, "signed-pid") == 0) {
    expect_invalid_path = 1;
    set_capability_control_path(
        &capability, DYN_VMM_SOCKET_PREFIX "+1.sock");
  } else if (strcmp(shape, "suffix") == 0) {
    expect_invalid_path = 1;
    set_capability_control_path(
        &capability, DYN_VMM_SOCKET_PREFIX "1.sock.extra");
  } else if (strcmp(shape, "raw") == 0) {
    invalid_fd = dup(fake_cuda_internal_export_alias(0));
    require(invalid_fd >= 0, "cannot duplicate fake raw CUDA export FD");
    goto import;
  } else {
    require(0, "unknown invalid capability shape");
  }
  invalid_fd = make_capability(&capability, size, sealed);
  if (expect_invalid_path)
    require(
        !capability_is_valid(invalid_fd),
        "malformed capability endpoint passed production validation");
  import :require(
      cuMemImportFromShareableHandle(
          &importer, (void*)(uintptr_t)invalid_fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) != CUDA_SUCCESS &&
          fcntl(invalid_fd, F_GETFD) >= 0 && fake_cuda_import_count() == 0,
      "invalid capability did not fail closed or changed caller ownership");
  close(invalid_fd);
  close(valid_fd);
}

static void
test_cross_process_child(int transport)
{
  CUmemGenericAllocationHandle importer;
  int token = receive_application_fd(transport);
  int imported_raw_fd;
  char status = 1;

  require(
      cuMemImportFromShareableHandle(&importer, (void*)(uintptr_t)token, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) ==
          CUDA_SUCCESS,
      "cross-process capability import failed");
  imported_raw_fd = fake_cuda_last_imported_fd();
  errno = 0;
  require(
      fcntl(token, F_GETFD) >= 0 && imported_raw_fd >= 0 && fcntl(imported_raw_fd, F_GETFD) == -1 && errno == EBADF,
      "cross-process import retained raw broker FD or closed token");
  close(token);
  require(write(transport, &status, 1) == 1, "cannot acknowledge capability import");
}

static void
test_cross_process(const char* executable)
{
  CUmemGenericAllocationHandle owner;
  struct dyn_vmm_capability capability;
  char descriptor[32];
  char child_status = 0;
  int transport[2];
  int token;
  int raw_alias;
  int child_wait_status;
  pid_t child;

  require(socketpair(AF_UNIX, SOCK_STREAM, 0, transport) == 0, "cannot create capability transport");
  child = fork();
  require(child >= 0, "cannot fork capability importer");
  if (child == 0) {
    close(transport[0]);
    snprintf(descriptor, sizeof(descriptor), "%d", transport[1]);
    execl(executable, executable, "cross-process-child", descriptor, NULL);
    _exit(127);
  }
  close(transport[1]);
  establish_mapping(&owner, &token, 0x100000);
  capability = read_capability(token);
  require(strcmp(capability.owner_participant_id, "") != 0, "cross-process token omitted owner identity");
  send_application_fd(transport[0], token);
  require(
      read(transport[0], &child_status, 1) == 1 && child_status == 1,
      "cross-process importer did not acknowledge success");
  raw_alias = fake_cuda_internal_export_alias(1);
  require(
      raw_alias >= 0 && matching_fd_count(raw_alias) == 1 && fcntl(token, F_GETFD) >= 0,
      "owner retained raw broker FD or lost application token");
  close(token);
  close(transport[0]);
  require(
      waitpid(child, &child_wait_status, 0) == child && WIFEXITED(child_wait_status) &&
          WEXITSTATUS(child_wait_status) == 0,
      "cross-process importer failed");
}

int
main(int argc, char** argv)
{
  if (argc == 3 && strcmp(argv[1], "cross-process-child") == 0) {
    test_cross_process_child(atoi(argv[2]));
    return 0;
  }
  if (argc == 2 && strcmp(argv[1], "cross-process") == 0) {
    test_cross_process(argv[0]);
    return 0;
  }
  if (argc == 2 && strcmp(argv[1], "capability-self") == 0) {
    test_repeated_export_and_self_import();
    return 0;
  }
  if (argc == 2 && strcmp(argv[1], "colliding-raw-identity") == 0) {
    test_colliding_raw_export_identity();
    return 0;
  }
  if (argc == 2 && strcmp(argv[1], "canonical-capability-path") == 0) {
    test_canonical_capability_path();
    return 0;
  }
  if (argc == 3 && strcmp(argv[1], "invalid-capability") == 0) {
    test_invalid_capability(argv[2]);
    return 0;
  }
  if (argc == 3 && strcmp(argv[1], "owner-failure") == 0) {
    test_owner_restore_failure(argv[2]);
    return 0;
  }
  if (argc == 3 && strcmp(argv[1], "importer-failure") == 0) {
    test_importer_restore_failure(argv[2]);
    return 0;
  }
  if (argc == 3 && strcmp(argv[1], "access-shape") == 0) {
    test_access_shape(argv[2]);
    return 0;
  }
  if (argc == 3 && strcmp(argv[1], "access-range") == 0) {
    test_access_range(argv[2]);
    return 0;
  }
  if (argc == 2 && strcmp(argv[1], "owner-importer-success") == 0) {
    test_owner_importer_success();
    return 0;
  }
  const int application_live = argc == 2 && strcmp(argv[1], "live") == 0;
  CUmemAllocationProp properties;
  CUmemAccessDesc access;
  CUmemGenericAllocationHandle logical;
  CUmemGenericAllocationHandle logical_before_restore;
  struct dyn_vmm_header request = {
      .magic = DYN_VMM_MAGIC,
      .version = DYN_VMM_VERSION,
  };
  struct dyn_vmm_header response;
  struct dyn_vmm_record* record;
  void* inspect_payload;
  void* ignored;
  void* restore_payload;
  size_t metadata_size;
  int received_fd;
  int export_fd;

  require(
      application_live ||
          (argc == 2 && strcmp(argv[1], "released") == 0),
      "usage: lifecycle_test live|released");
  memset(&properties, 0, sizeof(properties));
  properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  properties.location.id = 0;
  properties.requestedHandleTypes =
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  memset(&access, 0, sizeof(access));
  access.location = properties.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  require(
      cuMemCreate(&logical, 4096, &properties, 0) == CUDA_SUCCESS &&
          logical != 0x1234,
      "create did not return a logical handle");
  require(
      cuMemMap(0x100000, 4096, 0, logical, 0) == CUDA_SUCCESS &&
          cuMemSetAccess(0x100000, 4096, &access, 1) == CUDA_SUCCESS &&
          cuMemExportToShareableHandle(
              &export_fd, logical,
              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) == CUDA_SUCCESS,
      "cannot establish owner mapping");
  close(export_fd);
  if (!application_live)
    require(
        cuMemRelease(logical) == CUDA_SUCCESS,
        "cannot release application handle before checkpoint");
  logical_before_restore = logical;

  request.operation = DYN_VMM_INSPECT;
  response = exchange(&request, NULL, &inspect_payload, &received_fd);
  require(
      response.count == 1 && received_fd < 0,
      "inspect did not return one owner record");
  require(
      strlen(response.participant_id) == 32,
      "inspect did not return stable participant identity");
  record = inspect_payload;
  require(
      record->flags ==
          (application_live ? DYN_VMM_APPLICATION_HANDLE_LIVE : 0),
      "inspect recorded the wrong application handle state");
  metadata_size = sizeof(*record) + record->properties_size +
      (size_t)record->access_count * record->access_size;
  require(
      metadata_size == response.payload_size,
      "inspect returned invalid metadata lengths");

  memset(&request, 0, sizeof(request));
  request.magic = DYN_VMM_MAGIC;
  request.version = DYN_VMM_VERSION;
  request.operation = DYN_VMM_DETACH_OWNERS;
  memcpy(request.allocation_uuid, record->allocation_uuid, sizeof(request.allocation_uuid));
  request.object_id = 1;
  (void)exchange(&request, NULL, &ignored, &received_fd);
  require(
      fake_cuda_release_count() == 1,
      "detach did not leave exactly one old real-handle release");

  restore_payload = calloc(1, metadata_size + record->size);
  require(restore_payload != NULL, "cannot allocate owner restore payload");
  memcpy(restore_payload, inspect_payload, metadata_size);
  ((struct dyn_vmm_record*)restore_payload)->object_id = 1;
  memset((char*)restore_payload + metadata_size, 0x5a, (size_t)record->size);
  memset(&request, 0, sizeof(request));
  request.magic = DYN_VMM_MAGIC;
  request.version = DYN_VMM_VERSION;
  request.operation = DYN_VMM_RESTORE_OWNER;
  request.object_id = 1;
  request.payload_size = metadata_size + record->size;
  (void)exchange(&request, restore_payload, &ignored, &received_fd);
  require(
      received_fd >= 0 && fake_cuda_create_count() == 2,
      "restore did not create and export fresh owner backing");
  close(received_fd);
  require(
      logical == logical_before_restore,
      "logical handle changed across prepare and restore");

  if (application_live) {
    require(
        cuMemRelease(logical) == CUDA_SUCCESS &&
            fake_cuda_release_count() == 2 &&
            fake_cuda_last_consumed_handle() == 0x1235,
        "live logical handle did not release rebound real handle once");
  } else {
    require(
        fake_cuda_release_count() == 2 &&
            cuMemRelease(logical) == CUDA_ERROR_INVALID_HANDLE &&
            fake_cuda_release_count() == 2,
        "released logical handle retained a restored application binding");
  }
  require(
      fake_cuda_logical_forward_count() == 0,
      "logical token reached the fake UMD during restore");
  free(restore_payload);
  free(inspect_payload);
  return 0;
}
