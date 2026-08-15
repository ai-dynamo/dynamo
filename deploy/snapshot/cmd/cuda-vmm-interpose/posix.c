/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _GNU_SOURCE

#include "posix.h"

#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include "util.h"

#define EXPORT_TIMEOUT_SECONDS 30

static bool
zero_bytes(const void* value, size_t size)
{
  const uint8_t* bytes = value;
  size_t index;

  for (index = 0; index < size; index++) {
    if (bytes[index] != 0)
      return false;
  }
  return true;
}

int
snapshot_vmm_posix_create_capability(const struct snapshot_vmm_posix_capability* capability, int* output)
{
  const int seals = F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL;
  int fd;

  fd = memfd_create("snapshot-cuda-vmm-capability", MFD_CLOEXEC | MFD_ALLOW_SEALING);
  if (fd < 0 || snapshot_vmm_write_all(fd, capability, sizeof(*capability)) != 0 ||
      fcntl(fd, F_ADD_SEALS, seals) != 0) {
    if (fd >= 0)
      close(fd);
    return -1;
  }
  *output = fd;
  return 0;
}

int
snapshot_vmm_posix_read_capability(int fd, struct snapshot_vmm_posix_capability* capability)
{
  const int seals = F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL;
  struct stat status;

  memset(capability, 0, sizeof(*capability));
  if (fd < 0 || fcntl(fd, F_GET_SEALS) != seals || fstat(fd, &status) != 0 ||
      status.st_size != (off_t)sizeof(*capability) ||
      snapshot_vmm_pread_all(fd, capability, sizeof(*capability)) != 0 ||
      capability->magic != SNAPSHOT_VMM_POSIX_CAPABILITY_MAGIC ||
      capability->version != SNAPSHOT_VMM_POSIX_CAPABILITY_VERSION ||
      !zero_bytes(capability->reserved, sizeof(capability->reserved)) ||
      !snapshot_vmm_is_lower_hex_id(capability->creator_participant) ||
      zero_bytes(capability->allocation_id, sizeof(capability->allocation_id)) ||
      capability->creator_endpoint[0] != '/' ||
      memchr(capability->creator_endpoint, '\0', sizeof(capability->creator_endpoint)) == NULL ||
      zero_bytes(capability->authorization, sizeof(capability->authorization)) ||
      !zero_bytes(capability->reserved_identity, sizeof(capability->reserved_identity)))
    return -1;
  return 0;
}

int
snapshot_vmm_posix_request_export(
    const struct snapshot_vmm_posix_capability* capability, int* output, char* error, size_t error_size)
{
  struct sockaddr_un address = {.sun_family = AF_UNIX};
  struct snapshot_vmm_header request;
  struct snapshot_vmm_header response;
  int client = -1;
  int result = -1;

  *output = -1;
  if (error != NULL && error_size != 0)
    error[0] = '\0';
  memset(&request, 0, sizeof(request));
  request.magic = SNAPSHOT_VMM_MAGIC;
  request.version = SNAPSHOT_VMM_VERSION;
  request.operation = SNAPSHOT_VMM_EXPORT;
  snprintf(request.participant_id, sizeof(request.participant_id), "%s", capability->creator_participant);
  memcpy(request.authorization, capability->authorization, sizeof(request.authorization));
  memcpy(request.allocation_id, capability->allocation_id, sizeof(request.allocation_id));
  snprintf(address.sun_path, sizeof(address.sun_path), "%s", capability->creator_endpoint);
  client = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (client < 0 || snapshot_vmm_set_socket_timeouts(client, EXPORT_TIMEOUT_SECONDS) != 0 ||
      connect(client, (const struct sockaddr*)&address, sizeof(address)) != 0 ||
      snapshot_vmm_send_header(client, &request, -1) != 0 ||
      snapshot_vmm_receive_header(client, &response, output) != 0) {
    if (error != NULL && error_size != 0)
      snprintf(error, error_size, "%s", "cannot contact creator endpoint");
    goto done;
  }
  if (!snapshot_vmm_header_strings_terminated(&response) || response.magic != SNAPSHOT_VMM_MAGIC ||
      response.version != SNAPSHOT_VMM_VERSION || response.operation != SNAPSHOT_VMM_EXPORT || response.count != 0 ||
      response.payload_size != 0 || strcmp(response.participant_id, capability->creator_participant) != 0) {
    if (error != NULL && error_size != 0)
      snprintf(error, error_size, "%s", "invalid creator export response");
    if (*output >= 0) {
      close(*output);
      *output = -1;
    }
    goto done;
  }
  if (response.status != 0) {
    if (error != NULL && error_size != 0) {
      if (response.message[0] != '\0')
        snprintf(error, error_size, "%.*s", (int)sizeof(response.message), response.message);
      else
        snprintf(error, error_size, "%s", "creator export request failed");
    }
    if (*output >= 0) {
      close(*output);
      *output = -1;
    }
    goto done;
  }
  if (memcmp(response.allocation_id, capability->allocation_id, sizeof(response.allocation_id)) != 0 || *output < 0) {
    if (error != NULL && error_size != 0)
      snprintf(error, error_size, "%s", "invalid creator export response");
    if (*output >= 0) {
      close(*output);
      *output = -1;
    }
    goto done;
  }
  result = 0;
done:
  if (client >= 0)
    close(client);
  return result;
}
