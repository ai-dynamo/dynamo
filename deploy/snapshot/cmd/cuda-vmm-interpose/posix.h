/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef SNAPSHOT_CUDA_VMM_POSIX_H
#define SNAPSHOT_CUDA_VMM_POSIX_H

#include <stddef.h>
#include <stdint.h>
#include <sys/un.h>

#include "protocol.h"

#define SNAPSHOT_VMM_POSIX_CAPABILITY_MAGIC 0x44564d43U
#define SNAPSHOT_VMM_POSIX_CAPABILITY_VERSION 2U

struct snapshot_vmm_posix_capability {
  uint32_t magic;
  uint16_t version;
  uint16_t reserved;
  char job_id[SNAPSHOT_VMM_ID_SIZE];
  char creator_participant[SNAPSHOT_VMM_ID_SIZE];
  uint8_t allocation_id[SNAPSHOT_VMM_ALLOCATION_ID_SIZE];
  char creator_endpoint[sizeof(((struct sockaddr_un*)0)->sun_path)];
  uint8_t authorization[SNAPSHOT_VMM_TOKEN_SIZE];
  uint8_t reserved_alignment[2];
  uint32_t resource_kind;
  uint32_t num_devices;
  uint64_t allocation_size;
  uint64_t handle_types;
  uint64_t object_flags;
  uint8_t reserved_identity[8];
};

_Static_assert(sizeof(struct snapshot_vmm_posix_capability) == 256, "VMM POSIX capability layout changed");

/* On success, output receives a capability FD owned by the caller. */
int snapshot_vmm_posix_create_capability(const struct snapshot_vmm_posix_capability* capability, int* output);
/* Reads without taking ownership of fd. */
int snapshot_vmm_posix_read_capability(int fd, struct snapshot_vmm_posix_capability* capability);
/* On success, output receives a raw export FD owned by the caller. */
int snapshot_vmm_posix_request_export(
    const struct snapshot_vmm_posix_capability* capability, int* output, char* error, size_t error_size);

#endif
